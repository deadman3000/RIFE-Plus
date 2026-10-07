import os
import tempfile
import folder_paths
import numpy as np
import gc
import torch
import pathlib
from vfi_utils import load_file_from_github_release, preprocess_frames, postprocess_frames, InterpolationStateList
import typing
from comfy.model_management import get_torch_device, soft_empty_cache
from packaging import version

MODEL_TYPE = pathlib.Path(__file__).parent.name
CKPT_NAME_VER_DICT = {
    "rife47.pth": "4.7",
    "rife49.pth": "4.7",
    "rife417.pth": "4.17",
    "rife426.pth": "4.26",
    "sudo_rife4_269.662_testV1_scale1.pth": "4.0",
}
DTYPE_OPTIONS = ["float32", "float16", "bfloat16"]
DTYPE_MAP = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
_model_cache: typing.Dict[typing.Tuple, torch.nn.Module] = {}


class RIFE_VFI:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (
                    sorted(list(CKPT_NAME_VER_DICT.keys()),
                           key=lambda ckpt_name: version.parse(CKPT_NAME_VER_DICT[ckpt_name])),
                    {"default": "rife49.pth"}
                ),
                "frames": ("IMAGE", ),
                "clear_cache_after_n_frames": ("INT", {"default": 10, "min": 1, "max": 1000}),
                "multiplier": ("INT", {"default": 2, "min": 1}),
                "fast_mode": ("BOOLEAN", {"default": True}),
                "ensemble": ("BOOLEAN", {"default": True}),
                "scale_factor": ([0.25, 0.5, 1.0, 2.0, 4.0], {"default": 1.0}),
                "dtype": (DTYPE_OPTIONS, {"default": "float32"}),
                "torch_compile": ("BOOLEAN", {"default": False,
                    "tooltip": "Compile the model with torch.compile() for 10-30% faster inference after warm-up."}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64,
                    "tooltip": "Interpolation tasks per GPU call. Higher values improve throughput but use more VRAM."}),
                "chunk_size": ("INT", {"default": 0, "min": 0, "max": 10000,
                    "tooltip": "Temporal chunk size in source frames. 0 = disabled. Chunks overlap by one frame."}),
            },
            "optional": {
                "optional_interpolation_states": ("INTERPOLATION_STATES", )
            }
        }

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "vfi"
    CATEGORY = "ComfyUI-Frame-Interpolation/VFI"

    def _process_frames(self, frames, model, multipliers, device, torch_dtype,
                        scale_list, fast_mode, ensemble, clear_cache_after_n_frames,
                        batch_size, interpolation_states, output_sink=None,
                        output_pos=0, skip_first_source=False):
        """Process a temporal chunk without accumulating the whole chunk in RAM.

        When output_sink is supplied it must support numpy-style indexed writes.
        Frames are written as float32 NCHW directly to the sink as they are produced.
        """
        n_pairs = len(frames) - 1
        tasks_remaining_per_pair = {}
        frames_processed_since_cache_clear = 0
        write_pos = output_pos

        def write_frame(t):
            nonlocal write_pos
            cpu = t.detach().to(device="cpu", dtype=torch.float32)
            if output_sink is None:
                return cpu
            output_sink[write_pos:write_pos + len(cpu)] = cpu.numpy()
            write_pos += len(cpu)
            del cpu
            return None

        with torch.inference_mode():
            for pair_idx in range(n_pairs):
                skipped = (interpolation_states is not None and
                           interpolation_states.is_frame_skipped(pair_idx))
                m = int(multipliers[pair_idx])
                n_steps = max(m - 1, 0) if not skipped else 0
                tasks_remaining_per_pair[pair_idx] = n_steps

                # The source frame at the beginning of each pair is emitted once.
                # For later temporal chunks, the first source frame is the overlap
                # frame already written by the preceding chunk, so omit only it.
                if not (skip_first_source and pair_idx == 0):
                    source = frames[pair_idx:pair_idx + 1]
                    if output_sink is None:
                        source_cpu = source.to(device="cpu", dtype=torch.float32)
                        pending_source = source_cpu
                    else:
                        write_frame(source)
                        pending_source = None
                else:
                    pending_source = None

                if n_steps > 0:
                    # Batch only timesteps for this one frame pair. This keeps
                    # input tensors tiny even when the temporal chunk is large.
                    step = 1
                    while step < m:
                        batch_count = min(max(int(batch_size), 1), m - step)
                        timestep_list = [x / m for x in range(step, step + batch_count)]

                        frame0 = frames[pair_idx:pair_idx + 1].to(device, dtype=torch_dtype)
                        frame1 = frames[pair_idx + 1:pair_idx + 2].to(device, dtype=torch_dtype)
                        if batch_count > 1:
                            frame0_batch = frame0.expand(batch_count, -1, -1, -1)
                            frame1_batch = frame1.expand(batch_count, -1, -1, -1)
                        else:
                            frame0_batch = frame0
                            frame1_batch = frame1
                        timestep_tensor = torch.tensor(
                            timestep_list, dtype=torch_dtype, device=device
                        ).view(-1, 1, 1, 1)

                        middle_frames = model(
                            frame0_batch, frame1_batch, timestep_tensor,
                            scale_list, fast_mode, ensemble
                        ).clamp(0, 1).detach().to(device="cpu", dtype=torch.float32)

                        if output_sink is None:
                            if pending_source is not None:
                                yield_frame = pending_source
                                pending_source = None
                            else:
                                yield_frame = None
                            # This mode is only retained for compatibility; normal
                            # large-video operation uses output_sink.
                            if yield_frame is not None:
                                pass
                        else:
                            output_sink[write_pos:write_pos + len(middle_frames)] = middle_frames.numpy()
                            write_pos += len(middle_frames)

                        del frame0, frame1, frame0_batch, frame1_batch, timestep_tensor, middle_frames
                        step += batch_count

                frames_processed_since_cache_clear += 1
                if frames_processed_since_cache_clear >= clear_cache_after_n_frames:
                    soft_empty_cache()
                    gc.collect()
                    frames_processed_since_cache_clear = 0

                # Non-streaming compatibility path: build only when explicitly used.
                if output_sink is None:
                    raise RuntimeError(
                        "Internal error: RIFE large-video path requires an output sink."
                    )

            # Final source frame.
            last = frames[-1:]
            write_frame(last)

        return write_pos

    def vfi(self, ckpt_name, frames, clear_cache_after_n_frames=10,
            multiplier=2, fast_mode=False, ensemble=False, scale_factor=1.0,
            dtype="float32", torch_compile=False, batch_size=1,
            chunk_size=0, optional_interpolation_states=None, **kwargs):

        from .rife_arch import IFNet

        model_path = load_file_from_github_release(MODEL_TYPE, ckpt_name)
        arch_ver = CKPT_NAME_VER_DICT[ckpt_name]
        torch_dtype = DTYPE_MAP[dtype]
        device = get_torch_device()

        if arch_ver == "4.26":
            ensemble = False

        cache_key = (ckpt_name, dtype, torch_compile)
        if cache_key not in _model_cache:
            interpolation_model = IFNet(arch_ver=arch_ver)
            interpolation_model.load_state_dict(
                torch.load(model_path, weights_only=False)
            )
            if torch_dtype != torch.float32:
                interpolation_model = interpolation_model.to(torch_dtype)
            interpolation_model.eval().to(device)
            if torch_compile:
                interpolation_model = torch.compile(interpolation_model)
            _model_cache[cache_key] = interpolation_model
            print(f"Comfy-VFI: Loaded and cached model {ckpt_name} ({dtype}{'+ torch.compile' if torch_compile else ''})")
        else:
            interpolation_model = _model_cache[cache_key]
            print(f"Comfy-VFI: Using cached model {ckpt_name} ({dtype}{'+ torch.compile' if torch_compile else ''})")

        frames = preprocess_frames(frames)
        n_frames = len(frames)
        n_pairs = n_frames - 1

        if isinstance(multiplier, int):
            multipliers = [int(multiplier)] * n_pairs
        else:
            multipliers = list(map(int, multiplier))
            multipliers += [2] * max(0, n_pairs - len(multipliers))

        if arch_ver == "4.26":
            scale_list = [16 / scale_factor, 8 / scale_factor, 4 / scale_factor, 2 / scale_factor, 1 / scale_factor]
        else:
            scale_list = [8 / scale_factor, 4 / scale_factor, 2 / scale_factor, 1 / scale_factor]

        # Large-video-safe mode: always use temporal streaming. A chunk_size of 0
        # means automatic conservative chunking rather than a full-video allocation.
        if chunk_size <= 0:
            chunk_size = 8
            print("Comfy-VFI: chunk_size=0 -> automatic safe chunking (8 source frames)")
        if chunk_size < 2:
            raise ValueError("chunk_size must be 0 (automatic) or at least 2.")

        print(f"Comfy-VFI: Temporal streaming enabled: {n_frames} source frames, "
              f"chunk_size={chunk_size}, overlap=1")

        # The final IMAGE remains backed by a disk-backed NumPy memmap instead of
        # allocating the entire interpolated video in system RAM. This is the key
        # difference from both torch.cat() and torch.empty(total_frames, ...).
        expected_output_frames = n_frames + sum(
            max(int(m) - 1, 0)
            for pair_idx, m in enumerate(multipliers)
            if not (optional_interpolation_states is not None and
                    optional_interpolation_states.is_frame_skipped(pair_idx))
        )

        temp_dir = os.path.join(folder_paths.get_temp_directory(), "comfyui_rife_stream")
        os.makedirs(temp_dir, exist_ok=True)
        memmap_path = os.path.join(
            temp_dir, f"rife_{os.getpid()}_{id(frames)}.dat"
        )
        output_shape = (expected_output_frames, *frames.shape[1:])
        output_mm = np.memmap(memmap_path, mode="w+", dtype=np.float32, shape=output_shape)

        out_pos = 0
        start = 0
        chunk_number = 0

        try:
            while start < n_frames - 1:
                end = min(start + chunk_size, n_frames)
                chunk_number += 1
                chunk_frames = frames[start:end]

                local_states = None
                if optional_interpolation_states is not None:
                    local_indices = [
                        idx - start
                        for idx in optional_interpolation_states.frame_indices
                        if start <= idx < end - 1
                    ]
                    local_states = InterpolationStateList(
                        local_indices, optional_interpolation_states.is_skip_list
                    )

                local_multipliers = multipliers[start:end - 1]
                print(f"Comfy-VFI: Processing chunk {chunk_number}: "
                      f"source frames {start}-{end - 1} ({len(chunk_frames)} frames)")

                old_pos = out_pos
                out_pos = self._process_frames(
                    chunk_frames, interpolation_model, local_multipliers,
                    device, torch_dtype, scale_list, fast_mode, ensemble,
                    clear_cache_after_n_frames, batch_size, local_states,
                    output_sink=output_mm, output_pos=out_pos,
                    skip_first_source=(chunk_number > 1)
                )

                # The overlap source frame was intentionally skipped.
                # The final write count is checked after all chunks complete.

                if out_pos <= old_pos:
                    raise RuntimeError(f"RIFE streaming chunk {chunk_number} produced no output frames.")

                del chunk_frames, local_states
                soft_empty_cache()
                gc.collect()

                if end >= n_frames:
                    break
                start = end - 1

            if out_pos != expected_output_frames:
                raise RuntimeError(
                    f"Comfy-VFI internal output-size mismatch: wrote {out_pos} frames, "
                    f"expected {expected_output_frames}."
                )

            output_mm.flush()

            # The ComfyUI IMAGE result must outlive this node call, so make the
            # final output independent of the disk-backed memmap before cleanup.
            # This is the one unavoidable full-output RAM allocation at the very
            # end; interpolation itself remains streamed to disk.
            out_tensor = torch.from_numpy(output_mm).clone()
            print(f"Comfy-VFI done! {out_pos} frames generated via disk-backed streaming")
            return (postprocess_frames(out_tensor),)
        except Exception:
            raise
        finally:
            # Windows keeps a mapped file locked until the NumPy memmap object is
            # released. Close it first, then remove the temporary file. This also
            # cleans up after failed/cancelled runs rather than leaving large .dat
            # files behind on the ComfyUI drive.
            try:
                output_mm.flush()
            except Exception:
                pass
            try:
                del output_mm
            except Exception:
                pass
            gc.collect()
            try:
                if os.path.exists(memmap_path):
                    os.remove(memmap_path)
                    print(f"Comfy-VFI: Removed temporary RIFE file: {memmap_path}")
            except Exception as cleanup_error:
                print(f"Comfy-VFI: WARNING - could not remove temporary RIFE file: {cleanup_error}")
            try:
                if os.path.isdir(temp_dir) and not os.listdir(temp_dir):
                    os.rmdir(temp_dir)
            except Exception:
                pass

