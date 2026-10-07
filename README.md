This is a version of RIFE for the ComfyUI Frame Interpolation node. 
It adds forward feed chunking to numpy disk pairs so that there is less of a restriction on VRAM preventing OOM's and memory overspills. 
Each chunk is stored temporarily on disk and concatonated to the output.
Tested on a 16GB GPU using a 15 second 24 fps video at 2K with a 5x multiplier then sent to a 2x nth node to output 60 fpos interpolated MP4.

To install backup your existing vfi_utils.py file in custom_nodes\comfyui-frame-interpolation folder and copy the new file there. In the subfolder \vfi_models\rife backup __init__.py and copy the new file. Restart ComfyUI.
Try 32 chunks to begin with if you have a 16GB GPU. Other lower VRAM GPU's may need to lower this value.

This is a test version and may not work as intended for everyone. It is provided on as as is basis. Your mileage may vary.
