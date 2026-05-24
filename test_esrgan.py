import sys
from PIL import Image
from manga_translator.upscale import upscale_anime

img = Image.new("RGB", (100, 100))
try:
    res = upscale_anime(img, "models/upscale/RealESRGAN_x4plus_anime_6B.pth")
    print("Success")
except Exception as e:
    import traceback
    traceback.print_exc()
