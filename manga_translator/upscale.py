from __future__ import annotations

from PIL import Image


def upscale_anime(image: Image.Image, model_path: str, scale: int = 4) -> Image.Image:
    try:
        import numpy as np
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
    except Exception as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError("Real-ESRGAN dependencies are not installed.") from exc

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=scale)
    upsampler = RealESRGANer(
        scale=scale,
        model_path=model_path,
        model=model,
        tile=256,
        tile_pad=10,
        pre_pad=0,
        half=False,
        gpu_id=None,
    )
    output, _ = upsampler.enhance(np.array(image.convert("RGB")), outscale=scale)
    return Image.fromarray(output)
