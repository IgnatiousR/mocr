from __future__ import annotations

import importlib.util
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import get_config_value
from .model_manager import default_realesrgan_model_path, default_translation_model_path, resolve_project_path


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    detail: str
    action: str = ""

    def as_line(self) -> str:
        line = f"[{self.status}] {self.name}: {self.detail}"
        if self.action and self.status != "OK":
            line += f"\n  Action: {self.action}"
        return line


def module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _path_status(name: str, value: str, required: bool, action: str) -> CheckResult:
    if not value:
        return CheckResult(name, "Missing" if required else "Optional", "path is not configured", action)
    path = resolve_project_path(value)
    if path.exists():
        return CheckResult(name, "OK", str(path))
    return CheckResult(name, "Missing" if required else "Optional", f"path does not exist: {path}", action)


def collect_setup_checks(
    translation_model_path: str | None = None,
    font_path: str | None = None,
    realesrgan_model_path: str | None = None,
    output_dir: str | None = None,
) -> list[CheckResult]:
    translation_model_path = translation_model_path if translation_model_path is not None else get_config_value("TRANSLATION_MODEL_PATH") or str(default_translation_model_path())
    font_path = font_path if font_path is not None else get_config_value("FONT_PATH")
    realesrgan_model_path = realesrgan_model_path if realesrgan_model_path is not None else get_config_value("REALESRGAN_MODEL_PATH") or str(default_realesrgan_model_path())
    output_dir = output_dir if output_dir is not None else get_config_value("OUTPUT_DIR", "outputs")

    checks: list[CheckResult] = []
    version = sys.version_info
    if (version.major, version.minor) in {(3, 10), (3, 11)}:
        checks.append(CheckResult("Python", "OK", sys.version.split()[0]))
    else:
        checks.append(CheckResult("Python", "Missing", sys.version.split()[0], "Use Python 3.10 or 3.11."))

    base_modules = ["gradio", "pydantic", "PIL", "numpy", "cv2", "rich"]
    missing_base = [name for name in base_modules if not module_available(name)]
    checks.append(
        CheckResult(
            "Base packages",
            "OK" if not missing_base else "Missing",
            "installed" if not missing_base else ", ".join(missing_base),
            "" if not missing_base else "python -m pip install -r requirements-base.txt",
        )
    )

    ocr_modules = ["paddle", "paddleocr", "manga_ocr", "torch"]
    missing_ocr = [name for name in ocr_modules if not module_available(name)]
    checks.append(
        CheckResult(
            "OCR packages",
            "OK" if not missing_ocr else "Missing",
            "installed" if not missing_ocr else ", ".join(missing_ocr),
            "" if not missing_ocr else "Install PaddlePaddle CPU, then python -m pip install -r requirements-ocr.txt",
        )
    )

    has_llama = module_available("llama_cpp")
    checks.append(
        CheckResult(
            "Translation package",
            "OK" if has_llama else "Optional",
            "llama-cpp-python installed" if has_llama else "llama-cpp-python is not installed",
            "" if has_llama else "python -m pip install -r requirements-translate.txt",
        )
    )
    checks.append(
        _path_status(
            "Translation model",
            translation_model_path or "",
            required=False,
            action="Download the model with python scripts/download_models.py --translation or place it in models/translation/.",
        )
    )

    checks.append(
        _path_status(
            "Font",
            font_path or "",
            required=False,
            action="Set FONT_PATH in .env or paste a .ttf path in the UI.",
        )
    )

    upscale_modules = ["realesrgan", "basicsr"]
    missing_upscale = [name for name in upscale_modules if not module_available(name)]
    checks.append(
        CheckResult(
            "Upscale packages",
            "OK" if not missing_upscale else "Optional",
            "installed" if not missing_upscale else ", ".join(missing_upscale),
            "" if not missing_upscale else "python -m pip install -r requirements-upscale.txt",
        )
    )
    checks.append(
        _path_status(
            "Real-ESRGAN model",
            realesrgan_model_path or "",
            required=False,
            action="Download it with python scripts/download_models.py --upscale or leave upscaling disabled.",
        )
    )

    out_path = Path(output_dir or "outputs").expanduser()
    checks.append(CheckResult("Output directory", "OK", str(out_path), "The app will create it if needed."))

    usage = shutil.disk_usage(Path.cwd())
    free_gb = usage.free / (1024**3)
    checks.append(CheckResult("Free disk space", "OK" if free_gb >= 5 else "Missing", f"{free_gb:.1f} GB free", "Keep at least 5-10 GB free."))

    ram_check = _ram_check()
    if ram_check:
        checks.append(ram_check)

    return checks


def setup_status_markdown(
    translation_model_path: str = "",
    font_path: str = "",
    realesrgan_model_path: str = "",
    output_dir: str = "",
) -> str:
    checks = collect_setup_checks(
        translation_model_path=translation_model_path or None,
        font_path=font_path or None,
        realesrgan_model_path=realesrgan_model_path or None,
        output_dir=output_dir or None,
    )
    lines = ["### Setup Status"]
    for check in checks:
        lines.append(f"- **{check.status}** `{check.name}`: {check.detail}")
        if check.action and check.status != "OK":
            lines.append(f"  - {check.action}")
    return "\n".join(lines)


def _ram_check() -> CheckResult | None:
    if sys.platform != "win32":
        return None
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        total_gb = status.ullTotalPhys / (1024**3)
        return CheckResult(
            "RAM",
            "OK" if total_gb >= 12 else "Missing",
            f"{total_gb:.1f} GB total",
            "16 GB RAM is recommended for local OCR and translation.",
        )
    except Exception:
        return None


def main() -> None:
    for check in collect_setup_checks():
        print(check.as_line())


if __name__ == "__main__":
    main()
