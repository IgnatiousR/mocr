from __future__ import annotations

import importlib.util
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import get_config_value
from .model_manager import (
    NEURAL_INPAINTERS,
    default_inpaint_model_path,
    default_realesrgan_model_path,
    default_translation_model_path,
    get_inpainter_backend,
    get_translation_backend,
    normalize_inpainter_backend,
    resolve_project_path,
)


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


def pip_install_command(requirements_file: str) -> str:
    return f'"{sys.executable}" -m pip install -r {requirements_file}'


def missing_modules(modules: list[str]) -> list[str]:
    return [module for module in modules if not module_available(module)]


def translation_dependency_modules(translation_backend: str) -> list[str]:
    if translation_backend in {"ctranslate2", "sugoi", "jparacrawl", "jparacrawl-big"}:
        return ["ctranslate2", "sentencepiece"]
    if translation_backend in {"fugumt", "fugu", "fugu-mt", "fugumt-ja-en", "transformers", "nllb", "m2m100", "mbart"}:
        return ["transformers", "torch"]
    return ["llama_cpp"]


def missing_translation_dependencies(translation_backend: str) -> list[str]:
    return missing_modules(translation_dependency_modules(translation_backend))


def missing_upscale_dependencies() -> list[str]:
    return missing_modules(["realesrgan", "basicsr"])


def missing_inpaint_dependencies(inpainter_backend: str) -> list[str]:
    if normalize_inpainter_backend(inpainter_backend) in NEURAL_INPAINTERS:
        return missing_modules(["torch"])
    return []


def dependency_status_message(name: str, missing: list[str], requirements_file: str) -> str:
    if not missing:
        return ""
    return f"Missing packages for {name}: {', '.join(missing)}. Run: {pip_install_command(requirements_file)}"


def _package_status(name: str, modules: list[str], missing_status: str, action: str) -> CheckResult:
    missing = missing_modules(modules)
    return CheckResult(
        name,
        "OK" if not missing else missing_status,
        "installed" if not missing else ", ".join(missing),
        "" if not missing else action,
    )


def _path_status(name: str, value: str, required: bool, action: str) -> CheckResult:
    if not value:
        return CheckResult(name, "Missing" if required else "Optional", "path is not configured", action)
    path = resolve_project_path(value)
    if path.exists():
        return CheckResult(name, "OK", str(path))
    return CheckResult(name, "Missing" if required else "Optional", f"path does not exist: {path}", action)


def _python_check() -> CheckResult:
    version = sys.version_info
    if (version.major, version.minor) in {(3, 10), (3, 11)}:
        return CheckResult("Python", "OK", sys.version.split()[0])
    return CheckResult("Python", "Missing", sys.version.split()[0], "Use Python 3.10 or 3.11.")


def _translation_package_check(translation_backend: str) -> CheckResult:
    modules = translation_dependency_modules(translation_backend)
    missing = missing_modules(modules)

    return CheckResult(
        "Translation package",
        "OK" if not missing else "Optional",
        "installed" if not missing else ", ".join(missing),
        "" if not missing else pip_install_command("requirements-translate.txt"),
    )


def collect_setup_checks(
    translation_backend: str | None = None,
    translation_model_path: str | None = None,
    font_path: str | None = None,
    realesrgan_model_path: str | None = None,
    inpainter_backend: str | None = None,
    inpaint_model_path: str | None = None,
    output_dir: str | None = None,
) -> list[CheckResult]:
    translation_backend = (translation_backend if translation_backend is not None else get_translation_backend()).strip().lower()
    translation_model_path = translation_model_path if translation_model_path is not None else get_config_value("TRANSLATION_MODEL_PATH") or str(default_translation_model_path(translation_backend))
    font_path = font_path if font_path is not None else get_config_value("FONT_PATH")
    realesrgan_model_path = realesrgan_model_path if realesrgan_model_path is not None else get_config_value("REALESRGAN_MODEL_PATH") or str(default_realesrgan_model_path())
    inpainter_backend = normalize_inpainter_backend(inpainter_backend if inpainter_backend is not None else get_inpainter_backend())
    inpaint_model_path = inpaint_model_path if inpaint_model_path is not None else get_config_value("INPAINT_MODEL_PATH") or str(default_inpaint_model_path(inpainter_backend))
    output_dir = output_dir if output_dir is not None else get_config_value("OUTPUT_DIR", "outputs")

    checks: list[CheckResult] = []
    checks.append(_python_check())
    checks.append(
        _package_status(
            "Base packages",
            ["gradio", "pydantic", "PIL", "numpy", "cv2", "psutil", "rich"],
            "Missing",
            pip_install_command("requirements-base.txt"),
        )
    )
    checks.append(
        _package_status(
            "OCR packages",
            ["paddle", "paddleocr", "manga_ocr", "torch"],
            "Missing",
            f"Install PaddlePaddle CPU, then {pip_install_command('requirements-ocr.txt')}",
        )
    )
    checks.append(_translation_package_check(translation_backend))
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

    checks.append(
        _package_status(
            "Upscale packages",
            ["realesrgan", "basicsr"],
            "Optional",
            pip_install_command("requirements-upscale.txt"),
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

    checks.append(CheckResult("Inpainter", "OK", inpainter_backend))
    if inpainter_backend in NEURAL_INPAINTERS:
        checks.append(
            _package_status(
                "Inpaint packages",
                ["torch"],
                "Optional",
                pip_install_command("requirements-inpaint.txt"),
            )
        )
        checks.append(
            _path_status(
                "Inpaint model",
                inpaint_model_path or "",
                required=False,
                action="Download it with python scripts/download_models.py --inpaint or use the UI download button.",
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
    translation_backend: str = "",
    translation_model_path: str = "",
    font_path: str = "",
    realesrgan_model_path: str = "",
    inpainter_backend: str = "",
    inpaint_model_path: str = "",
    output_dir: str = "",
) -> str:
    checks = collect_setup_checks(
        translation_backend=translation_backend or None,
        translation_model_path=translation_model_path or None,
        font_path=font_path or None,
        realesrgan_model_path=realesrgan_model_path or None,
        inpainter_backend=inpainter_backend or None,
        inpaint_model_path=inpaint_model_path or None,
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
