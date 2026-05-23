# CPU-First Manga Image Translator

A local Python/Gradio app for translating Japanese manga or image text on a normal laptop. It is designed for CPU-first use with about 16 GB RAM. After models are installed, processing stays local.

## What The App Does

1. Detect text regions with PaddleOCR.
2. OCR Japanese crops with Manga OCR.
3. Translate with a local GGUF model through `llama-cpp-python`, or Sugoi V4 through CTranslate2.
4. Let you review/edit OCR and translation text.
5. Clean original text with OpenCV inpainting.
6. Render translated text with Pillow.
7. Optionally upscale the final image with Real-ESRGAN.

Supported input formats: PNG, JPG, JPEG, WEBP.

## Recommended Setup Path

Use Python 3.10 or 3.11. Python 3.11 is recommended on Windows.

Open PowerShell in the project folder. For example, replace `<project-folder>` with the path where you cloned or downloaded this repository:

```powershell
cd "<project-folder>"
```

Create a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the base UI/runtime first:

```powershell
python -m pip install -r requirements-base.txt
```

Run the setup checker:

```powershell
python scripts\check_setup.py
```

At this point the UI can launch, but OCR/translation/upscaling may show as missing until you install those layers and model files.

## Install Layers

### 1. Minimal UI/Demo

Installs the web UI, image loading, OpenCV cleanup, rendering, and setup checker:

```powershell
python -m pip install -r requirements-base.txt
```

### 2. OCR

Install PaddlePaddle CPU first:

```powershell
python -m pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
```

Then install OCR packages:

```powershell
python -m pip install -r requirements-ocr.txt
```

### 3. Local Translation

```powershell
python -m pip install -r requirements-translate.txt
```

If `llama-cpp-python` fails to build on Windows, install Microsoft C++ Build Tools, then retry. Sugoi V4 uses `ctranslate2` and `sentencepiece` instead of `llama-cpp-python`.

### 4. Optional Real-ESRGAN Upscaling

```powershell
python -m pip install -r requirements-upscale.txt
```

This is optional and can be very slow on CPU.

### Full Install

For everything at once:

```powershell
python -m pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
python -m pip install -r requirements.txt
```

## Configure Paths With `.env`

Copy `.env.example` to `.env`:

```powershell
Copy-Item .env.example .env
```

Edit `.env`:

```text
TRANSLATION_BACKEND=llama
TRANSLATION_MODEL_PATH=models/translation/gemma-2-2b-jpn-it-translate-Q4_K_M.gguf
FONT_PATH=C:/Windows/Fonts/arial.ttf
REALESRGAN_MODEL_PATH=models/upscale/RealESRGAN_x4plus_anime_6B.pth
OUTPUT_DIR=outputs
LLAMA_THREADS=4
LLAMA_CONTEXT=2048
TRANSLATION_MODEL_REPO=webbigdata/gemma-2-2b-jpn-it-translate-gguf
TRANSLATION_MODEL_FILE=gemma-2-2b-jpn-it-translate-Q4_K_M.gguf
REALESRGAN_MODEL_URL=https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth
```

UI fields override `.env` values. If a UI field is empty, the app uses `.env`, then built-in defaults.

## Where Models Are Stored

The app uses project-local model folders:

```text
models/
  translation/
  upscale/
```

The app never silently downloads large model files during image processing. If a model is missing, the app tells you to download it first.

OCR libraries such as PaddleOCR and Manga OCR may still use their own package/user cache for internal OCR weights. The app-controlled LLM GGUF and Real-ESRGAN weights live in `models/`.

## Translation Model

The app supports two local translation backends:

- `llama`: loads a local `.gguf` model through `llama-cpp-python`.
- `sugoi`: loads the Sugoi V4 Japanese-to-English CTranslate2 model directory.

### GGUF Models

Recommended GGUF repo:

```text
webbigdata/gemma-2-2b-jpn-it-translate-gguf
```

Use a quantized Q4 GGUF file for a 16 GB RAM laptop.

Default project-local path:

```text
models/translation/gemma-2-2b-jpn-it-translate-Q4_K_M.gguf
```

Recommended small GGUF options:

| Model | Repo | File | Notes |
| --- | --- | --- | --- |
| Gemma 2 2B Japanese translate | `webbigdata/gemma-2-2b-jpn-it-translate-gguf` | `gemma-2-2b-jpn-it-translate-Q4_K_M.gguf` | Current default; small and translation-focused. |
| Qwen2.5 1.5B Instruct | `Qwen/Qwen2.5-1.5B-Instruct-GGUF` | `qwen2.5-1.5b-instruct-q4_k_m.gguf` | Smallest alternative; fastest, but lower quality. |
| Qwen2.5 3B Instruct | `Qwen/Qwen2.5-3B-Instruct-GGUF` | `qwen2.5-3b-instruct-q4_k_m.gguf` | Recommended alternative for CPU use. |
| Qwen2.5 7B Instruct | `Qwen/Qwen2.5-7B-Instruct-GGUF` | `qwen2.5-7b-instruct-q4_k_m.gguf` | Better quality, but slower and heavier on RAM. |

To use a different model, update `.env` with the matching repo, file, and local path:

```text
TRANSLATION_MODEL_REPO=Qwen/Qwen2.5-3B-Instruct-GGUF
TRANSLATION_MODEL_FILE=qwen2.5-3b-instruct-q4_k_m.gguf
TRANSLATION_MODEL_PATH=models/translation/qwen2.5-3b-instruct-q4_k_m.gguf
```

### Sugoi V4 CTranslate2

Sugoi V4 is a small Japanese-to-English NMT model for manga/anime-style text. It is not a GGUF file; it uses a CTranslate2 model directory with SentencePiece tokenizers.

Recommended Sugoi V4 repo:

```text
entai2965/sugoi-v4-ja-en-ctranslate2
```

Use these `.env` values:

```text
TRANSLATION_BACKEND=sugoi
TRANSLATION_MODEL_REPO=entai2965/sugoi-v4-ja-en-ctranslate2
TRANSLATION_MODEL_FILE=
TRANSLATION_MODEL_PATH=models/translation/sugoi-v4-ja-en-ctranslate2
```

Install translation packages first:

```powershell
python -m pip install -r requirements-translate.txt
```

Download with:

```powershell
python scripts\download_models.py --translation
```

Or manually place the GGUF file in `models/translation/` and update `TRANSLATION_MODEL_PATH` if the filename differs.

If auto-translate is enabled and the model is missing, the app reports:

```text
Download translation model first: models/translation/...
```

## Real-ESRGAN Model

Real-ESRGAN is optional. It is used only as final anime upscaling, not text removal.

Set this in `.env` or the UI:

```text
REALESRGAN_MODEL_PATH=models/upscale/RealESRGAN_x4plus_anime_6B.pth
```

Download with:

```powershell
python scripts\download_models.py --upscale
```

Leave `Real-ESRGAN anime upscale` disabled if the model file or packages are missing.

If upscaling is enabled and the model is missing, the app reports:

```text
Download Real-ESRGAN model first: models/upscale/...
```

To download both configured models:

```powershell
python scripts\download_models.py --all
```

## Run

```powershell
python app.py
```

Open:

```text
http://127.0.0.1:7860
```

The app includes collapsed `Models` and `Setup Status` panels. Use `Models` for explicit download buttons, and click `Refresh setup status` after changing `.env` or UI paths.

## Stop

If running in the terminal, press:

```text
Ctrl+C
```

If it was started in the background:

```powershell
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force
```

## Basic Usage

1. Upload one or more images.
2. Open the collapsed `Setup Status` panel if you want to confirm dependencies and model paths.
3. Click `Analyze first image`.
4. Review/edit the table:
   - `enabled`: whether to clean/render the region
   - `source_text`: OCR result
   - `translated_text`: text rendered into the image
   - `notes`: missing setup or processing messages
5. Click `Compose reviewed image`.
6. Download from `Final image`.

For batch mode, upload multiple images and click `Run full batch`. Processing is sequential to reduce RAM spikes.

## Outputs

Outputs are saved under `outputs/` by default:

```text
outputs/
  originals/
  overlays/
  cleaned/
  final/
  batches/
  regions/
```

Filenames include a short hash so same-named files from different folders do not collide.

## CPU Tips

- Start with `LLAMA_THREADS=4`.
- Keep `LLAMA_CONTEXT=2048`.
- Use a Q4 GGUF translation model, or `TRANSLATION_BACKEND=sugoi` for the smaller Sugoi V4 CTranslate2 model.
- Keep Real-ESRGAN disabled unless you are okay with slow CPU upscaling.
- Test one image before running a batch.
- Close memory-heavy apps before large batches.

## Troubleshooting

Run this first:

```powershell
python scripts\check_setup.py
```

`PaddleOCR is not installed`

Install PaddlePaddle CPU, then OCR requirements:

```powershell
python -m pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
python -m pip install -r requirements-ocr.txt
```

`Manga OCR is not installed`

```powershell
python -m pip install -r requirements-ocr.txt
```

`Translation model not found`

Download the model or place it in `models/translation/`:

```powershell
python scripts\download_models.py --translation
```

`llama-cpp-python is not installed`

```powershell
python -m pip install -r requirements-translate.txt
```

`ctranslate2 or sentencepiece is not installed`

```powershell
python -m pip install -r requirements-translate.txt
```

`Real-ESRGAN model path is missing or invalid`

Download the model, place it in `models/upscale/`, or disable upscaling:

```powershell
python scripts\download_models.py --upscale
```

`Walkthrough.svelte` or stale Gradio frontend errors

Install pinned dependencies, restart the app, then hard refresh the browser:

```powershell
python -m pip install -r requirements-base.txt --upgrade
```

Open:

```text
http://127.0.0.1:7860/?fresh=1
```

`localhost is not accessible`

Stop old Python servers and run again:

```powershell
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force
python app.py
```

## Development Checks

```powershell
python -m pytest -q
python -m compileall manga_translator scripts app.py
python scripts\check_setup.py
python scripts\download_models.py --help
```

## Current Limitations

- Detection quality depends on PaddleOCR.
- OCR quality depends on crop quality and Manga OCR.
- Typesetting is basic and may need manual edits.
- OpenCV inpainting is lightweight but not as smart as neural inpainting.
- Real-ESRGAN is optional and slow on CPU.
- The first OCR run may download model files.
