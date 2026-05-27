# MOCR Flutter Migration + Manga Translation Roadmap

# PHASE 0 — RESTRUCTURE CURRENT REPO

Goal:
Separate AI logic from Gradio before adding Flutter.

Current likely state:

```text
Gradio UI
    ↔ AI logic mixed together
```

Target:

```text
Flutter UI
    ↔ Python backend
        ↔ AI modules
```

---

# Step 0.1 — Create Clean Backend Structure

Recommended structure:

```text
backend/
├── api/
├── core/
├── models/
├── ocr/
├── translation/
├── segmentation/
├── inpainting/
├── rendering/
├── utils/
├── temp/
└── main.py
```

---

# Step 0.2 — Move Existing Logic

Extract:

| Existing Logic       | Move To             |
| -------------------- | ------------------- |
| PaddleOCR            | backend/ocr         |
| MangaOCR             | backend/ocr         |
| Gemma translation    | backend/translation |
| image utilities      | backend/utils       |
| processing pipelines | backend/core        |

Goal:
NO Gradio imports inside AI modules.

---

# Step 0.3 — Create Central Pipeline

Create:

```text
backend/core/pipeline.py
```

This becomes:

```python
segment()
ocr()
translate()
inpaint()
render()
```

All UI systems call this.

---

# PHASE 1 — REPLACE GRADIO WITH FASTAPI

Goal:
Turn backend into local AI service.

---

# Step 1.1 — Add FastAPI

Install:

```bash
pip install fastapi uvicorn
```

---

# Step 1.2 — Create Endpoints

Initial endpoints:

```text
GET  /health
POST /ocr
POST /translate
POST /process
```

Later:

```text
POST /segment
POST /inpaint
POST /render
```

---

# Step 1.3 — Add WebSocket

Add:

```text
/ws
```

Used for:

* progress updates
* inference state
* streaming translation
* cancellation

---

# Step 1.4 — Backend Startup Lifecycle

On backend launch:

```text
load OCR models
load segmentation model
load Gemma
load LaMa
```

Keep models resident.

NEVER reload models per request.

---

# PHASE 2 — FLUTTER DESKTOP SHELL

Goal:
Basic desktop UI replacing Gradio.

---

# Step 2.1 — Create Flutter Desktop App

Enable:

```bash
flutter config --enable-windows-desktop
```

or:

* macOS
* Linux

---

# Step 2.2 — Flutter Launches Backend

On startup:

```dart
Process.start("backend.exe")
```

Then:

* poll `/health`
* wait until ready

---

# Step 2.3 — Add Local Communication

Flutter:

* HTTP client
* WebSocket client

Communication:

```text
Flutter
    ↔ localhost:8000
```

---

# Step 2.4 — Basic Viewer

Implement:

* image viewer
* zoom
* pan
* page switching
* image loading

Do NOT build editor yet.

---

# PHASE 3 — COMIC TEXT DETECTION

Goal:
Reliable manga text localization.

Current repo likely only has OCR.

You need proper text detection.

---

# Step 3.1 — Add Comic Text Detection

Recommended:

* PaddleOCR detector initially

Later:

* ComicTextDetector
* DBNet
* YOLO text detector

Pipeline:

```text
page
    ↓
text detector
    ↓
text regions
    ↓
MangaOCR
```

---

# Step 3.2 — Store Structured OCR Data

Store:

```json
{
  "text_regions": [],
  "polygons": [],
  "recognized_text": []
}
```

NOT just translated image output.

Important for:

* editing
* overlays
* corrections

---

# PHASE 4 — SPEECH BUBBLE SEGMENTATION

Goal:
Bubble-aware rendering and cleanup.

This is your next major missing system.

---

# Step 4.1 — Start Simple

Initially use:

```text
OpenCV contours
```

Pipeline:

```text
threshold
→ contours
→ candidate bubbles
```

Good enough for early prototype.

---

# Step 4.2 — Add AI Segmentation Later

Recommended:

* YOLO-seg
* SAM
* Mask R-CNN

Output:

```text
bubble masks
polygon boundaries
```

---

# Step 4.3 — Bubble Data Structure

Store:

```json
{
  "bubble_id": 1,
  "mask": "...",
  "polygon": [],
  "text_ids": []
}
```

Critical for:

* layout
* text fitting
* speaker grouping

---

# PHASE 5 — TRANSLATION PIPELINE

Goal:
Context-aware manga translation.

---

# Step 5.1 — Group OCR By Bubble

Avoid:

```text
translate each line independently
```

Instead:

```text
bubble
    ↓
combined dialogue
    ↓
translation
```

Huge quality improvement.

---

# Step 5.2 — Add Translation Memory

Store:

* previous speaker context
* repeated phrases
* names

Improves consistency.

---

# Step 5.3 — Gemma Integration

Recommended:

* llama.cpp
* GGUF quantized model

Use:

* Q4_K_M
* Q5_K_M

for memory balance.

---

# PHASE 6 — INPAINTING

Goal:
Remove Japanese text cleanly.

---

# Step 6.1 — Generate Text Masks

Use:

* OCR polygons
* expanded masks

---

# Step 6.2 — Add LaMa

Pipeline:

```text
original image
+
text mask
    ↓
LaMa
    ↓
clean artwork
```

---

# Step 6.3 — Cache Results

Important.

Cache:

* masks
* cleaned pages
* OCR

Otherwise processing becomes slow.

---

# PHASE 7 — TYPESSETTING ENGINE

Goal:
Professional manga rendering.

This is where apps become “good.”

---

# Step 7.1 — Overlay Rendering

Flutter renders:

* translated text
* editable text layers
* bubble-aware alignment

---

# Step 7.2 — Smart Layout

Implement:

* auto wrapping
* font scaling
* vertical centering
* padding rules

---

# Step 7.3 — Editable Translation Layer

User can:

* click text
* edit translation
* move bubbles
* resize text

This is extremely valuable.

---

# PHASE 8 — PERFORMANCE OPTIMIZATION

Goal:
Production usability.

---

# Step 8.1 — Convert Models To ONNX

Where possible:

* OCR
* segmentation
* LaMa

Benefits:

* faster inference
* easier deployment
* DirectML support

---

# Step 8.2 — Add GPU Backends

Support:

* CUDA
* DirectML

Optional:

* Vulkan

---

# Step 8.3 — Async Processing

Use:

* worker queues
* multiprocessing
* background tasks

Avoid blocking API thread.

---

# PHASE 9 — PACKAGING

Goal:
Single downloadable app.

---

# Step 9.1 — Package Python Backend

Use:

* Nuitka
  or
* PyInstaller

---

# Step 9.2 — Bundle Models

Structure:

```text
app/
├── flutter_ui/
├── backend.exe
├── models/
└── runtime/
```

---

# Step 9.3 — First-Run Setup

Implement:

* model download
* integrity check
* cache setup

---

# RECOMMENDED DEVELOPMENT ORDER

Correct order:

```text
1. Refactor backend
2. FastAPI
3. Flutter shell
4. OCR pipeline
5. Text detection
6. Bubble segmentation
7. Translation improvements
8. Inpainting
9. Typesetting
10. Optimization
11. Packaging
```

---

# DO NOT START WITH

Avoid immediately building:

* realtime translation
* live overlays
* diffusion inpainting
* full editor
* mobile support

Build stable foundations first.

---

# MINIMUM VIABLE PRODUCT (MVP)

Good first milestone:

```text
Load manga page
→ detect text
→ OCR
→ translate
→ render overlay
```

WITHOUT:

* inpainting
* realtime
* editable layers

Then iterate.

---

# YOUR MOST IMPORTANT SYSTEMS

Priority order:

```text
1. OCR accuracy
2. Bubble-aware layout
3. Translation quality
4. Inpainting quality
5. Typography
```

Typography and layout matter far more than most beginners expect.
