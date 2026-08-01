# Manga AI Bot

Single CLI-ready package for generating manhwa-style panels and chapters.

## Install (editable)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

Or install via requirements (runtime only) and add repo to `PYTHONPATH`:

```bash
pip install -r requirements.txt
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
```

## Run

- Module:

```bash
python -m manga_ai --panels 8 --num-steps 25 --export
```

- Console script (after `pip install -e .`):

```bash
manga-ai --panels 8 --num-steps 25 --export
```

## Optional tokens

```bash
export HF_TOKEN=your_token
# or
export OPENAI_API_KEY=your_token
```

Outputs land in `output/`.

## Use a reference screenshot

To keep generated panels close to an existing screenshot's layout, use img2img reference conditioning:

```bash
python -m manga_ai \
  --reference-image data/reference_dialog.png \
  --img2img-strength 0.35 \
  --reference-resize-mode fit \
  --genre drama \
  --setting "two people talking in a quiet classroom after school" \
  --tone "calm emotional conversation" \
  --panels 4 \
  --no-randomize \
  --export
```

Equivalent `.env` settings:

```bash
REFERENCE_IMAGE=data/reference_dialog.png
IMG2IMG_STRENGTH=0.35
REFERENCE_RESIZE_MODE=fit
```

Use lower `IMG2IMG_STRENGTH` values, such as `0.25`-`0.4`, to keep the screenshot layout closer. Use higher values, such as `0.5`-`0.7`, when you want more visible changes.

## Use Blender as composition control

To make the bot generate a rough 3D composition per panel before diffusion, enable Blender control:

```bash
python -m manga_ai \
  --use-blender-control \
  --blender-control-strength 0.34 \
  --genre drama \
  --setting "room with a table and window" \
  --protagonist Kai \
  --antagonist Maria \
  --panels 4 \
  --export
```

This writes per-panel control images and scene metadata to `output/control/`, then uses each control image as the img2img layout reference for the final manga panel. Blender is responsible for rough geometry, camera, character placement, and props; diffusion is still responsible for the final manga style.

Environment equivalents:

```bash
USE_BLENDER_CONTROL=1
BLENDER_EXECUTABLE=blender
BLENDER_CONTROL_STRENGTH=0.34
BLENDER_CONTROL_WIDTH=768
BLENDER_CONTROL_HEIGHT=1024
```

If Blender is not installed, the bot falls back to a simple PIL layout sketch by default. Disable that with `--no-blender-control-fallback` if you want Blender absence to skip control generation.

## Use an image API instead of local diffusion

To avoid downloading or running a diffusion model on your laptop, use the API backend:

```bash
export OPENAI_API_KEY=your_api_key

PYTHONPATH=src python3 -m manga_ai \
  --image-backend api \
  --image-api-provider openai \
  --image-api-model gpt-image-1 \
  --image-api-size 1024x1024 \
  --image-api-quality medium \
  --use-blender-control \
  --blender-control-strength 0.34 \
  --panels 4 \
  --export
```

In this mode, the laptop only runs story planning, optional Blender/control rendering, bubble placement, and chapter assembly. Final manga artwork is generated through the remote image API.
If you also want remote LLM story generation through the Hugging Face router, set `HF_TOKEN` or `LLM_API_KEY` separately.

Environment equivalents:

```bash
IMAGE_BACKEND=api
IMAGE_API_PROVIDER=openai
IMAGE_API_MODEL=gpt-image-1
IMAGE_API_SIZE=1024x1024
IMAGE_API_QUALITY=medium
OPENAI_API_KEY=your_api_key
```

For a Hugging Face model through Hugging Face Inference Providers:

```bash
export HF_TOKEN=your_hugging_face_token

PYTHONPATH=src python3 -m manga_ai \
  --image-backend api \
  --image-api-provider huggingface \
  --image-api-model black-forest-labs/FLUX.1-dev \
  --hf-image-provider fal-ai \
  --image-api-size 1024x1024 \
  --use-blender-control \
  --panels 4 \
  --export
```

You can swap `--image-api-model` for another Hugging Face image model that supports `text-to-image` or `image-to-image`. When Blender control is enabled, the bot uses Hugging Face `image-to-image`; without a control/reference image, it uses `text-to-image`.

For Cloudflare Workers AI:

```bash
export CLOUDFLARE_ACCOUNT_ID=your_account_id
export CLOUDFLARE_API_TOKEN=your_workers_ai_token

PYTHONPATH=src python3 -m manga_ai \
  --image-backend api \
  --image-api-provider cloudflare \
  --image-api-model @cf/black-forest-labs/flux-2-klein-4b \
  --image-api-size 1024x1024 \
  --cloudflare-use-multipart \
  --use-blender-control \
  --panels 4 \
  --export
```

Environment equivalents:

```bash
IMAGE_BACKEND=api
IMAGE_API_PROVIDER=cloudflare
IMAGE_API_MODEL=@cf/black-forest-labs/flux-2-klein-4b
IMAGE_API_SIZE=1024x1024
CLOUDFLARE_ACCOUNT_ID=your_account_id
CLOUDFLARE_API_TOKEN=your_workers_ai_token
CLOUDFLARE_USE_MULTIPART=true
USE_BLENDER_CONTROL=true
```

Cloudflare FLUX.2 Klein models use multipart form data and support reference inputs named `input_image_0`, `input_image_1`, etc. The bot sends the Blender/PIL control image as `input_image_0` when reference conditioning is available.

If Cloudflare returns `401 Authentication error`, recreate the token from **Workers AI > Use REST API** or make a custom account-scoped token with `Workers AI Read` and `Workers AI Edit`, then confirm `CLOUDFLARE_ACCOUNT_ID` is from the same Cloudflare account.

To use Cloudflare for story and panel planning too:

```bash
LLM_PROVIDER=cloudflare
LLM_MODEL=@cf/meta/llama-3.1-8b-instruct
```

With those set, the prototype path is:

```text
Cloudflare LLM -> panel planner -> Blender/PIL control -> Cloudflare image API -> bubbles/export
```

## Run the v0.1 end-to-end demo

The small demo uses a fixed 3-panel classroom plan so you can verify the full pipeline before improving the story planner:

```bash
menv/bin/python generate_demo.py scenario.txt
```

If no scenario file is passed, it uses a built-in short scenario. The demo writes:

```text
output/panel_01_blender.png
output/panel_01_manga.png
output/panel_02_blender.png
output/panel_02_manga.png
output/panel_03_blender.png
output/panel_03_manga.png
output/page_01.png
```

It forces Cloudflare image-to-image mode with `@cf/runwayml/stable-diffusion-v1-5-img2img`, sends the Blender/PIL control render as `image_b64`, then draws speech bubbles locally.

The demo also runs deterministic manga style normalization before bubbles are added:

```text
raw generated panel
-> grayscale resize
-> fixed contrast/gamma
-> optional histogram match to references/manga_style_reference.png
-> fixed screentone dots
-> line sharpening/grain
-> speech bubbles
-> page composition
-> light page-wide normalization
```

If you have one panel with the exact manga look you want, save it as:

```text
references/manga_style_reference.png
```

Otherwise the demo uses the first generated raw panel as a temporary reference. The demo keeps comparison files such as `panel_01_manga_raw.png`, `panel_01_manga_normalized.png`, and `page_01_before_page_normalization.png`.

Environment switches:

```bash
STYLE_NORMALIZATION_ENABLED=true
STYLE_REFERENCE_IMAGE=references/manga_style_reference.png
STYLE_NORMALIZATION_GAMMA=0.96
STYLE_NORMALIZATION_CONTRAST=1.08
STYLE_NORMALIZATION_GRAIN_STRENGTH=0.02
```

## Test Control Poses

Generate a visual pose sheet without launching Blender:

```bash
menv/bin/python -B test_poses.py --backend pil
```

Generate the same sheet through Blender:

```bash
menv/bin/python -B test_poses.py --backend blender
```

Run the automated pose/control tests:

```bash
PYTHONPATH=src menv/bin/python -B -m pytest tests/test_pose_control.py -q
```

## Post-Processing Package

Generated manga panels are normalized by the split `manga_ai.post_processing` package:

```text
src/manga_ai/post_processing/
├── pipeline.py
├── config.py
├── image_loader.py
├── resize.py
├── tone_normalizer.py
├── histogram_matcher.py
├── line_enhancer.py
├── screentones.py
├── grain.py
├── metrics.py
├── outlier_detector.py
└── ai_restyler.py
```

The compatibility import still works:

```python
from manga_ai.postprocess.style_normalization import normalize_panel_image
```

New code should import from:

```python
from manga_ai.post_processing import MangaStyleConfig, normalize_panel_image
```

Run the post-processing tests:

```bash
PYTHONPATH=src menv/bin/python -B -m pytest tests/test_post_processing.py -q
```

Run every post-processing module test separately:

```bash
PYTHONPATH=src menv/bin/python -B -m pytest tests/test_post_processing_modules.py -q
```

Run the focused automated suite for the manga bot modules:

```bash
PYTHONPATH=src menv/bin/python -B -m pytest \
  tests/test_effects.py \
  tests/test_integrations.py \
  tests/test_models.py \
  tests/test_pipelines.py \
  tests/test_post_processing.py \
  tests/test_post_processing_modules.py \
  tests/test_pose_control.py \
  tests/test_runners.py \
  -q
```

## Use the sibling bubble project

The manga bot can use your separate `../bubble` project through its `manhwa_bubbles` package. This is now the preferred bubble backend, with fallback to the old internal PIL bubbles if the package is not available.

```bash
BUBBLE_BACKEND=manhwa_bubbles
BUBBLE_PROJECT_PATH=../bubble
BUBBLE_USE_YOLO=false
BUBBLE_PREFER_CAIRO=true
BUBBLE_FALLBACK_TO_INTERNAL=true
```

The bubble stage runs after panel style normalization and before page composition:

```text
Cloudflare image
-> style normalization
-> manhwa_bubbles
-> page composition
```

`BUBBLE_USE_YOLO=false` avoids downloading YOLO models. The bot passes approximate character positions from its panel plan/control data instead.

## Configure tokens once

You can set your Hugging Face or Router token once and all runners will pick it up.

- Create a `.env` file at the repo root (same folder as `run.sh` and `run_colab.sh`):

```bash
echo 'HF_TOKEN=your_token_here' >> .env
```

- Variables recognized by the app (any of these will work):

```
HF_TOKEN
HF_API_KEY
HUGGING_FACE_HUB_TOKEN
LLM_API_KEY
```

- Colab/remote usage:
  - After rsync to `/content/wrk`, create `/content/wrk/.env`:
    ```bash
    echo 'HF_TOKEN=your_token_here' >> /content/wrk/.env
    ```
  - Then run:
    ```bash
    bash run_colab.sh --num-steps 25 --export
    ```

- Shell session (temporary):
  ```bash
  export HF_TOKEN=your_token_here
  python -m manga_ai --export
  ```

- macOS (persist for your user): add to `~/.zshrc` then `source ~/.zshrc`:
  ```bash
  echo 'export HF_TOKEN=your_token_here' >> ~/.zshrc
  ```

# Manhwa AI Generator

A small pipeline to generate webtoon/manhwa panels with scenario generation, manhwa-specific styling/effects, optional validation, and Webtoon-ready export.

This README proposes a clean, maintainable structure and a safe migration path from the current flat layout.

## Recommended Project Structure

Use a src-based layout with a Python package. This keeps imports clean, enables testing/packaging, and separates app code from scripts and outputs.

```
.
├─ src/
│  └─ manga_ai/
│     ├─ __init__.py
│     ├─ cli.py                  # CLI parsing + wiring
│     ├─ config.py               # Config dataclasses + env/CLI merge
│     ├─ pipelines/
│     │  ├─ diffusion.py         # Pipeline loading, device/dtype, caching
│     │  ├─ scenario.py          # LLM+fallback scenario, story synthesis
│     │  └─ assemble.py          # ManhwaAssembler and chapter assembly
│     ├─ effects/
│     │  ├─ style.py             # ManhwaStyler (color, resizing)
│     │  └─ text.py              # ManhwaEffects (speech/thought/sfx)
│     ├─ utils/
│     │  ├─ prompts.py           # get_manhwa_prompts and prompt helpers
│     │  ├─ images.py            # cv helpers, face detection, pose stubs
│     │  ├─ export.py            # Webtoon slicing
│     │  └─ validator.py         # Optional BLIP/CLIP validation
│     └─ runners/
│        └─ generate_chapter.py  # Or __main__.py entry that orchestrates
│
├─ scripts/
│  └─ generate.py                # Thin wrapper: imports runners and runs
│
├─ tests/
│  ├─ test_scenario.py
│  ├─ test_export.py
│  └─ test_effects.py
│
├─ assets/
│  └─ fonts/NanumGothic.ttf      # Optional bundled fonts
│
├─ data/                         # Input/reference assets (kept small)
├─ output/                       # Generated outputs (gitignored)
├─ requirements.txt
├─ .env.example
└─ README.md
```

### Why this structure
- src/ package avoids accidental imports from the project root and keeps tooling sane.
- Clear separation between pipelines, effects, utils, and runners.
- A single script entry (`scripts/generate.py`) for CLI users, while library entry points live under `src/manga_ai/`.
- Tests live separately and target the package API.

## Mapping from current files
- `test10.py` → `src/manga_ai/runners/generate_chapter.py` (or `__main__.py`) and the 
  `scripts/generate.py` thin CLI shim.
- `cli.py` → `src/manga_ai/cli.py`
- `config.py` → `src/manga_ai/config.py`
- `manhwa_utils.py` → split into:
  - `src/manga_ai/effects/style.py` (ManhwaStyler)
  - `src/manga_ai/pipelines/assemble.py` (ManhwaAssembler)
  - `src/manga_ai/utils/prompts.py` (get_manhwa_prompts)
- `manhwa_effects.py` → `src/manga_ai/effects/text.py`
- `export.py` → `src/manga_ai/utils/export.py`
- `validator.py` → `src/manga_ai/utils/validator.py`
- Any future pose/vision helpers → `src/manga_ai/utils/images.py`

You can keep thin “shim” files at old paths during migration that import from the new modules so existing commands keep working.

## Safe migration plan (incremental, no downtime)
1. Create the directories under `src/manga_ai/` with empty `__init__.py` files.
2. Move `config.py` and `cli.py` first; adjust imports in `test10.py` to `from manga_ai.config import Config` and `from manga_ai.cli import parse_args, update_config_from_args`.
3. Split `manhwa_utils.py` into `effects/style.py`, `pipelines/assemble.py`, and `utils/prompts.py`. Update imports in `test10.py` accordingly.
4. Move `manhwa_effects.py` → `effects/text.py`, `export.py` → `utils/export.py`, and `validator.py` → `utils/validator.py`.
5. Create `src/manga_ai/pipelines/diffusion.py` and move pipeline loading + device/dtype selection there.
6. Create `src/manga_ai/pipelines/scenario.py` and move scenario generation + fallback + story synthesis there.
7. Add `scripts/generate.py` that calls the runner module; deprecate directly calling `test10.py`.
8. When ready, replace `test10.py` with a shim or remove it. Update README and any notebooks.

Tip: do this in small PRs/commits. After each move, run a quick smoke test.

## Install
Create and activate a virtual environment (recommended):

```zsh
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

If using CUDA, install the torch build matching your CUDA version (see PyTorch website). The default in `requirements.txt` installs CPU or platform-appropriate wheels.

## Run (current layout)
The current entry script is `test10.py`. Example:

```zsh
python test10.py \
  --device cuda \
  --panels 10 \
  --genre action \
  --setting "neon city at night" \
  --episode-title "Ep 1: Ghost in the Glass" \
  --tone dramatic \
  --protagonist Jin \
  --antagonist Raven \
  --export
```

Outputs:
- `output/story.txt` — readable prose summary
- `output/run_log_*.txt` — reproducible scenario/prompts/seeds log
- `output/manhwa_chapter.png` — assembled chapter image
- `output/export/` — optional Webtoon slices when `--export` is set

## Run (after migration)
Add the project root to `PYTHONPATH` or install the package in editable mode:

```zsh
pip install -e .
python -m manga_ai.runners.generate_chapter [args]
# or
python scripts/generate.py [args]
```

## Quality tooling (optional but recommended)
Add pre-commit hooks and type checking:

```zsh
pip install black isort flake8 mypy
# Example usage
black src tests
isort src tests
flake8 src tests
mypy src
```

## Environment variables
`config.py` supports env overrides in addition to CLI flags. See `.env.example` for common variables:
- `HF_TOKEN` — Hugging Face router token
- `SD_MODEL` — Stable Diffusion model id
- `LLM_MODEL` — LLM id for scenario generation
- `OUTPUT_DIR` — output directory path

You can export them in your shell or load via `dotenv` if desired.

## Notes for Colab/servers
- Ensure GPU is available (`torch.cuda.is_available()`), otherwise the code gracefully falls back to CPU FP32.
- Gated models will auto-fall back to `stabilityai/stable-diffusion-2-1` if unauthorized; pass `--hf-token` to unlock where permitted.

## Next steps
- Move modules into `src/manga_ai/` incrementally as described.
- Add a small test suite under `tests/` for prompts, export slicing, and scenario fallback.
- Optionally add a `pyproject.toml` for packaging, and a lightweight CI.
