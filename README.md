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
OPENAI_API_KEY
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
