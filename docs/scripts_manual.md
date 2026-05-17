# Script Usage Manual

This manual describes how to run the project scripts from the repository root:

```powershell
cd C:\proxectos\puzzscan
```

Always prefer `uv run` so commands use the project environment.

## Environment Setup

Install or refresh dependencies:

```powershell
uv sync
```

Create a local `.env` file when you want to customize SAM 3 behavior:

```powershell
Copy-Item .env.example .env
```

Available SAM 3 variables:

```text
SAM3_CONFIDENCE_THRESHOLD=0.5
SAM3_TEXT_PROMPT=puzzle piece
```

## SAM 3 Piece Extraction

Command:

```powershell
uv run python -m src.sam3_extractor
```

Do not run this as `uv run python src/sam3_extractor.py`; direct file execution
can fail with `ModuleNotFoundError: No module named 'utils'` because Python
does not include the project root on the import path.

Default input:

```text
data/ciudad/piezas_*.jpg
```

Main outputs:

```text
output/ciudad/piezas_sam3/<origen>/pieza_1.png
output/ciudad/piezas_sam3/<origen>/pieza_2.png
output/ciudad/piezas_sam3/<origen>_debug.jpg
data/puzzscan_v2.db
logs/
```

Behavior:

- Loads SAM 3 once and reuses it for all matching images.
- Uses `SAM3_TEXT_PROMPT` to ask the model for puzzle pieces.
- Saves each detected piece as a transparent PNG.
- Saves a debug overlay with bounding boxes and piece numbers.
- Numbers pieces from top-left to bottom-right.
- Keeps the debug number consistent with `pieza_<number>.png`.
- Clears previous `sam3_piezas` rows for the same puzzle and source image before
  inserting fresh results.

Database table:

```text
sam3_piezas
```

Useful fields:

- `piece_index`: top-left to bottom-right piece number.
- `ruta_imagen`: extracted transparent PNG path.
- `bounding_box_json`: source-image bounding box.
- `posicion_original`: center point in the source image.
- `debug_image_path`: matching debug overlay path.

## OpenCV Baseline Extraction

Command:

```powershell
uv run python -m src.detector
```

Default input:

```text
data/ciudad/piezas_*.jpg
```

Main outputs:

```text
output/ciudad/piezas/<origen>/pieza_<db_id>.png
output/ciudad/piezas/<origen>_anotada.jpg
data/puzzscan_v2.db
logs/
```

Behavior:

- Uses thresholding and contours to find puzzle pieces.
- Normalizes piece scale around a target area.
- Rotates pieces by PCA before cropping.
- Saves transparent PNG crops.
- Saves an annotated source image.
- Stores records in the `piezas` table.

## Comparison Notebook Generator

Command:

```powershell
uv run python scripts/generate_notebook.py
```

Output:

```text
notebooks/compare_segmentation.ipynb
```

The generated notebook compares OpenCV and SAM 3 results for:

```text
PUZZLE = "ciudad"
ORIGEN = "piezas_1"
```

Run both extraction pipelines before opening the notebook if you want the
comparison cells to display results from both methods.

## Tests and Checks

Run the normal test suite:

```powershell
uv run pytest
```

Run the SAM 3 extractor tests:

```powershell
uv run pytest tests/test_sam3_extractor.py -q
```

Run Ruff on edited files:

```powershell
uv run ruff check src tests scripts
uv run ruff format src tests scripts
```

Slow tests that require the real SAM 3 model and suitable hardware are marked
with `slow`:

```powershell
uv run pytest tests/test_sam3_extractor.py -m slow -v
```

## Troubleshooting

### `ModuleNotFoundError: No module named 'utils'`

Run modules from the project root:

```powershell
uv run python -m src.sam3_extractor
uv run python -m src.detector
```

### Broken `.venv` or Missing Python

If `uv run` reports that the virtual environment points to a missing Python,
recreate it:

```powershell
Remove-Item -Recurse -Force .venv
uv sync
```

### SAM 3 Warnings About GPU or Deprecated Packages

Warnings such as Flash Attention being disabled or `pkg_resources` being
deprecated are not necessarily fatal. Look for the final traceback or error
message to identify the real blocker.

### No Pieces Detected

Check:

- The input images exist under `data/<puzzle>/`.
- `SAM3_TEXT_PROMPT` matches what should be detected.
- `SAM3_CONFIDENCE_THRESHOLD` is not too high.
- The source image has enough contrast and visible separation between pieces.
