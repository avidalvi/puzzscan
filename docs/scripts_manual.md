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

## Piece Profile Generation

Command:

```powershell
uv run python scripts/profile_pieces.py --puzzle ciudad --origen piezas_1
```

Optional filters:

```powershell
uv run python scripts/profile_pieces.py --puzzle ciudad
uv run python scripts/profile_pieces.py
```

Inputs:

```text
data/puzzscan_v2.db, table sam3_piezas
output/<puzzle>/piezas_sam3/<origen>/pieza_<n>.png
```

Main output:

```text
data/puzzscan_v2.db, table pieza_perfiles
```

Behavior:

- Reads SAM 3 pieces from `sam3_piezas`.
- Uses the PNG alpha channel as the source mask.
- Extracts the outer contour and converts it to a continuous curve.
- Smooths raster stair-steps and small pixel artefacts.
- Detects four principal corners.
- Splits the piece into faces in `1-2`, `2-3`, `3-4`, `4-1` order.
- Normalizes each face as an open parametric curve from `(0, 0)` to `(1, 0)`.
- Classifies faces as `lisa`, `macho`, `hembra`, or `desconocida`.
- Reduces each face to 36 arc-length-spaced control points.
- Stores profile geometry, descriptors, status, and quality flags.

Important details:

- The normalized face is not forced to be a function `Y=f(X)`. Puzzle tabs and
  holes can legitimately move backwards in X.
- The reduced descriptor uses 36 control points per face, stored as 72 floats
  `(x, y)`.
- Existing profiles for the same `pieza_id` and `profile_version` are replaced
  before insertion.

## Piece Profile Exploration Notebook

Command:

```powershell
uv run python scripts/generate_piece_profile_notebook.py
```

Output:

```text
notebooks/piece_profile_exploration.ipynb
```

The notebook inspects the same five representative pieces through the full
profile pipeline:

- RGB, alpha, and pipeline mask.
- Mask quality.
- Raw contour vs smoothed puzzle contour.
- Corner detection.
- Face split.
- Normalized parametric face curves.
- Control-point descriptor reconstruction and RMSE.
- Final profile summary.

Representative pieces are selected deterministically by area quantiles, not at
random, so the notebook shows small, medium, and large examples.

When editing `src.piece_profile`, restart the notebook kernel before rerunning
cells so Python does not reuse a cached module.

## Tests and Checks

Run the normal test suite:

```powershell
uv run pytest
```

Run the SAM 3 extractor tests:

```powershell
uv run pytest tests/test_sam3_extractor.py -q
```

Run the piece profile tests:

```powershell
uv run pytest tests/test_piece_profile.py tests/test_piece_profile_db.py -q
```

Run the piece search engine tests:

```powershell
uv run pytest tests/test_piece_search.py -q
```

## Piece Search Engine

The search engine (`src/piece_search.py`) matches puzzle pieces by comparing
their geometric face profiles. It ranks compatible candidates for each of the
four cardinal directions (N, E, S, W) of a center piece.

Generate the exploration notebook:

```powershell
uv run python scripts/generate_piece_search_notebook.py
```

Open the notebook:

```powershell
uv run jupyter notebook notebooks/piece_search_exploration.ipynb
```

### Key Design Decisions

- **RMSE over DTW**: Point-to-point RMSE on the 36-point control descriptor is
  the primary geometric score. DTW is available as a diagnostic only.
- **Visual signals are complementary**: Luminance, colour (Lab), and texture
  (gradient) profiles are weighted at 0.20/0.20/0.10 vs 1.00 for geometry.
- **No persistence in first iteration**: Rankings exist only in memory.
- **Unknown face types**: `desconocida` is allowed only when
  `ALLOW_UNKNOWN_FACE = True`, with a penalty.
- **Direction/face mapping**: The default mapping
  (N→1-2, E→2-3, S→3-4, W→4-1) must be validated visually for each puzzle.

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

### Notebook Import Uses an Old `src.piece_profile`

If the profile notebook raises an import error for a function that exists on
disk, restart the notebook kernel. Jupyter can keep an older imported module in
memory after code changes.
