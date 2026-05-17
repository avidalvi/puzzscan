# puzzscan

Puzzle-piece segmentation experiments using two pipelines:

- **SAM 3 zero-shot extraction**: detects pieces from a text prompt and exports
  transparent PNG crops plus a numbered debug image.
- **OpenCV extraction**: baseline classical image-processing pipeline.

The project uses `uv` for dependency management.

## Quick Start

From the project root:

```powershell
cd C:\proxectos\puzzscan
uv sync
```

Optional SAM 3 settings can be copied from the example file:

```powershell
Copy-Item .env.example .env
```

## Run SAM 3 Extraction

Use module execution so Python can resolve both `src` and `utils` imports:

```powershell
uv run python -m src.sam3_extractor
```

The default entry point processes:

```text
data/ciudad/piezas_*.jpg
```

Outputs are written to:

```text
output/ciudad/piezas_sam3/
```

For each source image, the extractor creates:

- `output/<puzzle>/piezas_sam3/<origen>/pieza_1.png`, `pieza_2.png`, ...
- `output/<puzzle>/piezas_sam3/<origen>_debug.jpg`
- database rows in `data/puzzscan_v2.db`, table `sam3_piezas`

Piece numbers are ordered from top-left to bottom-right, and the debug labels
match the individual extracted image names.

## Run OpenCV Baseline

```powershell
uv run python -m src.detector
```

Outputs are written to:

```text
output/ciudad/piezas/
```

and database rows are stored in `data/puzzscan_v2.db`, table `piezas`.

## Generate Comparison Notebook

```powershell
uv run python scripts/generate_notebook.py
```

This creates:

```text
notebooks/compare_segmentation.ipynb
```

## Documentation

See [docs/scripts_manual.md](docs/scripts_manual.md) for a fuller manual of the
available scripts, expected inputs, outputs, and common troubleshooting notes.
