"""Unit tests for src/sam3_extractor.py and src/sam3_db.py.

Tests that require a real GPU + SAM 3 model are marked @pytest.mark.slow
and are skipped by default. Run them explicitly with:
    uv run pytest tests/test_sam3_extractor.py -m slow -v
"""

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# T4.1 — Config loading
# ---------------------------------------------------------------------------


def test_load_config_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Config values must fall back to safe defaults when .env is absent."""
    monkeypatch.delenv("SAM3_CONFIDENCE_THRESHOLD", raising=False)
    monkeypatch.delenv("SAM3_TEXT_PROMPT", raising=False)

    # Re-import to pick up monkeypatched env (os.getenv evaluated at import time,
    # so we read values directly from the module after patching os.environ)
    import importlib

    import src.sam3_extractor as mod

    importlib.reload(mod)

    assert mod.CONFIDENCE_THRESHOLD == 0.5
    assert mod.TEXT_PROMPT == "puzzle piece"


def test_load_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Config values must be read from environment variables."""
    monkeypatch.setenv("SAM3_CONFIDENCE_THRESHOLD", "0.7")
    monkeypatch.setenv("SAM3_TEXT_PROMPT", "jigsaw piece")

    import importlib

    import src.sam3_extractor as mod

    importlib.reload(mod)

    assert mod.CONFIDENCE_THRESHOLD == 0.7
    assert mod.TEXT_PROMPT == "jigsaw piece"


# ---------------------------------------------------------------------------
# T4.2 — Contour extraction
# ---------------------------------------------------------------------------


def test_extract_contour_square() -> None:
    """A square mask should produce a 4-point convex hull contour."""
    from src.sam3_extractor import _extract_contour

    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 255  # 60x60 square

    contour = _extract_contour(mask)

    assert len(contour) >= 4, "Expected at least 4 contour points for a square"
    for pt in contour:
        assert len(pt) == 2, f"Each point must be [x, y], got {pt}"
        x, y = pt
        assert 19 <= x <= 80 and 19 <= y <= 80, f"Point {pt} outside expected bounds"


def test_extract_contour_empty_mask() -> None:
    """An all-zero mask should return an empty contour list."""
    from src.sam3_extractor import _extract_contour

    mask = np.zeros((100, 100), dtype=np.uint8)
    assert _extract_contour(mask) == []


# ---------------------------------------------------------------------------
# T4.3 — Alpha channel crop
# ---------------------------------------------------------------------------


def test_crop_with_alpha_shape() -> None:
    """Cropped RGBA image must have 4 channels and correct crop bounds."""
    from src.sam3_extractor import _crop_with_alpha

    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[:, :] = [100, 150, 200]  # fill with solid colour

    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[50:100, 60:120] = 255  # 50x60 region

    rgba = _crop_with_alpha(img, mask)

    assert rgba.shape[2] == 4, "Output must have 4 channels (RGBA)"
    assert rgba.shape[0] == 50, f"Height should be 50, got {rgba.shape[0]}"
    assert rgba.shape[1] == 60, f"Width should be 60, got {rgba.shape[1]}"


def test_crop_with_alpha_transparency() -> None:
    """Pixels outside the mask must have Alpha == 0."""
    from src.sam3_extractor import _crop_with_alpha

    img = np.full((100, 100, 3), 200, dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:40, 10:40] = 255  # top-left square

    rgba = _crop_with_alpha(img, mask)
    alpha_channel = rgba[:, :, 3]

    # All alpha values in the crop region should be either 0 or 255
    unique_alpha = np.unique(alpha_channel)
    for v in unique_alpha:
        assert v in (0, 255), f"Unexpected alpha value {v}"


def test_crop_with_alpha_empty_mask() -> None:
    """Empty mask should return a 1x1 RGBA placeholder without crashing."""
    from src.sam3_extractor import _crop_with_alpha

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    result = _crop_with_alpha(img, mask)
    assert result.shape == (1, 1, 4)


# ---------------------------------------------------------------------------
# T4.4 — Debug image
# ---------------------------------------------------------------------------


def test_build_debug_image_shape() -> None:
    """Debug image must have the same spatial dimensions as the source."""
    from src.sam3_extractor import _build_debug_image

    img = np.zeros((300, 400, 3), dtype=np.uint8)
    mask1 = np.zeros((300, 400), dtype=np.uint8)
    mask1[50:100, 50:150] = 255
    mask2 = np.zeros((300, 400), dtype=np.uint8)
    mask2[150:250, 200:350] = 255

    debug = _build_debug_image(img, [mask1, mask2])

    assert debug.shape == (300, 400, 3), f"Unexpected shape {debug.shape}"
    assert debug.dtype == np.uint8


def test_sort_detections_top_left_to_bottom_right() -> None:
    """Detections should be ordered row-major by source bounding boxes."""
    from src.sam3_extractor import (
        _PieceDetection,
        _sort_detections_top_left_to_bottom_right,
    )

    mask = np.zeros((10, 10), dtype=np.uint8)
    detections = [
        _PieceDetection(mask, [50.0, 40.0, 60.0, 50.0], 0.9, [], 1, 55, 45),
        _PieceDetection(mask, [40.0, 10.0, 50.0, 20.0], 0.9, [], 1, 45, 15),
        _PieceDetection(mask, [10.0, 10.0, 20.0, 20.0], 0.9, [], 1, 15, 15),
    ]

    ordered = _sort_detections_top_left_to_bottom_right(detections)

    assert [detection.box for detection in ordered] == [
        [10.0, 10.0, 20.0, 20.0],
        [40.0, 10.0, 50.0, 20.0],
        [50.0, 40.0, 60.0, 50.0],
    ]


# ---------------------------------------------------------------------------
# T4.5 — Database: insert and clear
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Provide a temporary SQLite DB and patch DB_PATH in sam3_db."""
    db_file = tmp_path / "test.db"

    import src.sam3_db as db_mod

    monkeypatch.setattr(db_mod, "DB_PATH", db_file)

    db_mod.init_sam3_db()
    return db_file


def test_insert_and_query(tmp_db: Path) -> None:
    """insert_sam3_pieza must create a retrievable record."""
    from src.sam3_db import insert_sam3_pieza

    pieza_id = insert_sam3_pieza(
        nombre_puzzle="test_puzzle",
        origen="piezas_1",
        piece_index=0,
        ruta_imagen="output/test/pieza_1.png",
        contour_json=[[10, 10], [50, 10], [50, 50], [10, 50]],
        bounding_box_json=[10.0, 10.0, 50.0, 50.0],
        confidence_score=0.85,
        area_pixels=1600,
        posicion_original={"x": 30, "y": 30},
        text_prompt="puzzle piece",
    )

    assert pieza_id > 0, "Expected a positive integer ID"

    conn = sqlite3.connect(tmp_db)
    row = conn.execute(
        "SELECT nombre_puzzle, confidence_score FROM sam3_piezas WHERE id = ?",
        (pieza_id,),
    ).fetchone()
    conn.close()

    assert row is not None
    assert row[0] == "test_puzzle"
    assert abs(row[1] - 0.85) < 1e-6


def test_clear_origen(tmp_db: Path) -> None:
    """clear_sam3_origen must delete all records for a given puzzle+origen."""
    from src.sam3_db import clear_sam3_origen, insert_sam3_pieza

    for i in range(3):
        insert_sam3_pieza(
            nombre_puzzle="test_puzzle",
            origen="piezas_1",
            piece_index=i,
            ruta_imagen="",
            contour_json=[],
            bounding_box_json=[0.0, 0.0, 1.0, 1.0],
            confidence_score=0.9,
            area_pixels=100,
            posicion_original={"x": 0, "y": 0},
            text_prompt="puzzle piece",
        )

    conn = sqlite3.connect(tmp_db)
    count_before = conn.execute("SELECT COUNT(*) FROM sam3_piezas").fetchone()[0]
    conn.close()
    assert count_before == 3

    clear_sam3_origen("test_puzzle", "piezas_1")

    conn = sqlite3.connect(tmp_db)
    count_after = conn.execute("SELECT COUNT(*) FROM sam3_piezas").fetchone()[0]
    conn.close()
    assert count_after == 0


def test_clear_origen_only_target(tmp_db: Path) -> None:
    """clear_sam3_origen must not delete records from other origins."""
    from src.sam3_db import clear_sam3_origen, insert_sam3_pieza

    common_kwargs: dict[str, object] = {
        "piece_index": 0,
        "ruta_imagen": "",
        "contour_json": [],
        "bounding_box_json": [0.0, 0.0, 1.0, 1.0],
        "confidence_score": 0.9,
        "area_pixels": 100,
        "posicion_original": {"x": 0, "y": 0},
        "text_prompt": "puzzle piece",
    }

    insert_sam3_pieza(nombre_puzzle="puzzle_a", origen="piezas_1", **common_kwargs)  # type: ignore[arg-type]
    insert_sam3_pieza(nombre_puzzle="puzzle_b", origen="piezas_2", **common_kwargs)  # type: ignore[arg-type]

    clear_sam3_origen("puzzle_a", "piezas_1")

    conn = sqlite3.connect(tmp_db)
    remaining = conn.execute("SELECT nombre_puzzle FROM sam3_piezas").fetchall()
    conn.close()

    assert len(remaining) == 1
    assert remaining[0][0] == "puzzle_b"


# ---------------------------------------------------------------------------
# T4.6 — process_image error handling (mocked)
# ---------------------------------------------------------------------------


def test_process_image_corrupt_path(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """process_image must log an error and not crash on a bad file path."""
    import src.sam3_db as db_mod

    monkeypatch.setattr(db_mod, "DB_PATH", tmp_db)

    from src.sam3_extractor import process_image

    # Non-existent file — should log error, not raise
    process_image(str(tmp_path / "nonexistent.jpg"))

    # No records should have been inserted
    conn = sqlite3.connect(tmp_db)
    count = conn.execute("SELECT COUNT(*) FROM sam3_piezas").fetchone()[0]
    conn.close()
    assert count == 0


def test_process_image_zero_detections(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """process_image must log a warning and insert 0 records
    when SAM 3 returns nothing.
    """
    import src.sam3_db as db_mod

    monkeypatch.setattr(db_mod, "DB_PATH", tmp_db)

    # Create a real (tiny) image so cv2.imread succeeds
    img_path = tmp_path / "piezas_1.jpg"
    dummy = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(img_path), dummy)

    # Mock inference to return empty results
    empty_state = {
        "masks": torch.zeros((0, 1, 100, 100)),
        "boxes": torch.zeros((0, 4)),
        "scores": torch.zeros((0,)),
        "masks_logits": torch.zeros((0, 100, 100)),
    }

    mock_processor = MagicMock()
    mock_processor.set_image.return_value = empty_state
    mock_processor.reset_all_prompts.return_value = None
    mock_processor.set_text_prompt.return_value = empty_state

    with patch("src.sam3_extractor.load_model", return_value=MagicMock()):
        with patch("src.sam3_extractor.Sam3Processor", return_value=mock_processor):  # type: ignore[attr-defined]
            from src.sam3_extractor import process_image

            process_image(str(img_path))

    conn = sqlite3.connect(tmp_db)
    count = conn.execute("SELECT COUNT(*) FROM sam3_piezas").fetchone()[0]
    conn.close()
    assert count == 0


# ---------------------------------------------------------------------------
# T4.7 — Slow tests (real GPU + model)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_process_image_real_inference() -> None:
    """Full end-to-end test with real SAM 3 model. Requires GPU + model download.

    Run with: uv run pytest tests/test_sam3_extractor.py -m slow -v
    """
    from src.sam3_db import init_sam3_db
    from src.sam3_extractor import load_model, process_image

    init_sam3_db()
    load_model()

    img_path = "data/ciudad/piezas_1.jpg"
    if not Path(img_path).exists():
        pytest.skip("Source image not available")

    process_image(img_path)

    # Check that some pieces were saved to disk
    output_files = list(Path("output/ciudad/piezas_sam3/piezas_1").glob("pieza_*.png"))
    assert len(output_files) > 0, "Expected at least one piece PNG to be saved"
