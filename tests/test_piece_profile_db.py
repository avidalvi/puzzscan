"""Unit tests for src/piece_profile_db.py."""

import sqlite3
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture()
def tmp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Provide a temporary SQLite DB and patch DB_PATH."""
    db_file = tmp_path / "test.db"

    import src.piece_profile_db as db_mod

    monkeypatch.setattr(db_mod, "DB_PATH", db_file)

    db_mod.init_piece_profile_db()
    return db_file


def test_table_created(tmp_db: Path) -> None:
    """init_piece_profile_db must create the pieza_perfiles table."""
    conn = sqlite3.connect(tmp_db)
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='pieza_perfiles'"
    ).fetchall()
    conn.close()
    assert len(tables) == 1
    assert tables[0][0] == "pieza_perfiles"


def test_init_idempotent(tmp_db: Path) -> None:
    """Calling init_piece_profile_db twice must not raise."""
    import src.piece_profile_db as db_mod

    db_mod.init_piece_profile_db()


def test_insert_and_query(tmp_db: Path) -> None:
    """insert_piece_profile must create a retrievable record."""
    from src.piece_profile_db import insert_piece_profile

    new_id = insert_piece_profile(
        pieza_id=1,
        pipeline="sam3",
        nombre_puzzle="test_puzzle",
        origen="piezas_1",
        piece_index=0,
        corner_points_json=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        corner_points_crop_json=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        corner_points_source_json=[
            [10.0, 10.0],
            [20.0, 10.0],
            [20.0, 20.0],
            [10.0, 20.0],
        ],
        corner_points_normalized_json=[[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]],
        face_order="1-2-3-4-1",
        faces_json={"1-2": {"tipo": "macho"}},
        piece_kind="interior",
        profile_status="valid",
        quality_flags_json=[],
        profile_version="piece_profile_v1",
    )

    assert new_id > 0

    conn = sqlite3.connect(tmp_db)
    row = conn.execute(
        "SELECT pieza_id, profile_status FROM pieza_perfiles WHERE id = ?",
        (new_id,),
    ).fetchone()
    conn.close()

    assert row is not None
    assert row[0] == 1
    assert row[1] == "valid"


def test_idempotent_unique(tmp_db: Path) -> None:
    """Insert twice for same (pieza_id, profile_version) must replace."""
    from src.piece_profile_db import insert_piece_profile

    kwargs: dict[str, Any] = {
        "pieza_id": 1,
        "pipeline": "sam3",
        "nombre_puzzle": "test",
        "origen": "o1",
        "piece_index": 0,
        "corner_points_json": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        "corner_points_crop_json": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        "corner_points_source_json": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        "corner_points_normalized_json": [
            [0.0, 0.0],
            [0.5, 0.0],
            [0.5, 0.5],
            [0.0, 0.5],
        ],
        "face_order": "1-2-3-4-1",
        "faces_json": {"1-2": {"tipo": "macho"}},
        "piece_kind": "interior",
        "profile_status": "valid",
        "quality_flags_json": [],
        "profile_version": "piece_profile_v1",
    }

    id1 = insert_piece_profile(**kwargs)

    kwargs["profile_status"] = "needs_review"
    id2 = insert_piece_profile(**kwargs)

    assert id2 != id1

    conn = sqlite3.connect(tmp_db)
    rows = conn.execute(
        "SELECT id FROM pieza_perfiles"
        " WHERE pieza_id=1 AND profile_version='piece_profile_v1'"
    ).fetchall()
    conn.close()

    assert len(rows) == 1
    assert rows[0][0] == id2


def test_fetch_sam3_pieces_empty(tmp_db: Path, tmp_path: Path) -> None:
    """fetch_sam3_pieces must return empty list when no sam3_piezas records exist."""
    from src.piece_profile_db import fetch_sam3_pieces

    # Ensure sam3_piezas table exists
    conn = sqlite3.connect(tmp_db)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS sam3_piezas (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nombre_puzzle TEXT NOT NULL,
        origen TEXT NOT NULL,
        piece_index INTEGER NOT NULL,
        ruta_imagen TEXT NOT NULL,
        contour_json TEXT NOT NULL,
        bounding_box_json TEXT NOT NULL,
        confidence_score REAL NOT NULL,
        area_pixels INTEGER NOT NULL,
        posicion_original TEXT NOT NULL,
        text_prompt TEXT NOT NULL,
        debug_image_path TEXT,
        estado TEXT DEFAULT 'AVAILABLE',
        fecha_creacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.close()

    pieces = fetch_sam3_pieces()
    assert pieces == []
