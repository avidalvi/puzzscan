"""Database functions for piece profile geometry pipeline.

Creates the `pieza_perfiles` table and provides CRUD operations
without modifying existing tables (piezas, sam3_piezas).
"""

import json
import sqlite3
from pathlib import Path
from typing import Any

from utils.logger import logger

DB_PATH = Path("data/puzzscan_v2.db")


def init_piece_profile_db() -> None:
    """Create pieza_perfiles table if it doesn't exist.

    Does NOT modify existing tables.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS pieza_perfiles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pieza_id INTEGER NOT NULL,
        pipeline TEXT NOT NULL,
        nombre_puzzle TEXT NOT NULL,
        origen TEXT NOT NULL,
        piece_index INTEGER NOT NULL,
        corner_points_json TEXT NOT NULL,
        corner_points_crop_json TEXT NOT NULL,
        corner_points_source_json TEXT NOT NULL,
        corner_points_normalized_json TEXT NOT NULL,
        face_order TEXT NOT NULL,
        faces_json TEXT NOT NULL,
        piece_kind TEXT NOT NULL,
        profile_status TEXT NOT NULL,
        quality_flags_json TEXT NOT NULL,
        profile_version TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(pieza_id, profile_version)
    )
    """)

    conn.commit()
    conn.close()
    logger.info(f"Piece profile database table initialized at {DB_PATH}")


def clear_piece_profile(pieza_id: int, profile_version: str) -> None:
    """Remove existing profile for (pieza_id, profile_version) — idempotent."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "DELETE FROM pieza_perfiles WHERE pieza_id = ? AND profile_version = ?",
        (pieza_id, profile_version),
    )
    conn.commit()
    conn.close()


def insert_piece_profile(
    pieza_id: int,
    pipeline: str,
    nombre_puzzle: str,
    origen: str,
    piece_index: int,
    corner_points_json: list[list[float]],
    corner_points_crop_json: list[list[float]],
    corner_points_source_json: list[list[float]],
    corner_points_normalized_json: list[list[float]],
    face_order: str,
    faces_json: dict[str, Any],
    piece_kind: str,
    profile_status: str,
    quality_flags_json: list[str],
    profile_version: str,
) -> int:
    """Insert a piece profile row. Clears any previous version first."""
    clear_piece_profile(pieza_id, profile_version)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO pieza_perfiles (
            pieza_id, pipeline, nombre_puzzle, origen, piece_index,
            corner_points_json, corner_points_crop_json,
            corner_points_source_json, corner_points_normalized_json,
            face_order, faces_json, piece_kind, profile_status,
            quality_flags_json, profile_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            pieza_id,
            pipeline,
            nombre_puzzle,
            origen,
            piece_index,
            json.dumps(corner_points_json),
            json.dumps(corner_points_crop_json),
            json.dumps(corner_points_source_json),
            json.dumps(corner_points_normalized_json),
            face_order,
            json.dumps(faces_json),
            piece_kind,
            profile_status,
            json.dumps(quality_flags_json),
            profile_version,
        ),
    )

    new_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return new_id or 0


def fetch_sam3_pieces(
    nombre_puzzle: str | None = None,
    origen: str | None = None,
) -> list[sqlite3.Row]:
    """Fetch SAM 3 piece rows with optional filters.

    Args:
        nombre_puzzle: If set, filter by puzzle name.
        origen: If set, filter by source image stem.

    Returns:
        List of sqlite3.Row objects with column access.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    query = "SELECT * FROM sam3_piezas"
    params: list[str] = []
    conditions: list[str] = []

    if nombre_puzzle is not None:
        conditions.append("nombre_puzzle = ?")
        params.append(nombre_puzzle)
    if origen is not None:
        conditions.append("origen = ?")
        params.append(origen)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += " ORDER BY piece_index"

    rows = conn.execute(query, params).fetchall()
    conn.close()
    return rows
