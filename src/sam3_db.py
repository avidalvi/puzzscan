"""Database functions for SAM 3 segmentation pipeline.

Adds sam3_piezas table and CRUD operations without touching the existing
`piezas` table used by the OpenCV pipeline.
"""

import json
import sqlite3
from pathlib import Path

from utils.logger import logger

DB_PATH = Path("data/puzzscan_v2.db")


def init_sam3_db() -> None:
    """Create sam3_piezas table if it doesn't exist.

    Does NOT modify the existing `piezas` table.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
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

    conn.commit()
    conn.close()
    logger.info(f"SAM 3 database table initialized at {DB_PATH}")


def clear_sam3_origen(nombre_puzzle: str, origen: str) -> None:
    """Delete all SAM 3 records for a given puzzle and origin (idempotency).

    Args:
        nombre_puzzle: Name of the puzzle (e.g. "ciudad").
        origen: Source image stem (e.g. "piezas_1").
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "DELETE FROM sam3_piezas WHERE nombre_puzzle = ? AND origen = ?",
        (nombre_puzzle, origen),
    )
    deleted = cursor.rowcount
    conn.commit()
    conn.close()
    if deleted > 0:
        logger.info(
            f"Deleted {deleted} previous SAM 3 records for "
            f"puzzle='{nombre_puzzle}', origen='{origen}'"
        )


def insert_sam3_pieza(
    nombre_puzzle: str,
    origen: str,
    piece_index: int,
    ruta_imagen: str,
    contour_json: list[list[int]],
    bounding_box_json: list[float],
    confidence_score: float,
    area_pixels: int,
    posicion_original: dict[str, int],
    text_prompt: str,
    debug_image_path: str | None = None,
) -> int:
    """Insert a new SAM 3 piece record and return its ID.

    Args:
        nombre_puzzle: Name of the puzzle.
        origen: Source image stem.
        piece_index: Index of the piece within the image.
        ruta_imagen: Path to the cropped PNG with alpha channel.
        contour_json: Polygon contour as list of [x, y] pairs.
        bounding_box_json: Bounding box as [x0, y0, x1, y1].
        confidence_score: Model confidence (0.0–1.0).
        area_pixels: Mask area in pixels.
        posicion_original: Center coordinates {"x": ..., "y": ...}.
        text_prompt: Text prompt used for segmentation.
        debug_image_path: Optional path to the debug overlay image.

    Returns:
        The auto-assigned ID of the inserted record.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO sam3_piezas (
            nombre_puzzle, origen, piece_index, ruta_imagen,
            contour_json, bounding_box_json, confidence_score,
            area_pixels, posicion_original, text_prompt, debug_image_path
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            nombre_puzzle,
            origen,
            piece_index,
            ruta_imagen,
            json.dumps(contour_json),
            json.dumps(bounding_box_json),
            confidence_score,
            area_pixels,
            json.dumps(posicion_original),
            text_prompt,
            debug_image_path,
        ),
    )

    pieza_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return pieza_id or 0


def update_sam3_ruta_imagen(pieza_id: int, ruta_imagen: str) -> None:
    """Update the ruta_imagen field for a given piece ID.

    Args:
        pieza_id: The ID of the piece to update.
        ruta_imagen: The new image path.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "UPDATE sam3_piezas SET ruta_imagen = ? WHERE id = ?",
        (ruta_imagen, pieza_id),
    )
    conn.commit()
    conn.close()


def update_sam3_debug_path(pieza_id: int, debug_image_path: str) -> None:
    """Update debug_image_path for all pieces of the same origen.

    Args:
        pieza_id: Any piece ID from the batch (used to find nombre_puzzle + origen).
        debug_image_path: Path to the debug overlay image.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT nombre_puzzle, origen FROM sam3_piezas WHERE id = ?",
        (pieza_id,),
    )
    row = cursor.fetchone()
    if row:
        conn.execute(
            """UPDATE sam3_piezas SET debug_image_path = ?
               WHERE nombre_puzzle = ? AND origen = ?""",
            (debug_image_path, row[0], row[1]),
        )
        conn.commit()
    conn.close()
