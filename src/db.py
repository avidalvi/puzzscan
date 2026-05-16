import sqlite3
import json
from pathlib import Path
from utils.logger import logger

DB_PATH = Path("data/puzzscan_v2.db")

def init_db() -> None:
    """Initializes the SQLite database and creates the pieces table if it doesn't exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS piezas (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nombre_puzzle TEXT NOT NULL,
        origen TEXT NOT NULL,
        ruta_imagen TEXT NOT NULL,
        posicion_original TEXT NOT NULL,
        factor_escala REAL NOT NULL,
        colores_dominantes TEXT NOT NULL,
        luminosidad REAL NOT NULL,
        estado TEXT DEFAULT 'AVAILABLE',
        fecha_creacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    conn.commit()
    conn.close()
    logger.info(f"Database initialized at {DB_PATH}")

def clear_origen(nombre_puzzle: str, origen: str) -> None:
    """Deletes all records for a given puzzle and origin to ensure idempotency."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM piezas WHERE nombre_puzzle = ? AND origen = ?", (nombre_puzzle, origen))
    deleted = cursor.rowcount
    conn.commit()
    conn.close()
    if deleted > 0:
        logger.info(f"Deleted {deleted} previous records for puzzle '{nombre_puzzle}', origin '{origen}'")

def insert_pieza(
    nombre_puzzle: str,
    origen: str,
    ruta_imagen: str,
    posicion_original: dict,
    factor_escala: float,
    colores_dominantes: list,
    luminosidad: float
) -> int:
    """Inserts a new piece into the database and returns its ID."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
    INSERT INTO piezas (
        nombre_puzzle, origen, ruta_imagen, posicion_original, factor_escala, 
        colores_dominantes, luminosidad
    ) VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        nombre_puzzle,
        origen,
        ruta_imagen,
        json.dumps(posicion_original),
        factor_escala,
        json.dumps(colores_dominantes),
        luminosidad
    ))
    
    pieza_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return pieza_id or 0
