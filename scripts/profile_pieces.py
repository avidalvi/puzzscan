"""Entry point script for piece profile generation.

Generates geometric profiles for SAM 3 extracted pieces and persists them
to the pieza_perfiles table.

Usage:
    uv run python scripts/profile_pieces.py --puzzle ciudad --origen piezas_1
    uv run python scripts/profile_pieces.py --puzzle ciudad
    uv run python scripts/profile_pieces.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.piece_profile import build_piece_profile, profile_to_db_dict
from src.piece_profile_db import (
    fetch_sam3_pieces,
    init_piece_profile_db,
    insert_piece_profile,
)
from utils.logger import logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate piece geometric profiles")
    parser.add_argument(
        "--puzzle", type=str, default=None, help="Filter by puzzle name"
    )
    parser.add_argument(
        "--origen", type=str, default=None, help="Filter by source image stem"
    )
    args = parser.parse_args()

    init_piece_profile_db()

    pieces = fetch_sam3_pieces(nombre_puzzle=args.puzzle, origen=args.origen)
    logger.info(f"Found {len(pieces)} SAM 3 piece(s) to process")

    counts = {"valid": 0, "needs_review": 0, "invalid": 0, "errors": 0}

    for piece_row in pieces:
        try:
            profile = build_piece_profile(dict(piece_row))
            profile_dict = profile_to_db_dict(profile)
            insert_piece_profile(**profile_dict)

            status = profile.profile_status
            counts[status] = counts.get(status, 0) + 1

            logger.info(
                f"  Piece {profile.piece_index} (pieza_id={profile.pieza_id}): "
                f"status={status} kind={profile.piece_kind} "
                f"flags={profile.quality_flags}"
            )
        except Exception:
            logger.exception(
                f"Error processing piece id={piece_row['id']} "
                f"index={piece_row['piece_index']}"
            )
            counts["errors"] += 1

    logger.info(
        f"Profile generation complete: "
        f"valid={counts['valid']}, needs_review={counts['needs_review']}, "
        f"invalid={counts['invalid']}, errors={counts['errors']}"
    )


if __name__ == "__main__":
    main()
