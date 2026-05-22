"""Piece search engine: load profiles, rank candidates by geometry and visual signals.

Phases implemented:
  1. In-memory data model (dataclasses)
  2. Direction/face conventions
  3. Topological filter (face type)
  4. Curve inversion and geometric RMSE
  5. DTW diagnostic (constrained Sakoe-Chiba)
  6. Ranking (top-k per cardinal direction)
  7. Visual signals (luminance, color, texture)
  9. 3x3 layout assembly
  10. Quantitative validation (margin, decision)
"""

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from utils.logger import logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DB_PATH = Path("data/puzzscan_v2.db")
PROFILE_VERSION: str = "piece_profile_v2"
CONTROL_POINTS_PER_FACE: int = 36
GEOMETRY_TRIM_SAMPLES: int = 4

DIRECTION_TO_CENTER_FACE: dict[str, str] = {
    "N": "1-2",
    "E": "2-3",
    "S": "3-4",
    "W": "4-1",
}

OPPOSITE_DIRECTION: dict[str, str] = {
    "N": "S",
    "E": "W",
    "S": "N",
    "W": "E",
}

CENTER_TO_CANDIDATE_FACE: dict[str, str] = {
    "1-2": "3-4",
    "2-3": "4-1",
    "3-4": "1-2",
    "4-1": "2-3",
}

_COMPATIBLE_TYPES: dict[str, str] = {
    "macho": "hembra",
    "hembra": "macho",
    "lisa": "lisa",
    "desconocida": "desconocida",
}

GEOMETRY_WEIGHT: float = 1.0
LUMINANCE_WEIGHT: float = 0.20
COLOR_WEIGHT: float = 0.20
TEXTURE_WEIGHT: float = 0.10

PENALTY_NEEDS_REVIEW: float = 0.05
PENALTY_UNKNOWN_FACE: float = 0.10
PENALTY_HIGH_RECONSTRUCTION: float = 0.03
PENALTY_PER_QUALITY_FLAG: float = 0.02
MAX_RECONSTRUCTION_RMSE: float = 0.03

DTW_WINDOW_RATIO: float = 0.05


# ---------------------------------------------------------------------------
# Phase 1: In-memory data model
# ---------------------------------------------------------------------------


@dataclass
class SearchFace:
    """A single face of a puzzle piece."""

    face_key: str
    face_type: str
    points: np.ndarray
    reduced_descriptor: np.ndarray
    metrics: dict[str, float]


@dataclass
class SearchPiece:
    """A puzzle piece with all four faces loaded from DB."""

    pieza_id: int
    nombre_puzzle: str
    origen: str
    piece_index: int
    ruta_imagen: str
    estado: str
    profile_status: str
    quality_flags: list[str]
    faces: dict[str, SearchFace]
    corner_points_source: np.ndarray
    corner_points_crop: np.ndarray
    piece_kind: str
    profile_version: str


@dataclass
class CandidateScore:
    """Score for a single candidate face match."""

    candidate_piece_id: int
    candidate_piece_index: int
    center_face_key: str
    candidate_face_key: str
    center_face_type: str
    candidate_face_type: str
    geometry_rmse: float
    geometry_dtw: float | None = None
    luminance_distance: float | None = None
    color_distance: float | None = None
    texture_distance: float | None = None
    score_total: float = 0.0
    penalties: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Phase 1: load_profiled_pieces
# ---------------------------------------------------------------------------


def _parse_faces_json(faces_json_raw: str | dict[str, Any]) -> dict[str, SearchFace]:
    """Parse faces_json into a dict of SearchFace."""
    raw: dict[str, Any] = (
        json.loads(faces_json_raw)
        if isinstance(faces_json_raw, str)
        else faces_json_raw
    )
    faces: dict[str, SearchFace] = {}
    for key, data in raw.items():
        points = np.array(data.get("points", []), dtype=np.float64)
        desc_raw = data.get("reduced_descriptor", [])
        desc = np.array(desc_raw, dtype=np.float64)
        if len(desc) > 0:
            desc = desc.reshape(-1, 2)
        faces[key] = SearchFace(
            face_key=key,
            face_type=data.get("tipo", "desconocida"),
            points=points,
            reduced_descriptor=desc,
            metrics=data.get("metrics", {}),
        )
    return faces


def load_profiled_pieces(
    puzzle: str | None = None,
    origen: str | None = None,
    profile_version: str = PROFILE_VERSION,
    allow_needs_review: bool = True,
) -> list[SearchPiece]:
    """Load profiled pieces from the database.

    Joins pieza_perfiles with sam3_piezas to obtain ruta_imagen and estado.
    Excludes invalid profiles.

    Args:
        puzzle: Filter by puzzle name.
        origen: Filter by source image stem.
        profile_version: Profile version to load.
        allow_needs_review: Whether to include needs_review profiles.

    Returns:
        List of SearchPiece objects.
    """
    query = """
        SELECT pp.*, sp.ruta_imagen, sp.estado
        FROM pieza_perfiles pp
        JOIN sam3_piezas sp ON pp.pieza_id = sp.id
        WHERE pp.profile_version = ?
    """
    params: list[Any] = [profile_version]

    if puzzle is not None:
        query += " AND pp.nombre_puzzle = ?"
        params.append(puzzle)
    if origen is not None:
        query += " AND pp.origen = ?"
        params.append(origen)

    query += " ORDER BY pp.pieza_id"

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(query, params).fetchall()
    conn.close()

    pieces: list[SearchPiece] = []
    for row in rows:
        profile_status = str(row["profile_status"])
        if profile_status == "invalid":
            continue
        if profile_status == "needs_review" and not allow_needs_review:
            continue

        faces = _parse_faces_json(str(row["faces_json"]))
        cp_source = np.array(
            json.loads(str(row["corner_points_source_json"])), dtype=np.float64
        )
        cp_crop = np.array(
            json.loads(str(row["corner_points_crop_json"])), dtype=np.float64
        )

        pieces.append(
            SearchPiece(
                pieza_id=int(row["pieza_id"]),
                nombre_puzzle=str(row["nombre_puzzle"]),
                origen=str(row["origen"]),
                piece_index=int(row["piece_index"]),
                ruta_imagen=str(row["ruta_imagen"]),
                estado=str(row["estado"]),
                profile_status=profile_status,
                quality_flags=json.loads(str(row["quality_flags_json"])),
                faces=faces,
                corner_points_source=cp_source,
                corner_points_crop=cp_crop,
                piece_kind=str(row["piece_kind"]),
                profile_version=str(row["profile_version"]),
            )
        )

    logger.info(
        "Loaded {} profiled pieces (puzzle={}, origen={}, version={})",
        len(pieces),
        puzzle,
        origen,
        profile_version,
    )
    return pieces


# ---------------------------------------------------------------------------
# Phase 2: Direction and face conventions
# ---------------------------------------------------------------------------


def face_for_direction(piece: SearchPiece, direction: str) -> SearchFace | None:
    """Return the face of *piece* that lies in the given cardinal direction."""
    face_key = DIRECTION_TO_CENTER_FACE.get(direction)
    if face_key is None:
        return None
    return piece.faces.get(face_key)


def candidate_face_for_direction(
    piece: SearchPiece, direction: str
) -> SearchFace | None:
    """Return the face of *piece* that would mate with the given direction.

    For a candidate placed in direction *direction* relative to the center,
    this is the face of the candidate that faces the center (opposite direction).
    """
    opp = OPPOSITE_DIRECTION.get(direction, "")
    return face_for_direction(piece, opp)


# ---------------------------------------------------------------------------
# Phase 3: Topological filter
# ---------------------------------------------------------------------------


def compatible_face_types(a: str, b: str, allow_unknown: bool = False) -> bool:
    """Return True if face types *a* and *b* are compatible for mating.

    macho <-> hembra, lisa <-> lisa.
    Unknown type is compatible only when *allow_unknown* is True.
    """
    if a == "desconocida" or b == "desconocida":
        return allow_unknown
    expected = _COMPATIBLE_TYPES.get(a)
    if expected is None:
        return False
    return b == expected


def filter_candidates_by_face_type(
    center_piece: SearchPiece,
    candidates: list[SearchPiece],
    direction: str,
    allow_unknown: bool = False,
) -> list[SearchPiece]:
    """Filter *candidates* to those whose mating face type is compatible."""
    center_face = face_for_direction(center_piece, direction)
    if center_face is None:
        return []

    result: list[SearchPiece] = []
    for cand in candidates:
        cand_face = candidate_face_for_direction(cand, direction)
        if cand_face is None:
            continue
        if compatible_face_types(
            center_face.face_type,
            cand_face.face_type,
            allow_unknown,
        ):
            result.append(cand)
    return result


# ---------------------------------------------------------------------------
# Phase 4: Curve inversion and geometric distance
# ---------------------------------------------------------------------------


def invert_face_curve(points: np.ndarray) -> np.ndarray:
    """Invert a normalized face curve for complementary comparison.

    The complement curve is reversed (endpoints swap), has Y inverted,
    and is re-mapped to the (0,0)-(1,0) interval.
    """
    pts = np.asarray(points, dtype=np.float64)
    result = pts[::-1].copy()
    result[:, 0] = 1.0 - result[:, 0]
    result[:, 1] = -result[:, 1]
    return result


def geometry_rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Compute point-to-point RMSE between two (N,2) curves."""
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def trim_curve_extremes(
    points: np.ndarray,
    trim_samples: int = GEOMETRY_TRIM_SAMPLES,
) -> np.ndarray:
    """Drop the low-confidence start/end points of a normalized face curve."""
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if trim_samples <= 0:
        return pts.copy()
    if n - 2 * trim_samples < 3:
        return pts.copy()
    return pts[trim_samples : n - trim_samples].copy()


def trimmed_geometry_rmse(
    a: np.ndarray,
    b: np.ndarray,
    trim_samples: int = GEOMETRY_TRIM_SAMPLES,
) -> float:
    """Compute RMSE after trimming an integer number of endpoint samples."""
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    return geometry_rmse(
        trim_curve_extremes(a, trim_samples),
        trim_curve_extremes(b, trim_samples),
    )


def score_face_pair(
    center_face: SearchFace,
    candidate_face: SearchFace,
) -> CandidateScore:
    """Score a pair of faces (center face vs inverted candidate face)."""
    inverted = invert_face_curve(candidate_face.reduced_descriptor)
    rmse = trimmed_geometry_rmse(center_face.reduced_descriptor, inverted)

    return CandidateScore(
        candidate_piece_id=0,
        candidate_piece_index=0,
        center_face_key=center_face.face_key,
        candidate_face_key=candidate_face.face_key,
        center_face_type=center_face.face_type,
        candidate_face_type=candidate_face.face_type,
        geometry_rmse=rmse,
        score_total=rmse * GEOMETRY_WEIGHT,
    )


# ---------------------------------------------------------------------------
# Phase 5: DTW diagnostic
# ---------------------------------------------------------------------------


def constrained_dtw_distance(
    a: np.ndarray,
    b: np.ndarray,
    window_ratio: float = DTW_WINDOW_RATIO,
) -> float:
    """Compute DTW distance with a Sakoe-Chiba band constraint.

    Args:
        a: (N, D) first sequence.
        b: (M, D) second sequence.
        window_ratio: Window size as fraction of min(N, M).

    Returns:
        DTW distance (lower = more similar).
    """
    n, m = a.shape[0], b.shape[0]
    window = max(1, int(min(n, m) * window_ratio))

    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0

    for i in range(1, n + 1):
        lo = max(1, i - window)
        hi = min(m + 1, i + window + 1)
        for j in range(lo, hi):
            cost = float(np.sqrt(np.sum((a[i - 1] - b[j - 1]) ** 2)))
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

    return float(dtw[n, m])


# ---------------------------------------------------------------------------
# Phase 6: Ranking
# ---------------------------------------------------------------------------


def _compute_penalties(face: SearchFace, piece: SearchPiece) -> float:
    """Compute penalty sum for a face match."""
    total = 0.0
    if piece.profile_status == "needs_review":
        total += PENALTY_NEEDS_REVIEW
    if face.face_type == "desconocida":
        total += PENALTY_UNKNOWN_FACE
    rmse = face.metrics.get("reconstruction_rmse", 0.0)
    if rmse > MAX_RECONSTRUCTION_RMSE:
        total += PENALTY_HIGH_RECONSTRUCTION
    total += len(piece.quality_flags) * PENALTY_PER_QUALITY_FLAG
    return total


def _penalty_tags(face: SearchFace, piece: SearchPiece) -> list[str]:
    """Human-readable list of penalty reasons."""
    tags: list[str] = []
    if piece.profile_status == "needs_review":
        tags.append("needs_review")
    if face.face_type == "desconocida":
        tags.append("unknown_face")
    if piece.quality_flags:
        tags.append(f"quality_flags({len(piece.quality_flags)})")
    return tags


def rank_candidates_for_direction(
    center_piece: SearchPiece,
    candidates: list[SearchPiece],
    direction: str,
    top_k: int = 3,
    compute_dtw: bool = False,
    allow_unknown: bool = True,
) -> list[CandidateScore]:
    """Rank compatible candidates for a single direction.

    Steps:
      1. Filter by face type compatibility.
      2. Exclude center piece.
      3. Compute geometry RMSE (with curve inversion).
      4. Optionally compute DTW.
      5. Apply penalties.
      6. Sort by ascending score and return top-k.

    Returns:
        List of CandidateScore sorted by score_total (best first).
    """
    center_face = face_for_direction(center_piece, direction)
    if center_face is None:
        return []

    center_desc = center_face.reduced_descriptor
    if len(center_desc) == 0:
        return []

    compatible = filter_candidates_by_face_type(
        center_piece,
        candidates,
        direction,
        allow_unknown=allow_unknown,
    )
    compatible = [c for c in compatible if c.pieza_id != center_piece.pieza_id]

    scores: list[CandidateScore] = []
    for cand in compatible:
        cand_face = candidate_face_for_direction(cand, direction)
        if cand_face is None or len(cand_face.reduced_descriptor) == 0:
            continue

        inverted = invert_face_curve(cand_face.reduced_descriptor)
        rmse = trimmed_geometry_rmse(center_desc, inverted)

        dtw_val: float | None = None
        if compute_dtw:
            dtw_val = constrained_dtw_distance(center_desc, inverted)

        penalty = _compute_penalties(cand_face, cand)
        score = rmse * GEOMETRY_WEIGHT + penalty

        scores.append(
            CandidateScore(
                candidate_piece_id=cand.pieza_id,
                candidate_piece_index=cand.piece_index,
                center_face_key=center_face.face_key,
                candidate_face_key=cand_face.face_key,
                center_face_type=center_face.face_type,
                candidate_face_type=cand_face.face_type,
                geometry_rmse=rmse,
                geometry_dtw=dtw_val,
                score_total=score,
                penalties=_penalty_tags(cand_face, cand),
            )
        )

    scores.sort(key=lambda s: s.score_total)
    return scores[:top_k]


def rank_all_cardinal_directions(
    center_piece: SearchPiece,
    candidates: list[SearchPiece],
    top_k: int = 3,
    compute_dtw: bool = False,
    allow_unknown: bool = True,
) -> dict[str, list[CandidateScore]]:
    """Rank candidates for all four cardinal directions.

    Returns:
        Dict mapping direction ("N", "E", "S", "W") to its ranked list.
    """
    result: dict[str, list[CandidateScore]] = {}
    for d in ("N", "E", "S", "W"):
        result[d] = rank_candidates_for_direction(
            center_piece,
            candidates,
            d,
            top_k=top_k,
            compute_dtw=compute_dtw,
            allow_unknown=allow_unknown,
        )
    return result


# ---------------------------------------------------------------------------
# Phase 7: Visual signals (luminance, colour, texture)
# ---------------------------------------------------------------------------

_FACE_CORNER_INDICES: dict[str, tuple[int, int]] = {
    "1-2": (0, 1),
    "2-3": (1, 2),
    "3-4": (2, 3),
    "4-1": (3, 0),
}


def _face_outward_normal(piece: SearchPiece, face_key: str) -> np.ndarray:
    """Return the outward-pointing unit normal for a face."""
    ci = _FACE_CORNER_INDICES.get(face_key)
    if ci is None:
        return np.array([0.0, -1.0])
    a = piece.corner_points_crop[ci[0]]
    b = piece.corner_points_crop[ci[1]]
    edge = b - a
    edge_len = np.linalg.norm(edge)
    if edge_len < 1e-12:
        return np.array([0.0, -1.0])
    edge_u = edge / edge_len
    normal = np.array([-edge_u[1], edge_u[0]])
    mid = (a + b) / 2.0
    centroid = piece.corner_points_crop.mean(axis=0)
    if np.dot(normal, centroid - mid) > 0:
        normal = -normal
    return normal


def _descriptor_to_crop(
    desc_pts: np.ndarray,
    corner_a: np.ndarray,
    corner_b: np.ndarray,
    outward_normal: np.ndarray,
    edge_length: float,
) -> np.ndarray:
    """Map (N,2) descriptor points from normalized to crop coordinates."""
    edge = corner_b - corner_a
    pts = np.zeros_like(desc_pts)
    for i, p in enumerate(desc_pts):
        pts[i] = corner_a + p[0] * edge + p[1] * outward_normal * edge_length
    return pts


def sample_face_band(
    image_rgba: np.ndarray,
    face_contour_crop: np.ndarray,
    inward_normal: np.ndarray,
    width_px: int = 8,
) -> np.ndarray:
    """Sample an inward band of pixels along the face contour.

    Returns:
        (n_points, width_px, 3) RGB values.
    """
    h, w = image_rgba.shape[:2]
    n_pts = len(face_contour_crop)
    band = np.zeros((n_pts, width_px, 3), dtype=np.float32)

    for i in range(n_pts):
        pt = face_contour_crop[i]
        for j in range(width_px):
            px = pt[0] + inward_normal[0] * j
            py = pt[1] + inward_normal[1] * j
            ix = int(np.clip(round(px), 0, w - 1))
            iy = int(np.clip(round(py), 0, h - 1))
            band[i, j] = image_rgba[iy, ix, :3].astype(np.float32)

    return band


def _resample_1d(signal: np.ndarray, n_points: int) -> np.ndarray:
    """Resample a 1D or 2D signal to *n_points* by linear interpolation."""
    src_len = len(signal)
    if src_len == n_points:
        return signal.copy()
    x_old = np.linspace(0.0, 1.0, src_len)
    x_new = np.linspace(0.0, 1.0, n_points)
    if signal.ndim == 1:
        interpolated = np.interp(x_new, x_old, signal)
        return np.asarray(interpolated)
    out = np.zeros((n_points, signal.shape[1]), dtype=signal.dtype)
    for j in range(signal.shape[1]):
        out[:, j] = np.interp(x_new, x_old, signal[:, j])
    return out


def luminance_profile(
    band: np.ndarray,
    n_points: int = CONTROL_POINTS_PER_FACE,
) -> np.ndarray:
    """Compute a luminance profile from an RGB band.

    Returns:
        (n_points,) array of average luminance values.
    """
    lum = 0.299 * band[..., 0] + 0.587 * band[..., 1] + 0.114 * band[..., 2]
    profile = lum.mean(axis=1)
    return _resample_1d(profile, n_points)


def color_profile_lab(
    band: np.ndarray,
    n_points: int = CONTROL_POINTS_PER_FACE,
) -> np.ndarray:
    """Compute a Lab colour profile from an RGB band.

    Returns:
        (n_points, 3) array of L, a, b values.
    """
    avg_rgb = band.mean(axis=1).astype(np.uint8)
    rgb_lin = avg_rgb.astype(np.float32) / 255.0
    mask = rgb_lin > 0.04045
    rgb_lin[mask] = ((rgb_lin[mask] + 0.055) / 1.055) ** 2.4
    rgb_lin[~mask] /= 12.92

    m = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float32,
    )
    xyz = rgb_lin @ m.T

    xn, yn, zn = 0.95047, 1.0, 1.08883
    xyz[:, 0] /= xn
    xyz[:, 1] /= yn
    xyz[:, 2] /= zn

    def _lab_f(t: np.ndarray) -> np.ndarray:
        result = np.zeros_like(t)
        m_ = t > 0.008856
        result[m_] = np.cbrt(t[m_])
        result[~m_] = 7.787 * t[~m_] + 16.0 / 116.0
        return result

    fx = _lab_f(xyz[:, 0])
    fy = _lab_f(xyz[:, 1])
    fz = _lab_f(xyz[:, 2])

    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b_val = 200.0 * (fy - fz)

    lab = np.column_stack([L, a, b_val])
    return _resample_1d(lab, n_points)


def texture_profile_gradient(
    band: np.ndarray,
    n_points: int = CONTROL_POINTS_PER_FACE,
) -> np.ndarray:
    """Compute a texture (gradient magnitude) profile from an RGB band.

    Returns:
        (n_points,) array of gradient magnitudes.
    """
    lum = 0.299 * band[..., 0] + 0.587 * band[..., 1] + 0.114 * band[..., 2]
    if band.shape[1] < 2:
        grad = np.zeros(len(lum))
    else:
        grad = np.abs(np.diff(lum, axis=1)).mean(axis=1)
    return _resample_1d(grad, n_points)


def luminance_distance(a: np.ndarray, b: np.ndarray) -> float:
    """RMSE between two luminance profiles."""
    return float(np.sqrt(np.mean((a - b) ** 2)))


def color_distance_lab(a: np.ndarray, b: np.ndarray) -> float:
    """RMSE between two Lab colour profiles (averaged across channels)."""
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def texture_distance(a: np.ndarray, b: np.ndarray) -> float:
    """RMSE between two texture profiles."""
    return float(np.sqrt(np.mean((a - b) ** 2)))


def compute_visual_scores(
    center_piece: SearchPiece,
    center_face_key: str,
    candidate: SearchPiece,
    candidate_face_key: str,
    image_cache: dict[int, np.ndarray] | None = None,
    band_width: int = 8,
) -> tuple[float, float, float]:
    """Compute luminance, colour, and texture distances for a face pair.

    Returns:
        (luminance_distance, colour_distance, texture_distance).
    """
    center_face = center_piece.faces.get(center_face_key)
    cand_face = candidate.faces.get(candidate_face_key)
    if center_face is None or cand_face is None:
        return (0.0, 0.0, 0.0)

    image = _load_or_cache_image(center_piece.ruta_imagen, image_cache)
    cand_image = _load_or_cache_image(candidate.ruta_imagen, image_cache)

    if image is None or cand_image is None:
        return (0.0, 0.0, 0.0)

    n = _face_outward_normal(center_piece, center_face_key)
    inward = -n
    ci = _FACE_CORNER_INDICES[center_face_key]
    a = center_piece.corner_points_crop[ci[0]]
    b = center_piece.corner_points_crop[ci[1]]
    edge_len = float(np.linalg.norm(b - a))

    center_pts = _descriptor_to_crop(
        center_face.reduced_descriptor,
        a,
        b,
        n,
        edge_len,
    )
    band_c = sample_face_band(image, center_pts, inward, band_width)

    n_cand = _face_outward_normal(candidate, candidate_face_key)
    cand_inward = -n_cand
    ci2 = _FACE_CORNER_INDICES[candidate_face_key]
    a2, b2 = candidate.corner_points_crop[ci2[0]], candidate.corner_points_crop[ci2[1]]
    edge_len2 = float(np.linalg.norm(b2 - a2))

    cand_pts = _descriptor_to_crop(
        cand_face.reduced_descriptor,
        a2,
        b2,
        n_cand,
        edge_len2,
    )
    band_cand = sample_face_band(cand_image, cand_pts, cand_inward, band_width)

    l1 = luminance_profile(band_c)
    l2 = luminance_profile(band_cand)
    c1 = color_profile_lab(band_c)
    c2 = color_profile_lab(band_cand)
    t1 = texture_profile_gradient(band_c)
    t2 = texture_profile_gradient(band_cand)

    return (
        luminance_distance(l1, l2),
        color_distance_lab(c1, c2),
        texture_distance(t1, t2),
    )


def _load_or_cache_image(
    path: str,
    cache: dict[int, np.ndarray] | None,
) -> np.ndarray | None:
    """Load an RGBA image, optionally using an id-keyed cache."""
    p = Path(path)
    if not p.exists():
        logger.warning("Image not found: {}", p)
        return None
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.shape[2] >= 3:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img
    return img_rgb


def rank_candidates_with_visual_signals(
    center_piece: SearchPiece,
    candidates: list[SearchPiece],
    direction: str,
    top_k: int = 3,
    compute_dtw: bool = False,
    use_dtw_for_ranking: bool = False,
    allow_unknown: bool = True,
) -> list[CandidateScore]:
    """Rank candidates by shape and attach visual signals as diagnostics.

    By default, score_total uses RMSE. When use_dtw_for_ranking=True, DTW is
    computed and used as the shape distance for ordering.
    """
    center_face = face_for_direction(center_piece, direction)
    if center_face is None:
        return []

    center_desc = center_face.reduced_descriptor
    if len(center_desc) == 0:
        return []

    compatible = filter_candidates_by_face_type(
        center_piece,
        candidates,
        direction,
        allow_unknown=allow_unknown,
    )
    compatible = [c for c in compatible if c.pieza_id != center_piece.pieza_id]

    scores: list[CandidateScore] = []
    for cand in compatible:
        cand_face = candidate_face_for_direction(cand, direction)
        if cand_face is None or len(cand_face.reduced_descriptor) == 0:
            continue

        inverted = invert_face_curve(cand_face.reduced_descriptor)
        rmse = trimmed_geometry_rmse(center_desc, inverted)

        dtw_val: float | None = None
        if compute_dtw or use_dtw_for_ranking:
            dtw_val = constrained_dtw_distance(center_desc, inverted)

        ld, cd, td = compute_visual_scores(
            center_piece,
            center_face.face_key,
            cand,
            cand_face.face_key,
        )

        penalty = _compute_penalties(cand_face, cand)
        shape_distance = (
            dtw_val if use_dtw_for_ranking and dtw_val is not None else rmse
        )
        score = shape_distance * GEOMETRY_WEIGHT + penalty

        scores.append(
            CandidateScore(
                candidate_piece_id=cand.pieza_id,
                candidate_piece_index=cand.piece_index,
                center_face_key=center_face.face_key,
                candidate_face_key=cand_face.face_key,
                center_face_type=center_face.face_type,
                candidate_face_type=cand_face.face_type,
                geometry_rmse=rmse,
                geometry_dtw=dtw_val,
                luminance_distance=ld,
                color_distance=cd,
                texture_distance=td,
                score_total=score,
                penalties=_penalty_tags(cand_face, cand),
            )
        )

    scores.sort(key=lambda s: s.score_total)
    return scores[:top_k]


# ---------------------------------------------------------------------------
# Phase 9: 3×3 layout assembly
# ---------------------------------------------------------------------------


def load_piece_rgba(piece: SearchPiece, load_fn: Any = None) -> np.ndarray | None:
    """Load a piece's PNG image preserving the alpha channel."""
    p = Path(piece.ruta_imagen)
    if not p.exists():
        logger.warning("Image not found: {}", p)
        return None
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.shape[2] == 4:
        rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    else:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgba = np.dstack([rgb, np.full(rgb.shape[:2], 255, dtype=np.uint8)])
    return rgba


def _composite_with_alpha(
    bg: np.ndarray,
    overlay: np.ndarray,
    x: int,
    y: int,
    scale: float = 1.0,
) -> None:
    """Composite an RGBA overlay onto a BGRA background at (x, y)."""
    if overlay.shape[2] < 4:
        return
    oh, ow = overlay.shape[:2]
    new_w = max(1, int(ow * scale))
    new_h = max(1, int(oh * scale))
    if scale != 1.0:
        resized = cv2.resize(overlay, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        resized = overlay

    y0, x0 = max(0, y), max(0, x)
    y1 = min(bg.shape[0], y + new_h)
    x1 = min(bg.shape[1], x + new_w)
    dy = y1 - y0
    dx = x1 - x0
    if dy <= 0 or dx <= 0:
        return

    oy0 = y0 - y
    ox0 = x0 - x
    alpha = resized[oy0 : oy0 + dy, ox0 : ox0 + dx, 3:4].astype(np.float32) / 255.0
    fg_rgb = resized[oy0 : oy0 + dy, ox0 : ox0 + dx, :3].astype(np.float32)
    bg_slice = bg[y0:y1, x0:x1].astype(np.float32)

    blended = fg_rgb * alpha + bg_slice * (1.0 - alpha)
    bg[y0:y1, x0:x1] = blended.astype(np.uint8)


def compose_3x3_layout(
    center_piece: SearchPiece,
    ranked: dict[str, list[CandidateScore]],
    all_pieces: dict[int, SearchPiece],
    cell_size: int = 300,
    padding: int = 10,
) -> np.ndarray:
    """Build a 3×3 grid image.

    Layout::

        NW  N  NE
         W  C  E
        SW  S  SE

    Direction-to-grid mapping:

        NW = candidate for direction W? or just empty by default
        Actually, this layout shows:
          - Center at (1,1)
          - N candidate at (1,0)
          - S candidate at (1,2)
          - E candidate at (2,1)
          - W candidate at (0,1)
        The diagonals (NW, NE, SW, SE) remain empty.

    Returns:
        (cell_size*3 + padding*4, cell_size*3 + padding*4, 3) uint8 RGB image.
    """
    grid_size = cell_size * 3 + padding * 4
    canvas = np.ones((grid_size, grid_size, 3), dtype=np.uint8) * 240

    def cell_origin(row: int, col: int) -> tuple[int, int]:
        x = padding + col * (cell_size + padding)
        y = padding + row * (cell_size + padding)
        return x, y

    position_map: dict[str, tuple[int, int]] = {
        "C": (1, 1),
        "N": (0, 1),
        "S": (2, 1),
        "E": (1, 2),
        "W": (1, 0),
    }

    for _pos, (row, col) in position_map.items():
        x0, y0 = cell_origin(row, col)
        cv2.rectangle(
            canvas,
            (x0, y0),
            (x0 + cell_size - 1, y0 + cell_size - 1),
            (200, 200, 200),
            -1,
        )

    center_rgba = load_piece_rgba(center_piece)
    if center_rgba is not None:
        cx, cy = cell_origin(1, 1)
        _composite_with_alpha(canvas, center_rgba, cx, cy, scale=0.85)

    for direction, (row, col) in position_map.items():
        if direction == "C":
            continue
        dir_scores = ranked.get(direction, [])
        if not dir_scores:
            continue
        top = dir_scores[0]
        cand_piece = all_pieces.get(top.candidate_piece_id)
        if cand_piece is None:
            continue
        cand_rgba = load_piece_rgba(cand_piece)
        if cand_rgba is not None:
            cx, cy = cell_origin(row, col)
            _composite_with_alpha(canvas, cand_rgba, cx, cy, scale=0.85)

    return canvas


# ---------------------------------------------------------------------------
# Phase 10: Quantitative validation helpers
# ---------------------------------------------------------------------------

DECISION_CONFIDENT = "confident"
DECISION_AMBIGUOUS = "ambiguous"
DECISION_NO_CANDIDATE = "no_candidate"
DECISION_LOW_QUALITY = "low_quality"


def score_margin(scores: list[CandidateScore]) -> float:
    """Return the margin between the first and second candidate scores.

    Returns infinity if fewer than 2 candidates.
    """
    if len(scores) < 2:
        return float("inf")
    return scores[1].score_total - scores[0].score_total


def classify_decision(
    scores: list[CandidateScore],
    margin_threshold: float = 0.05,
) -> str:
    """Classify the quality of a ranking decision.

    Returns one of 'confident', 'ambiguous', 'no_candidate'.
    """
    if not scores:
        return DECISION_NO_CANDIDATE
    if any(p == "needs_review" for s in scores for p in s.penalties):
        return DECISION_LOW_QUALITY
    if len(scores) >= 2:
        margin = scores[1].score_total - scores[0].score_total
        if margin < margin_threshold:
            return DECISION_AMBIGUOUS
    return DECISION_CONFIDENT


def summarize_rankings(
    rankings: dict[str, list[CandidateScore]],
    margin_threshold: float = 0.05,
) -> dict[str, dict[str, Any]]:
    """Produce a diagnostic summary for all four directions.

    Returns:
        Dict keyed by direction with fields:
          - top_k, geometry_rmse, score_total, margin, decision, etc.
    """
    summary: dict[str, dict[str, Any]] = {}
    for direction, scores in rankings.items():
        entry: dict[str, Any] = {
            "direction": direction,
            "n_candidates": len(scores),
            "decision": classify_decision(scores, margin_threshold),
            "scores": [],
        }
        for s in scores:
            entry["scores"].append(
                {
                    "candidate_id": s.candidate_piece_id,
                    "geometry_rmse": s.geometry_rmse,
                    "geometry_dtw": s.geometry_dtw,
                    "luminance_distance": s.luminance_distance,
                    "color_distance": s.color_distance,
                    "texture_distance": s.texture_distance,
                    "score_total": s.score_total,
                    "penalties": list(s.penalties),
                }
            )
        if len(scores) >= 2:
            entry["margin"] = scores[1].score_total - scores[0].score_total
        else:
            entry["margin"] = None
        summary[direction] = entry
    return summary
