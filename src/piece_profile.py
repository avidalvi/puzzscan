"""Geometric profile extraction for puzzle piece PNGs.

Pipeline: alpha mask → contour → corners → faces → normalized curves → profile.
"""

import json
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from dotenv import load_dotenv

from utils.logger import logger

load_dotenv()

# ---------------------------------------------------------------------------
# T1: Constants (overridable via .env)
# ---------------------------------------------------------------------------

PROFILE_VERSION: str = "piece_profile_v2"
POINTS_PER_FACE: int = 128
FOURIER_COEFFICIENTS: int = 16
CONTROL_POINTS_PER_FACE: int = 36
MIN_FACE_POINTS: int = 16
MIN_CLASSIFICATION_CONFIDENCE: float = 0.70
MAX_RECONSTRUCTION_RMSE: float = 0.03

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

FaceKey = str  # "1-2", "2-3", "3-4", "4-1"


@dataclass
class CornerCandidate:
    idx: int
    x: float
    y: float
    curvature: float
    convex: bool


@dataclass
class ProfileQuality:
    status: str  # "valid" | "needs_review" | "invalid"
    flags: list[str] = field(default_factory=list)
    area: float = 0.0
    num_components: int = 0
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    touches_border: bool = False


@dataclass
class FaceClassification:
    face_type: str  # "lisa" | "macho" | "hembra" | "desconocida"
    max_positive: float = 0.0
    max_negative: float = 0.0
    signed_area: float = 0.0
    peak_x: float = 0.0
    valley_x: float = 0.0
    classification_confidence: float = 0.0


@dataclass
class FaceProfile:
    face_key: FaceKey
    face_type: str
    points: np.ndarray  # (POINTS_PER_FACE, 2)
    reduced_descriptor: list[float]
    metrics: dict[str, float]
    source_contour_indices: list[int] = field(default_factory=list)


@dataclass
class PieceProfile:
    pieza_id: int
    pipeline: str
    nombre_puzzle: str
    origen: str
    piece_index: int
    corner_points: np.ndarray  # (4, 2) in crop coords
    corner_points_crop: np.ndarray
    corner_points_source: np.ndarray
    corner_points_normalized: np.ndarray
    face_order: str
    faces: dict[FaceKey, FaceProfile]
    piece_kind: str
    profile_status: str
    quality_flags: list[str]
    profile_version: str = PROFILE_VERSION


# ---------------------------------------------------------------------------
# T3: Alpha mask loading
# ---------------------------------------------------------------------------


def load_alpha_mask(image_path: Path) -> np.ndarray | None:
    """Load raw binary uint8 mask from PNG alpha channel.

    Returns None if the image cannot be read or has no alpha channel.
    """
    if not image_path.exists():
        logger.warning(f"Image not found: {image_path}")
        return None

    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        logger.warning(f"Could not read image: {image_path}")
        return None

    if img.shape[2] < 4:
        logger.warning(f"No alpha channel in {image_path}")
        return None

    alpha = img[:, :, 3]
    mask = (alpha > 0).astype(np.uint8) * 255
    return mask


def mask_from_contour(
    contour_json: list[list[int]],
    crop_shape: tuple[int, int],
    bounding_box_json: list[float],
) -> np.ndarray:
    """Build a binary mask from contour JSON (fallback when alpha is missing).

    Args:
        contour_json: List of [x, y] pairs in source-image coordinates.
        crop_shape: (height, width) of the cropped piece image.
        bounding_box_json: [x0, y0, x1, y1] in source-image coordinates.

    Returns:
        Binary uint8 mask of shape crop_shape.
    """
    bbox = bounding_box_json
    x0, y0 = int(bbox[0]), int(bbox[1])

    mask = np.zeros(crop_shape, dtype=np.uint8)
    pts = np.array(contour_json, dtype=np.int32).reshape((-1, 1, 2))
    translated = pts - np.array([x0, y0], dtype=np.int32)
    cv2.fillPoly(mask, [translated], 255)
    return mask


# ---------------------------------------------------------------------------
# T4: Mask validation
# ---------------------------------------------------------------------------


def validate_mask(mask: np.ndarray) -> ProfileQuality:
    """Check mask quality and return status + flags.

    Args:
        mask: Binary uint8 mask.

    Returns:
        ProfileQuality with status, flags, and metrics.
    """
    quality = ProfileQuality(status="valid")

    if mask is None or mask.sum() == 0:
        quality.status = "invalid"
        quality.flags.append("mask_empty")
        return quality

    area = int(mask.sum() / 255)
    quality.area = float(area)

    # Minimum area check
    if area < 100:
        quality.status = "invalid"
        quality.flags.append("area_too_small")
        return quality

    # Connected components
    num_labels, _ = cv2.connectedComponents(mask)
    quality.num_components = num_labels - 1  # exclude background

    if quality.num_components > 1:
        quality.flags.append("multiple_components")

    # Border touch
    h, w = mask.shape
    border_mask = np.zeros_like(mask)
    border_mask[0, :] = 1
    border_mask[-1, :] = 1
    border_mask[:, 0] = 1
    border_mask[:, -1] = 1
    touch_ratio = float((mask * border_mask).sum()) / 255 / max(1, area)
    quality.touches_border = touch_ratio > 0.05
    if quality.touches_border:
        quality.flags.append("touches_crop_border")

    nonzeros = cv2.findNonZero(mask)
    if nonzeros is not None:
        x, y, bw, bh = cv2.boundingRect(nonzeros)
        quality.bbox = (x, y, x + bw, y + bh)

    if quality.flags and quality.status == "valid":
        quality.status = "needs_review"

    return quality


# ---------------------------------------------------------------------------
# T5: Outer contour extraction and clockwise ordering
# ---------------------------------------------------------------------------


def extract_outer_contour(mask: np.ndarray) -> np.ndarray | None:
    """Extract the largest outer contour from a binary mask.

    Returns:
        (N, 2) array of contour points, or None if no contour found.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    pts = largest.squeeze()
    if pts.ndim == 1:
        pts = pts[np.newaxis, :]
    return pts.astype(np.float64)


def ensure_clockwise(contour: np.ndarray) -> np.ndarray:
    """Reverse contour if it is counter-clockwise (returns clockwise).

    Uses signed polygon area to determine orientation.
    In image coordinates (Y-down), clockwise polygons have negative area.
    """
    area = 0.0
    n = len(contour)
    for i in range(n):
        x1, y1 = contour[i]
        x2, y2 = contour[(i + 1) % n]
        area += (x2 - x1) * (y2 + y1)

    if area > 0:
        return contour[::-1]
    return contour


def contour_to_continuous_curve(
    contour: np.ndarray,
    n_points: int | None = None,
) -> np.ndarray:
    """Resample a closed contour as an evenly spaced continuous curve.

    The input contour comes from raster pixels. This function parameterizes it
    by cumulative arc length and interpolates X/Y coordinates, reducing the
    dependence of downstream curvature detection on pixel-step density.
    """
    if len(contour) < 3:
        return contour.astype(np.float64)

    pts = contour.astype(np.float64)
    target_points = n_points or len(pts)
    if target_points < 3:
        target_points = 3

    closed = np.vstack([pts, pts[0]])
    segment_lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    perimeter = cumulative[-1]
    if perimeter < 1e-12:
        return pts

    target = np.linspace(0.0, perimeter, target_points, endpoint=False)
    x = np.interp(target, cumulative, closed[:, 0])
    y = np.interp(target, cumulative, closed[:, 1])
    return np.column_stack([x, y])


def smooth_puzzle_contour(
    contour: np.ndarray,
    iterations: int = 2,
    n_points: int | None = None,
) -> np.ndarray:
    """Round raster spikes in a puzzle contour while keeping a closed curve.

    Uses Chaikin corner cutting followed by arc-length resampling. This is a
    geometry-level smoothing step: it removes pixel stair-steps and small
    spikes before corner detection, matching the expectation that puzzle edges
    are mostly rounded curves plus four principal corners.
    """
    if len(contour) < 3:
        return contour.astype(np.float64)

    smoothed = contour.astype(np.float64)
    for _ in range(max(0, iterations)):
        next_points: list[np.ndarray] = []
        for idx, point in enumerate(smoothed):
            next_point = smoothed[(idx + 1) % len(smoothed)]
            next_points.append(0.75 * point + 0.25 * next_point)
            next_points.append(0.25 * point + 0.75 * next_point)
        smoothed = np.array(next_points, dtype=np.float64)

    return contour_to_continuous_curve(smoothed, n_points or len(contour))


# ---------------------------------------------------------------------------
# T6: Corner detection
# ---------------------------------------------------------------------------


def _smooth_contour(contour: np.ndarray, window: int = 5) -> np.ndarray:
    """Smooth contour with a moving average."""
    if len(contour) < window:
        return contour
    kernel = np.ones(window) / window
    smoothed = np.zeros_like(contour)
    smoothed[:, 0] = np.convolve(
        np.pad(contour[:, 0], window // 2, mode="edge"),
        kernel,
        mode="valid",
    )
    smoothed[:, 1] = np.convolve(
        np.pad(contour[:, 1], window // 2, mode="edge"),
        kernel,
        mode="valid",
    )
    return smoothed


def _compute_curvature(contour: np.ndarray, window: int = 7) -> np.ndarray:
    """Compute local curvature at each point along the contour.

    Uses Menger curvature: 4 * area / (d1 * d2 * d3) for the
    triangle formed by (i-window, i, i+window).
    """
    n = len(contour)
    curvature = np.zeros(n, dtype=np.float64)

    for i in range(n):
        i_prev = (i - window) % n
        i_next = (i + window) % n

        a = contour[i_prev]
        b = contour[i]
        c = contour[i_next]

        d1 = np.linalg.norm(a - b)
        d2 = np.linalg.norm(b - c)
        d3 = np.linalg.norm(c - a)

        if d1 * d2 * d3 < 1e-12:
            curvature[i] = 0.0
            continue

        # Signed area of triangle (a, b, c)
        area = 0.5 * ((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
        # Menger curvature = 4 * area / (d1 * d2 * d3)
        # Positive area = convex (counter-clockwise triangle)
        # But we want convex corners on a clockwise contour
        curvature[i] = (4.0 * abs(area)) / (d1 * d2 * d3)
        # Sign: on a clockwise contour, convex corners have triangle area < 0
        if area < 0:
            curvature[i] = -curvature[i]

    return curvature


def detect_corner_candidates(
    contour: np.ndarray,
    curvature_window: int = 7,
    smooth_window: int = 5,
) -> list[CornerCandidate]:
    """Detect corner candidates based on curvature analysis.

    Returns a list of CornerCandidate objects sorted by decreasing curvature.
    """
    smoothed = _smooth_contour(contour, smooth_window)
    curvature = _compute_curvature(smoothed, curvature_window)

    candidates: list[CornerCandidate] = []
    n = len(contour)

    for i in range(n):
        if curvature[i] <= 0:
            continue
        # Local maximum check
        i_prev = (i - 1) % n
        i_next = (i + 1) % n
        if curvature[i] > curvature[i_prev] and curvature[i] >= curvature[i_next]:
            candidates.append(
                CornerCandidate(
                    idx=i,
                    x=float(contour[i, 0]),
                    y=float(contour[i, 1]),
                    curvature=float(curvature[i]),
                    convex=True,
                )
            )

    candidates.sort(key=lambda c: c.curvature, reverse=True)
    return candidates


def _select_four_corners(
    candidates: list[CornerCandidate],
    contour: np.ndarray,
    min_sep_ratio: float = 0.15,
) -> list[CornerCandidate] | None:
    """Select 4 corners from candidates maximising separation.

    Args:
        candidates: Sorted by descending curvature.
        contour: Full contour array (N, 2).
        min_sep_ratio: Minimum separation as fraction of contour length.

    Returns:
        Four selected corners or None if unable to pick 4.
    """
    if len(candidates) < 4:
        return None

    n = len(contour)
    total_len = n  # index distance along contour

    best: list[CornerCandidate] | None = None
    best_score = -1.0

    # Try all combinations of 4 from top candidates (up to 12)
    pool = candidates[: min(len(candidates), 12)]

    for combo in combinations(pool, 4):
        indices = sorted(c.idx for c in combo)
        # Separation: min circular distance between consecutive in sorted order
        sep = min(
            min(
                (indices[(j + 1) % 4] - indices[j]) % total_len,
                total_len - (indices[(j + 1) % 4] - indices[j]) % total_len,
            )
            for j in range(4)
        )
        if sep < total_len * min_sep_ratio:
            continue
        # Score: sum of curvatures + separation bonus
        score = sum(c.curvature for c in combo) + sep / total_len * 5
        if score > best_score:
            best_score = score
            best = list(combo)

    if best is None or len(best) < 4:
        return None

    return best


def _approx_poly_fallback(contour: np.ndarray) -> list[CornerCandidate] | None:
    """Fallback: use cv2.approxPolyDP adaptively to find 4 corners."""
    for epsilon_frac in np.linspace(0.005, 0.05, 20):
        epsilon = epsilon_frac * cv2.arcLength(contour.astype(np.float32), True)
        approx = cv2.approxPolyDP(contour.astype(np.float32), epsilon, True)
        if len(approx) == 4:
            corners: list[CornerCandidate] = []
            for pt in approx:
                x, y = pt[0]
                # Find closest index on original contour
                dists = np.sum((contour - np.array([x, y])) ** 2, axis=1)
                idx = int(np.argmin(dists))
                corners.append(
                    CornerCandidate(
                        idx=idx,
                        x=float(x),
                        y=float(y),
                        curvature=0.0,
                        convex=True,
                    )
                )
            return corners
    return None


def detect_corners(
    contour: np.ndarray,
) -> tuple[list[CornerCandidate], str]:
    """Detect four principal corners of a puzzle piece.

    Returns:
        Tuple of (corner_list, method_used) where method_used is
        "curvature" or "approx_poly_dp".
    """
    candidates = detect_corner_candidates(contour)
    corners = _select_four_corners(candidates, contour) if candidates else None

    if corners is not None:
        return corners, "curvature"

    fallback = _approx_poly_fallback(contour)
    if fallback is not None:
        return fallback, "approx_poly_dp"

    return [], "failed"


def order_corners_canonical(corners: np.ndarray) -> np.ndarray:
    """Order 4 corners as 1=top-left, 2=top-right, 3=bottom-right, 4=bottom-left.

    Args:
        corners: (4, 2) array of corner points.

    Returns:
        (4, 2) array in canonical order.
    """
    assert corners.shape == (4, 2), f"Expected (4, 2) corners, got {corners.shape}"

    # Split into top and bottom halves by y-coordinate
    sorted_by_y = corners[corners[:, 1].argsort()]
    top_two = sorted_by_y[:2]
    bottom_two = sorted_by_y[2:]

    top_left = top_two[top_two[:, 0].argmin()]
    top_right = top_two[top_two[:, 0].argmax()]
    bottom_left = bottom_two[bottom_two[:, 0].argmin()]
    bottom_right = bottom_two[bottom_two[:, 0].argmax()]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float64)


# ---------------------------------------------------------------------------
# T7: Split contour into faces
# ---------------------------------------------------------------------------


FACE_KEYS: list[FaceKey] = ["1-2", "2-3", "3-4", "4-1"]


def _corner_indices_on_contour(
    contour: np.ndarray,
    corners: np.ndarray,
) -> list[int]:
    """Find the closest contour index for each canonical corner."""
    indices: list[int] = []
    for corner in corners:
        dists = np.sum((contour - corner) ** 2, axis=1)
        idx = int(np.argmin(dists))
        indices.append(idx)
    return indices


def split_contour_into_faces(
    contour: np.ndarray,
    corner_indices: list[int],
) -> dict[FaceKey, np.ndarray]:
    """Split contour into 4 face segments between corners.

    Args:
        contour: (N, 2) clockwise contour array.
        corner_indices: [idx1, idx2, idx3, idx4] corresponding to corners 1-4.

    Returns:
        Dict with keys "1-2", "2-3", "3-4", "4-1" mapping to face point arrays.
    """
    faces: dict[FaceKey, np.ndarray] = {}

    for j, key in enumerate(FACE_KEYS):
        start = corner_indices[j]
        end_idx = corner_indices[(j + 1) % 4]

        if end_idx > start:
            face_pts = contour[start : end_idx + 1]
        else:
            face_pts = np.vstack([contour[start:], contour[: end_idx + 1]])

        faces[key] = face_pts

    return faces


# ---------------------------------------------------------------------------
# T8: Normalized face curves
# ---------------------------------------------------------------------------


def _signed_area(pts: np.ndarray) -> float:
    """Signed polygon area (negative = clockwise in image coords Y-down)."""
    n = len(pts)
    area = 0.0
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        area += (x2 - x1) * (y2 + y1)
    return float(area / 2.0)


def normalize_face_curve(
    face_points: np.ndarray,
    start_corner: np.ndarray,
    end_corner: np.ndarray,
    piece_centroid: np.ndarray,
) -> np.ndarray:
    """Normalize a face as a scaled parametric curve.

    Y positive points outward from the piece centroid.

    Args:
        face_points: (M, 2) array of contour points for this face.
        start_corner: (2,) start corner point.
        end_corner: (2,) end corner point.
        piece_centroid: (2,) centroid of the piece mask.

    Returns:
        (POINTS_PER_FACE, 2) normalized curve where:
            start corner is (0, 0), end corner is (1, 0), and intermediate
            points preserve the face contour order scaled by edge length.
    """
    edge_vec = end_corner - start_corner
    edge_len = np.linalg.norm(edge_vec)
    if edge_len < 1e-12:
        edge_len = np.float64(1.0)

    edge_unit = edge_vec / edge_len
    normal = np.array([-edge_unit[1], edge_unit[0]])  # perpendicular

    # Determine outward direction: normal points away from centroid
    mid = (start_corner + end_corner) / 2.0
    centroid_to_mid = mid - piece_centroid
    if np.dot(normal, centroid_to_mid) < 0:
        normal = -normal  # flip so normal points outward

    # Project face points in contour order. Do not sort by X: puzzle tabs and
    # holes are not necessarily single-valued functions Y=f(X).
    projected: list[list[float]] = []
    for pt in face_points:
        vec = pt - start_corner
        x_proj = np.dot(vec, edge_unit) / edge_len
        y_perp = np.dot(vec, normal) / edge_len
        projected.append([float(x_proj), float(y_perp)])

    projected_arr = np.array(projected, dtype=np.float64)
    if len(projected_arr) < 2:
        projected_arr = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)

    projected_arr[0] = [0.0, 0.0]
    projected_arr[-1] = [1.0, 0.0]

    segment_lengths = np.linalg.norm(np.diff(projected_arr, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total_len = cumulative[-1]
    if total_len < 1e-12:
        curve = np.column_stack(
            [np.linspace(0.0, 1.0, POINTS_PER_FACE), np.zeros(POINTS_PER_FACE)]
        )
        return curve

    target = np.linspace(0.0, total_len, POINTS_PER_FACE)
    x_target = np.interp(target, cumulative, projected_arr[:, 0])
    y_target = np.interp(target, cumulative, projected_arr[:, 1])
    curve = np.column_stack([x_target, y_target])
    curve[0] = [0.0, 0.0]
    curve[-1] = [1.0, 0.0]

    return curve


# ---------------------------------------------------------------------------
# T9: Face classification and piece type
# ---------------------------------------------------------------------------


def classify_face(curve: np.ndarray) -> FaceClassification:
    """Classify a normalized face curve.

    Args:
        curve: (POINTS_PER_FACE, 2) normalized curve.

    Returns:
        FaceClassification with type and metrics.
    """
    ys = curve[:, 1]
    max_pos = float(np.max(ys))
    max_neg = float(np.min(ys))
    signed_area = float(np.trapezoid(ys, curve[:, 0]))
    amplitude = max_pos - max_neg

    # Peak/valley positions
    peak_x = float(curve[np.argmax(ys), 0]) if max_pos > 0 else 0.0
    valley_x = float(curve[np.argmin(ys), 0]) if max_neg < 0 else 0.0

    # Classification logic
    if amplitude < 0.02:
        face_type = "lisa"
        confidence = max(0.0, 1.0 - amplitude / 0.02)
    elif abs(max_pos) > abs(max_neg) * 1.5 and max_pos > 0.01:
        face_type = "macho"
        confidence = min(1.0, abs(max_pos) / max(0.01, amplitude))
    elif abs(max_neg) > abs(max_pos) * 1.5 and max_neg < -0.01:
        face_type = "hembra"
        confidence = min(1.0, abs(max_neg) / max(0.01, amplitude))
    else:
        face_type = "desconocida"
        confidence = 0.5

    return FaceClassification(
        face_type=face_type,
        max_positive=max_pos,
        max_negative=max_neg,
        signed_area=signed_area,
        peak_x=peak_x,
        valley_x=valley_x,
        classification_confidence=confidence,
    )


def derive_piece_kind(face_types: dict[FaceKey, str]) -> str:
    """Derive piece kind from the four face classifications.

    - interior: 0 flat (lisa) faces
    - border: 1 flat face
    - corner: 2 consecutive flat faces
    - unknown: anything else

    Args:
        face_types: Dict mapping face keys to type strings.

    Returns:
        Piece kind string.
    """
    face_keys: list[FaceKey] = ["1-2", "2-3", "3-4", "4-1"]
    is_flat = [face_types[k] == "lisa" for k in face_keys]
    n_flat = sum(is_flat)

    if n_flat == 0:
        return "interior"
    if n_flat == 1:
        return "border"
    if n_flat == 2:
        # Check if they are consecutive
        for j in range(4):
            if is_flat[j] and is_flat[(j + 1) % 4]:
                return "corner"
        return "unknown"
    return "unknown"


# ---------------------------------------------------------------------------
# T10: Fourier descriptor
# ---------------------------------------------------------------------------


def fourier_descriptor(
    curve: np.ndarray, n_coefficients: int = FOURIER_COEFFICIENTS
) -> list[float]:
    """Compute complex Fourier descriptor for a parametric face curve.

    Args:
        curve: (POINTS_PER_FACE, 2) normalized curve.
        n_coefficients: Number of frequency components to keep.

    Returns:
        List of real/imag pairs for DC plus low positive and negative
        frequencies of z(s)=x(s)+i*y(s), normalized by the number of samples.
    """
    z = curve[:, 0] + 1j * curve[:, 1]
    spectrum = np.fft.fft(z) / len(z)
    descriptor: list[float] = []
    frequency_indices = [0]
    for k in range(1, n_coefficients):
        frequency_indices.extend([k, -k])

    for idx in frequency_indices:
        coefficient = spectrum[idx]
        descriptor.append(float(coefficient.real))
        descriptor.append(float(coefficient.imag))
    return descriptor


def reconstruct_curve_from_fourier(
    descriptor: list[float],
    points_per_face: int = POINTS_PER_FACE,
) -> np.ndarray:
    """Reconstruct a parametric face curve from complex Fourier descriptor.

    Args:
        descriptor: [real_0, imag_0, real_1, imag_1, ...].
        points_per_face: Number of output points.

    Returns:
        (points_per_face, 2) reconstructed curve.
    """
    n_coefficients = (len(descriptor) // 2 + 1) // 2
    spectrum = np.zeros(points_per_face, dtype=np.complex128)
    frequency_indices = [0]
    for k in range(1, n_coefficients):
        frequency_indices.extend([k, -k])

    for pos, idx in enumerate(frequency_indices):
        spectrum[idx] = complex(descriptor[2 * pos], descriptor[2 * pos + 1])

    z = np.fft.ifft(spectrum * points_per_face)

    return np.column_stack([z.real, z.imag])


def control_point_descriptor(
    curve: np.ndarray,
    n_control_points: int = CONTROL_POINTS_PER_FACE,
) -> list[float]:
    """Reduce an open parametric face curve to arc-length control points."""
    if len(curve) < 2:
        control = np.zeros((n_control_points, 2), dtype=np.float64)
        control[-1] = [1.0, 0.0]
        return control.ravel().tolist()

    segment_lengths = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total_len = cumulative[-1]
    if total_len < 1e-12:
        control = np.column_stack(
            [np.linspace(0.0, 1.0, n_control_points), np.zeros(n_control_points)]
        )
        return control.ravel().tolist()

    target = np.linspace(0.0, total_len, n_control_points)
    x = np.interp(target, cumulative, curve[:, 0])
    y = np.interp(target, cumulative, curve[:, 1])
    control = np.column_stack([x, y])
    control[0] = [0.0, 0.0]
    control[-1] = [1.0, 0.0]
    return control.ravel().tolist()


def reconstruct_curve_from_control_points(
    descriptor: list[float],
    points_per_face: int = POINTS_PER_FACE,
) -> np.ndarray:
    """Reconstruct an open parametric curve from arc-length control points."""
    control = np.array(descriptor, dtype=np.float64).reshape((-1, 2))
    if len(control) < 2:
        return np.column_stack(
            [np.linspace(0.0, 1.0, points_per_face), np.zeros(points_per_face)]
        )

    segment_lengths = np.linalg.norm(np.diff(control, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total_len = cumulative[-1]
    if total_len < 1e-12:
        return np.column_stack(
            [np.linspace(0.0, 1.0, points_per_face), np.zeros(points_per_face)]
        )

    target = np.linspace(0.0, total_len, points_per_face)
    x = np.interp(target, cumulative, control[:, 0])
    y = np.interp(target, cumulative, control[:, 1])
    curve = np.column_stack([x, y])
    curve[0] = [0.0, 0.0]
    curve[-1] = [1.0, 0.0]
    return curve


# ---------------------------------------------------------------------------
# T11: Profile assembly
# ---------------------------------------------------------------------------


def _piece_centroid(mask: np.ndarray) -> np.ndarray:
    """Compute centroid of a binary mask."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.array([0.0, 0.0])
    return np.array([float(np.mean(xs)), float(np.mean(ys))])


def _load_mask_with_fallback(
    ruta_imagen: Path,
    contour_json: list[list[int]],
    bbox_json: list[float],
    quality_flags: list[str],
) -> np.ndarray:
    """Load alpha mask with fallback to contour JSON."""
    mask = load_alpha_mask(ruta_imagen)

    if mask is None:
        if ruta_imagen.exists():
            fallback_img = cv2.imread(str(ruta_imagen), cv2.IMREAD_UNCHANGED)
            shape = fallback_img.shape[:2] if fallback_img is not None else (1, 1)
            crop_h, crop_w = shape
        else:
            crop_h, crop_w = 1, 1
        mask = mask_from_contour(contour_json, (crop_h, crop_w), bbox_json)
        quality_flags.append("missing_alpha")
        quality_flags.append("fallback_contour_json")

    assert mask is not None
    return mask


def _invalid_profile(
    pieza_id: int,
    pipeline: str,
    nombre_puzzle: str,
    origen: str,
    piece_index: int,
    face_order: str,
    quality_flags: list[str],
) -> PieceProfile:
    """Build an invalid PieceProfile with empty geometry."""
    return PieceProfile(
        pieza_id=pieza_id,
        pipeline=pipeline,
        nombre_puzzle=nombre_puzzle,
        origen=origen,
        piece_index=piece_index,
        corner_points=np.zeros((4, 2), dtype=np.float64),
        corner_points_crop=np.zeros((4, 2), dtype=np.float64),
        corner_points_source=np.zeros((4, 2), dtype=np.float64),
        corner_points_normalized=np.zeros((4, 2), dtype=np.float64),
        face_order=face_order,
        faces={},
        piece_kind="unknown",
        profile_status="invalid",
        quality_flags=quality_flags,
    )


def _process_faces(
    face_contours: dict[FaceKey, np.ndarray],
    corner_idx: list[int],
    corners_canonical: np.ndarray,
    centroid: np.ndarray,
    contour: np.ndarray,
    median_len: float,
    quality_flags: list[str],
) -> tuple[dict[FaceKey, FaceProfile], dict[FaceKey, str]]:
    """Validate face lengths and process each face."""
    faces_out: dict[FaceKey, FaceProfile] = {}
    face_types: dict[FaceKey, str] = {}

    for j, key in enumerate(FACE_KEYS):
        pts = face_contours[key]

        if len(pts) < MIN_FACE_POINTS:
            quality_flags.append(f"degenerate_face_{key}")
        if median_len > 0:
            ratio = len(pts) / median_len
            if ratio < 0.5 or ratio > 1.8:
                quality_flags.append(f"face_length_ratio_{key}")

        face, ftype = _process_one_face(
            j,
            key,
            pts,
            corner_idx,
            corners_canonical,
            centroid,
            contour,
        )
        faces_out[key] = face
        face_types[key] = ftype
        if face.metrics["reconstruction_rmse"] > MAX_RECONSTRUCTION_RMSE:
            quality_flags.append(f"high_reconstruction_error_{key}")

    return faces_out, face_types


def _process_one_face(
    j: int,
    key: FaceKey,
    face_pts: np.ndarray,
    corner_idx: list[int],
    corners_canonical: np.ndarray,
    centroid: np.ndarray,
    contour: np.ndarray,
) -> tuple[FaceProfile, str]:
    """Normalize, classify, and compute descriptor for one face."""
    start_corner = corners_canonical[j]
    end_corner = corners_canonical[(j + 1) % 4]

    curve = normalize_face_curve(face_pts, start_corner, end_corner, centroid)
    classification = classify_face(curve)
    descriptor = control_point_descriptor(curve)

    reconstructed = reconstruct_curve_from_control_points(descriptor)
    rmse = float(np.sqrt(np.mean(np.sum((curve - reconstructed) ** 2, axis=1))))

    start_idx = corner_idx[j]
    end_idx_val = corner_idx[(j + 1) % 4]
    if end_idx_val >= start_idx:
        src_indices = list(range(start_idx, end_idx_val + 1))
    else:
        src_indices = list(range(start_idx, len(contour))) + list(
            range(0, end_idx_val + 1)
        )

    face = FaceProfile(
        face_key=key,
        face_type=classification.face_type,
        points=curve,
        reduced_descriptor=descriptor,
        metrics={
            "max_positive": classification.max_positive,
            "max_negative": classification.max_negative,
            "signed_area": classification.signed_area,
            "peak_x": classification.peak_x,
            "valley_x": classification.valley_x,
            "classification_confidence": classification.classification_confidence,
            "reconstruction_rmse": rmse,
            "num_points": len(face_pts),
        },
        source_contour_indices=src_indices,
    )

    return face, classification.face_type


def build_piece_profile(piece_row: dict[str, Any]) -> PieceProfile:
    """Build a full piece profile from a sam3_piezas row.

    Args:
        piece_row: Dictionary with keys from sam3_piezas table.

    Returns:
        Populated PieceProfile dataclass.
    """
    pieza_id = int(piece_row["id"])
    nombre_puzzle = str(piece_row["nombre_puzzle"])
    origen = str(piece_row["origen"])
    piece_index = int(piece_row["piece_index"])
    ruta_imagen = Path(str(piece_row["ruta_imagen"]))
    contour_json_raw = piece_row["contour_json"]
    bbox_json_raw = piece_row["bounding_box_json"]

    contour_json: list[list[int]] = (
        json.loads(contour_json_raw)
        if isinstance(contour_json_raw, str)
        else contour_json_raw
    )
    bbox_json: list[float] = (
        json.loads(bbox_json_raw) if isinstance(bbox_json_raw, str) else bbox_json_raw
    )

    pipeline = "sam3"
    face_order = "1-2-3-4-1"
    quality_flags: list[str] = []
    base = {
        "pieza_id": pieza_id,
        "pipeline": pipeline,
        "nombre_puzzle": nombre_puzzle,
        "origen": origen,
        "piece_index": piece_index,
        "face_order": face_order,
    }

    # Step 1: Load mask
    mask = _load_mask_with_fallback(ruta_imagen, contour_json, bbox_json, quality_flags)

    # Step 2: Validate mask
    quality = validate_mask(mask)
    quality_flags.extend(quality.flags)

    if quality.status == "invalid":
        return _invalid_profile(**base, quality_flags=quality_flags)  # type: ignore[arg-type]

    # Step 3: Extract outer contour
    contour = extract_outer_contour(mask)
    if contour is None or len(contour) < 20:
        quality_flags.append("contour_not_found")
        return _invalid_profile(**base, quality_flags=quality_flags)  # type: ignore[arg-type]

    # Step 4: Ensure clockwise and smooth as a continuous puzzle curve
    contour = ensure_clockwise(contour)
    contour = contour_to_continuous_curve(contour)
    contour = smooth_puzzle_contour(contour)

    # Step 5: Detect corners
    corner_list, _corner_method = detect_corners(contour)

    if len(corner_list) < 4:
        quality_flags.append("corner_detection_failed")
        return _invalid_profile(**base, quality_flags=quality_flags)  # type: ignore[arg-type]

    corners_arr = np.array([[c.x, c.y] for c in corner_list], dtype=np.float64)
    corners_canonical = order_corners_canonical(corners_arr)

    # Step 6: Corner coordinate systems
    bbox_x0, bbox_y0 = int(bbox_json[0]), int(bbox_json[1])
    corner_points_crop = corners_canonical.copy()
    corner_points_source = corner_points_crop + np.array([bbox_x0, bbox_y0])
    h, w = mask.shape
    corner_points_normalized = corner_points_crop / np.array([w, h])

    # Step 7: Split into faces
    corner_idx = _corner_indices_on_contour(contour, corners_canonical)
    face_contours = split_contour_into_faces(contour, corner_idx)

    # Step 8: Validate face lengths and process each face
    centroid = _piece_centroid(mask)
    face_lengths = [len(pts) for pts in face_contours.values()]
    median_len = float(np.median(face_lengths)) if face_lengths else 0

    faces_out, face_types = _process_faces(
        face_contours,
        corner_idx,
        corners_canonical,
        centroid,
        contour,
        median_len,
        quality_flags,
    )

    # Step 9: Derive piece kind
    piece_kind = derive_piece_kind(face_types)

    # Step 11: Determine profile status
    if quality_flags:
        profile_status = "needs_review"
    else:
        profile_status = "valid"

    if (
        "corner_detection_failed" in quality_flags
        or "contour_not_found" in quality_flags
    ):
        profile_status = "invalid"

    return PieceProfile(
        pieza_id=pieza_id,
        pipeline=pipeline,
        nombre_puzzle=nombre_puzzle,
        origen=origen,
        piece_index=piece_index,
        corner_points=corners_canonical,
        corner_points_crop=corner_points_crop,
        corner_points_source=corner_points_source,
        corner_points_normalized=corner_points_normalized,
        face_order=face_order,
        faces=faces_out,
        piece_kind=piece_kind,
        profile_status=profile_status,
        quality_flags=quality_flags,
    )


def profile_to_db_dict(profile: PieceProfile) -> dict[str, Any]:
    """Convert a PieceProfile to a dict suitable for insert_piece_profile."""
    faces_json: dict[str, Any] = {}
    for key, face in profile.faces.items():
        faces_json[key] = {
            "tipo": face.face_type,
            "points": face.points.tolist(),
            "reduced_descriptor": face.reduced_descriptor,
            "metrics": face.metrics,
            "source_contour_indices": face.source_contour_indices,
        }

    return {
        "pieza_id": profile.pieza_id,
        "pipeline": profile.pipeline,
        "nombre_puzzle": profile.nombre_puzzle,
        "origen": profile.origen,
        "piece_index": profile.piece_index,
        "corner_points_json": profile.corner_points.tolist(),
        "corner_points_crop_json": profile.corner_points_crop.tolist(),
        "corner_points_source_json": profile.corner_points_source.tolist(),
        "corner_points_normalized_json": profile.corner_points_normalized.tolist(),
        "face_order": profile.face_order,
        "faces_json": faces_json,
        "piece_kind": profile.piece_kind,
        "profile_status": profile.profile_status,
        "quality_flags_json": profile.quality_flags,
        "profile_version": profile.profile_version,
    }
