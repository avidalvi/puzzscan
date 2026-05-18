"""Unit tests for src/piece_profile.py geometry functions."""

from pathlib import Path

import cv2
import numpy as np

from src.piece_profile import (
    POINTS_PER_FACE,
    CornerCandidate,
    FaceProfile,
    PieceProfile,
    _approx_poly_fallback,
    _corner_indices_on_contour,
    _select_four_corners,
    _signed_area,
    _smooth_contour,
    classify_face,
    contour_to_continuous_curve,
    control_point_descriptor,
    derive_piece_kind,
    detect_corner_candidates,
    detect_corners,
    ensure_clockwise,
    extract_outer_contour,
    fourier_descriptor,
    load_alpha_mask,
    mask_from_contour,
    normalize_face_curve,
    order_corners_canonical,
    profile_to_db_dict,
    reconstruct_curve_from_control_points,
    reconstruct_curve_from_fourier,
    smooth_puzzle_contour,
    split_contour_into_faces,
    validate_mask,
)

# ============================================================
# T14.1 — Alpha mask
# ============================================================


def test_load_alpha_mask_rgba(tmp_path: Path) -> None:
    """A valid RGBA PNG should produce a binary uint8 mask."""
    img = np.zeros((50, 50, 4), dtype=np.uint8)
    img[10:40, 10:40, 3] = 255  # alpha square
    img[:, :, :3] = [200, 150, 100]
    path = tmp_path / "test.png"
    cv2.imwrite(str(path), img)

    mask = load_alpha_mask(path)
    assert mask is not None
    assert mask.dtype == np.uint8
    assert mask.shape == (50, 50)
    assert mask.sum() > 0


def test_load_alpha_mask_no_alpha(tmp_path: Path) -> None:
    """A PNG without alpha channel should return None."""
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    path = tmp_path / "no_alpha.png"
    cv2.imwrite(str(path), img)

    mask = load_alpha_mask(path)
    assert mask is None


def test_load_alpha_mask_nonexistent() -> None:
    """A non-existent path should return None without crashing."""
    mask = load_alpha_mask(Path("/nonexistent/path.png"))
    assert mask is None


# ============================================================
# T14.2 — Orientation
# ============================================================


def test_ensure_clockwise_preserves_clockwise() -> None:
    """A clockwise contour must be returned unchanged."""
    # Clockwise square
    contour = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64)
    result = ensure_clockwise(contour)
    assert np.array_equal(result, contour)


def test_ensure_clockwise_reverses_counter_clockwise() -> None:
    """A counter-clockwise contour must be reversed."""
    contour = np.array([[0, 0], [0, 10], [10, 10], [10, 0]], dtype=np.float64)
    result = ensure_clockwise(contour)
    # Should be reversed
    assert np.array_equal(result, contour[::-1])


# ============================================================
# T14.3 — Corner ordering
# ============================================================


def test_order_corners_canonical() -> None:
    """Unordered corners should be converted to 1=TL, 2=TR, 3=BR, 4=BL."""
    # Given in random order: BR, TL, BL, TR
    corners = np.array(
        [[10.0, 10.0], [0.0, 0.0], [0.0, 10.0], [10.0, 0.0]], dtype=np.float64
    )
    ordered = order_corners_canonical(corners)
    expected = np.array(
        [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], dtype=np.float64
    )
    assert np.allclose(ordered, expected)


# ============================================================
# T14.4 — Curve normalization
# ============================================================


def test_normalize_face_straight_line() -> None:
    """A straight face line should produce Y ~= 0."""
    face = np.linspace([0, 0], [10, 0], 20)
    start = np.array([0.0, 0.0])
    end = np.array([10.0, 0.0])
    centroid = np.array([5.0, -5.0])  # below face → outward Y positive

    curve = normalize_face_curve(face, start, end, centroid)
    assert curve.shape == (POINTS_PER_FACE, 2)
    assert np.allclose(curve[0], [0.0, 0.0], atol=1e-6)
    assert np.allclose(curve[-1], [1.0, 0.0], atol=1e-6)
    assert np.max(np.abs(curve[:, 1])) < 1e-6


def test_normalize_face_outward_positive() -> None:
    """A face with an outward bump should have Y > 0."""
    n = 30
    xs = np.linspace(0, 10, n)
    ys = np.zeros(n)
    ys[n // 3 : 2 * n // 3] = 2.0  # bump outward
    face = np.column_stack([xs, ys])
    start = np.array([0.0, 0.0])
    end = np.array([10.0, 0.0])
    centroid = np.array([5.0, -5.0])  # below face

    curve = normalize_face_curve(face, start, end, centroid)
    assert np.any(curve[:, 1] > 0), "Outward bump should give Y > 0"


def test_normalize_face_inward_negative() -> None:
    """A face with an inward dent (toward centroid) should have Y < 0."""
    n = 30
    xs = np.linspace(0, 10, n)
    ys = np.zeros(n)
    ys[n // 3 : 2 * n // 3] = -2.0  # dent above face line
    face = np.column_stack([xs, ys])
    start = np.array([0.0, 0.0])
    end = np.array([10.0, 0.0])
    centroid = np.array([5.0, -5.0])  # above face → interior is above

    curve = normalize_face_curve(face, start, end, centroid)
    assert np.any(curve[:, 1] < 0), "Inward dent should give Y < 0"


# ============================================================
# T14.5 — Resampling
# ============================================================


def test_resample_always_poins_per_face() -> None:
    """Any face should be resampled to exactly POINTS_PER_FACE points."""
    face = np.random.rand(50, 2).astype(np.float64) * 10
    start = face[0]
    end = face[-1]
    centroid = np.array([5.0, 5.0])

    curve = normalize_face_curve(face, start, end, centroid)
    assert curve.shape == (POINTS_PER_FACE, 2)


def test_resample_preserves_parametric_backtracking() -> None:
    """Normalized tabs/holes may have repeated or decreasing X values."""
    face = np.array(
        [
            [0.0, 0.0],
            [4.0, 0.0],
            [4.0, 3.0],
            [6.0, 3.0],
            [6.0, 0.0],
            [10.0, 0.0],
        ],
        dtype=np.float64,
    )
    start = np.array([0.0, 0.0])
    end = np.array([10.0, 0.0])
    centroid = np.array([5.0, -5.0])

    curve = normalize_face_curve(face, start, end, centroid)
    assert curve.shape == (POINTS_PER_FACE, 2)
    assert np.max(curve[:, 1]) > 0.2


def test_resample_endpoints() -> None:
    """Resampled curve endpoints must be at (0,0) and (1,0)."""
    face = np.random.rand(50, 2).astype(np.float64) * 10
    start = face[0]
    end = face[-1]
    centroid = np.array([5.0, 5.0])

    curve = normalize_face_curve(face, start, end, centroid)
    assert np.allclose(curve[0], [0.0, 0.0], atol=1e-6)
    assert np.allclose(curve[-1], [1.0, 0.0], atol=1e-6)


# ============================================================
# T14.6 — Face classification
# ============================================================


def test_classify_flat_curve() -> None:
    """A flat curve (all Y ~= 0) should be classified as 'lisa'."""
    curve = np.column_stack(
        [np.linspace(0, 1, POINTS_PER_FACE), np.zeros(POINTS_PER_FACE)]
    )
    result = classify_face(curve)
    assert result.face_type == "lisa"
    assert result.classification_confidence > 0


def test_classify_positive_curve() -> None:
    """A curve with dominant positive peak should be 'macho'."""
    xs = np.linspace(0, 1, POINTS_PER_FACE)
    ys = np.exp(-((xs - 0.5) ** 2) / 0.01) * 0.1  # sharp positive peak
    curve = np.column_stack([xs, ys])
    result = classify_face(curve)
    assert result.face_type == "macho"


def test_classify_negative_curve() -> None:
    """A curve with dominant negative valley should be 'hembra'."""
    xs = np.linspace(0, 1, POINTS_PER_FACE)
    ys = -np.exp(-((xs - 0.5) ** 2) / 0.01) * 0.1  # sharp negative valley
    curve = np.column_stack([xs, ys])
    result = classify_face(curve)
    assert result.face_type == "hembra"


def test_classify_ambiguous_curve() -> None:
    """A curve with both large positive and large negative should be 'desconocida'."""
    xs = np.linspace(0, 1, POINTS_PER_FACE)
    ys = np.sin(xs * 2 * np.pi) * 0.1  # symmetric oscillation
    curve = np.column_stack([xs, ys])
    result = classify_face(curve)
    assert result.face_type == "desconocida"


# ============================================================
# T14.6b — Piece kind derivation
# ============================================================


def test_derive_piece_kind_interior() -> None:
    """Zero flat faces => interior."""
    kinds = {"1-2": "macho", "2-3": "hembra", "3-4": "macho", "4-1": "hembra"}
    assert derive_piece_kind(kinds) == "interior"


def test_derive_piece_kind_border() -> None:
    """One flat face => border."""
    kinds = {"1-2": "lisa", "2-3": "macho", "3-4": "macho", "4-1": "hembra"}
    assert derive_piece_kind(kinds) == "border"


def test_derive_piece_kind_corner() -> None:
    """Two consecutive flat faces => corner."""
    kinds = {"1-2": "lisa", "2-3": "lisa", "3-4": "macho", "4-1": "macho"}
    assert derive_piece_kind(kinds) == "corner"


def test_derive_piece_kind_unknown() -> None:
    """Non-consecutive flat faces => unknown."""
    kinds = {"1-2": "lisa", "2-3": "macho", "3-4": "lisa", "4-1": "macho"}
    assert derive_piece_kind(kinds) == "unknown"


# ============================================================
# Contour extraction
# ============================================================


def test_extract_outer_contour_square() -> None:
    """A square mask should produce a contour with many points."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 255
    contour = extract_outer_contour(mask)
    assert contour is not None
    assert contour.shape[1] == 2
    assert len(contour) >= 4


def test_extract_outer_contour_empty() -> None:
    """An empty mask should return None."""
    mask = np.zeros((50, 50), dtype=np.uint8)
    contour = extract_outer_contour(mask)
    assert contour is None


def test_contour_to_continuous_curve_resamples_evenly() -> None:
    """A closed contour should be resampled to the requested point count."""
    contour = np.array(
        [[0, 0], [10, 0], [10, 10], [0, 10]],
        dtype=np.float64,
    )
    result = contour_to_continuous_curve(contour, n_points=8)
    assert result.shape == (8, 2)
    assert np.allclose(result[0], [0, 0])
    assert np.allclose(result[2], [10, 0])
    assert np.allclose(result[4], [10, 10])
    assert np.allclose(result[6], [0, 10])


def test_smooth_puzzle_contour_preserves_shape_and_count() -> None:
    """Puzzle contour smoothing should keep a closed curve-sized point set."""
    contour = np.array(
        [[0, 0], [10, 0], [10, 2], [12, 2], [10, 10], [0, 10]],
        dtype=np.float64,
    )
    result = smooth_puzzle_contour(contour, iterations=2, n_points=24)
    assert result.shape == (24, 2)
    assert np.all(np.isfinite(result))
    assert result[:, 0].min() >= -1
    assert result[:, 0].max() <= 13


# ============================================================
# Mask validation
# ============================================================


def test_validate_mask_valid() -> None:
    """A clean solid mask should be valid."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:90, 10:90] = 255
    quality = validate_mask(mask)
    assert quality.status == "valid"


def test_validate_mask_empty() -> None:
    """An empty mask should be invalid."""
    mask = np.zeros((50, 50), dtype=np.uint8)
    quality = validate_mask(mask)
    assert quality.status == "invalid"


def test_validate_mask_touches_border() -> None:
    """A mask touching the border should be flagged."""
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[:, :] = 255  # fills entire image = touches all borders
    quality = validate_mask(mask)
    assert "touches_crop_border" in quality.flags


# ============================================================
# Fourier descriptor
# ============================================================


def test_fourier_descriptor_flat() -> None:
    """A flat parametric curve should produce complex coefficients."""
    curve = np.column_stack(
        [np.linspace(0, 1, POINTS_PER_FACE), np.zeros(POINTS_PER_FACE)]
    )
    desc = fourier_descriptor(curve, n_coefficients=8)
    assert len(desc) == 30  # (DC + 7 positive/negative frequency pairs) * 2
    assert abs(desc[0] - 0.5) < 0.01


def test_fourier_reconstruction() -> None:
    """Reconstructed parametric curve should have low 2D RMSE."""
    xs = np.linspace(0, 1, POINTS_PER_FACE)
    ys = np.sin(xs * np.pi) * 0.05
    curve = np.column_stack([xs, ys])

    desc = fourier_descriptor(curve, n_coefficients=24)
    reconstructed = reconstruct_curve_from_fourier(desc)
    rmse = float(np.sqrt(np.mean(np.sum((curve - reconstructed) ** 2, axis=1))))
    assert rmse < 0.05


def test_control_point_reconstruction_open_curve() -> None:
    """Control-point descriptor should preserve open curve endpoints."""
    xs = np.linspace(0, 1, POINTS_PER_FACE)
    ys = np.sin(xs * np.pi) * 0.1
    curve = np.column_stack([xs, ys])

    desc = control_point_descriptor(curve, n_control_points=12)
    reconstructed = reconstruct_curve_from_control_points(desc)
    rmse = float(np.sqrt(np.mean(np.sum((curve - reconstructed) ** 2, axis=1))))

    assert len(desc) == 24
    assert np.allclose(reconstructed[0], [0.0, 0.0], atol=1e-6)
    assert np.allclose(reconstructed[-1], [1.0, 0.0], atol=1e-6)
    assert rmse < 0.01


def test_control_point_descriptor_default_length() -> None:
    """Default control-point descriptor should use 36 (x, y) points."""
    curve = np.column_stack(
        [np.linspace(0, 1, POINTS_PER_FACE), np.zeros(POINTS_PER_FACE)]
    )
    desc = control_point_descriptor(curve)
    assert len(desc) == 72


# ============================================================
# Mask fallback
# ============================================================


def test_mask_from_contour() -> None:
    """mask_from_contour should produce a binary mask of the given shape."""
    contour = [[10, 10], [30, 10], [30, 30], [10, 30]]
    bbox = [5.0, 5.0, 35.0, 35.0]
    mask = mask_from_contour(contour, (40, 40), bbox)
    assert mask.shape == (40, 40)
    assert mask.dtype == np.uint8
    assert mask.sum() > 0


# ============================================================
# Face splitting
# ============================================================


def test_split_contour_into_faces() -> None:
    """Faces should cover the full contour and return 4 segments."""
    contour = np.array(
        [[0, 0], [1, 0], [2, 0], [2, 1], [2, 2], [1, 2], [0, 2], [0, 1]],
        dtype=np.float64,
    )
    corner_indices = [0, 3, 5, 7]  # TL, TR, BR, BL
    faces = split_contour_into_faces(contour, corner_indices)

    assert set(faces.keys()) == {"1-2", "2-3", "3-4", "4-1"}
    total = sum(len(pts) for pts in faces.values())
    # Account for shared corners
    assert total >= len(contour)


# ============================================================
# Corner detection basics
# ============================================================


def test_detect_corner_candidates_square() -> None:
    """A square contour should have 4 high-curvature corners."""
    n = 100
    t = np.linspace(0, 1, n)
    square = np.zeros((n, 2))
    square[:, 0] = np.interp(
        t,
        [0, 0.25, 0.5, 0.75, 1.0],
        [0, 10, 10, 0, 0],
    )
    square[:, 1] = np.interp(
        t,
        [0, 0.25, 0.5, 0.75, 1.0],
        [0, 0, 10, 10, 0],
    )

    candidates = detect_corner_candidates(square)
    assert len(candidates) >= 4


# ============================================================
# approx_poly_fallback alias
# ============================================================


def test_approx_poly_fallback_rectangle() -> None:
    """approxPolyDP fallback should find 4 corners for a clean rectangle."""
    contour = np.array(
        [
            [0, 0],
            [1, 0],
            [2, 0],
            [3, 0],
            [3, 1],
            [3, 2],
            [3, 3],
            [2, 3],
            [1, 3],
            [0, 3],
            [0, 2],
            [0, 1],
        ],
        dtype=np.float64,
    )
    result = _approx_poly_fallback(contour)
    assert result is not None
    assert len(result) == 4


# ============================================================
# _select_four_corners
# ============================================================


def test_select_four_corners_returns_none_if_few_candidates() -> None:
    """If fewer than 4 candidates, should return None."""
    candidates = [
        CornerCandidate(idx=0, x=0, y=0, curvature=1.0, convex=True),
        CornerCandidate(idx=1, x=1, y=0, curvature=0.5, convex=True),
    ]
    contour = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    result = _select_four_corners(candidates, contour)
    assert result is None


# ============================================================
# _corner_indices_on_contour
# ============================================================


def test_corner_indices_on_contour() -> None:
    """Should find the closest contour index for each corner."""
    contour = np.array(
        [[0, 0], [1, 0], [2, 0], [2, 1], [2, 2], [1, 2], [0, 2], [0, 1]],
        dtype=np.float64,
    )
    corners = np.array(
        [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]], dtype=np.float64
    )
    indices = _corner_indices_on_contour(contour, corners)
    assert len(indices) == 4
    # Each index must correspond to the right corner
    for idx, corner in zip(indices, corners, strict=True):
        assert np.allclose(contour[idx], corner)


# ============================================================
# _signed_area
# ============================================================


def test_signed_area_clockwise_negative() -> None:
    """Clockwise polygon (image coords, Y-down) should have negative signed area."""
    pts = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64)
    area = _signed_area(pts)
    assert area < 0


def test_signed_area_counter_clockwise_positive() -> None:
    """CCW polygon (image coords Y-down) should have positive signed area."""
    pts = np.array([[0, 0], [0, 10], [10, 10], [10, 0]], dtype=np.float64)
    area = _signed_area(pts)
    assert area > 0


# ============================================================
# profile_to_db_dict
# ============================================================


def test_profile_to_db_dict_structure() -> None:
    """profile_to_db_dict should return a dict with all required keys."""
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[5:45, 5:45] = 255

    face = FaceProfile(
        face_key="1-2",
        face_type="macho",
        points=np.zeros((POINTS_PER_FACE, 2)),
        reduced_descriptor=[0.1, 0.2],
        metrics={"max_positive": 0.1, "classification_confidence": 0.9},
    )

    profile = PieceProfile(
        pieza_id=1,
        pipeline="sam3",
        nombre_puzzle="test",
        origen="o1",
        piece_index=0,
        corner_points=np.zeros((4, 2)),
        corner_points_crop=np.zeros((4, 2)),
        corner_points_source=np.zeros((4, 2)),
        corner_points_normalized=np.zeros((4, 2)),
        face_order="1-2-3-4-1",
        faces={"1-2": face},
        piece_kind="interior",
        profile_status="valid",
        quality_flags=[],
    )

    d = profile_to_db_dict(profile)
    required_keys = [
        "pieza_id",
        "pipeline",
        "nombre_puzzle",
        "origen",
        "piece_index",
        "corner_points_json",
        "corner_points_crop_json",
        "corner_points_source_json",
        "corner_points_normalized_json",
        "face_order",
        "faces_json",
        "piece_kind",
        "profile_status",
        "quality_flags_json",
        "profile_version",
    ]
    for key in required_keys:
        assert key in d, f"Missing key: {key}"


# ============================================================
# smooth_contour
# ============================================================


def test_smooth_contour_no_change_large_window() -> None:
    """Smoothing with a window larger than contour should return contour unchanged."""
    contour = np.random.rand(5, 2).astype(np.float64)
    result = _smooth_contour(contour, window=10)
    assert np.allclose(result, contour)


# ============================================================
# detect_corners failure case
# ============================================================


def test_detect_corners_few_points() -> None:
    """A tiny contour should return failed method."""
    contour = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    corners, method = detect_corners(contour)
    assert len(corners) == 0
    assert method == "failed"
