"""Unit tests for src/piece_search.py."""

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from src.piece_search import (
    CONTROL_POINTS_PER_FACE,
    DIRECTION_TO_CENTER_FACE,
    OPPOSITE_DIRECTION,
    CandidateScore,
    SearchFace,
    SearchPiece,
    candidate_face_for_direction,
    classify_decision,
    compatible_face_types,
    compose_3x3_layout,
    constrained_dtw_distance,
    face_for_direction,
    filter_candidates_by_face_type,
    geometry_rmse,
    invert_face_curve,
    rank_all_cardinal_directions,
    rank_candidates_for_direction,
    score_face_pair,
    score_margin,
    trim_curve_extremes,
    trimmed_geometry_rmse,
)

# ============================================================
# Helpers
# ============================================================


def _make_face(face_key: str, face_type: str, seed: int = 0) -> SearchFace:
    """Create a SearchFace with a simple curve for testing."""
    rng = np.random.RandomState(seed)
    desc = np.column_stack(
        [
            np.linspace(0.0, 1.0, CONTROL_POINTS_PER_FACE),
            rng.uniform(-0.2, 0.2, CONTROL_POINTS_PER_FACE),
        ]
    )
    desc[0] = [0.0, 0.0]
    desc[-1] = [1.0, 0.0]
    return SearchFace(
        face_key=face_key,
        face_type=face_type,
        points=np.zeros((128, 2)),
        reduced_descriptor=desc,
        metrics={
            "reconstruction_rmse": 0.01,
            "classification_confidence": 0.9,
            "max_positive": 0.2,
            "max_negative": -0.1,
        },
    )


def _make_piece(
    pieza_id: int,
    piece_index: int = 0,
    profile_status: str = "valid",
    estado: str = "AVAILABLE",
    face_types: dict[str, str] | None = None,
) -> SearchPiece:
    """Create a SearchPiece for testing."""
    if face_types is None:
        face_types = {"1-2": "macho", "2-3": "hembra", "3-4": "hembra", "4-1": "macho"}
    faces = {
        k: _make_face(k, v, seed=pieza_id * 4 + i)
        for i, (k, v) in enumerate(face_types.items())
    }
    return SearchPiece(
        pieza_id=pieza_id,
        nombre_puzzle="test",
        origen="test",
        piece_index=piece_index,
        ruta_imagen=str(Path("data/test") / f"piece_{pieza_id}.png"),
        estado=estado,
        profile_status=profile_status,
        quality_flags=[],
        faces=faces,
        corner_points_source=np.array(
            [[0, 0], [100, 0], [100, 100], [0, 100]],
            dtype=np.float64,
        ),
        corner_points_crop=np.array(
            [[0, 0], [100, 0], [100, 100], [0, 100]],
            dtype=np.float64,
        ),
        piece_kind="interior",
        profile_version="piece_profile_v2",
    )


# ============================================================
# Phase 1: Data model
# ============================================================


def test_search_piece_faces_are_parsed() -> None:
    """A SearchPiece must have accessible SearchFace objects per face key."""
    piece = _make_piece(1)
    assert len(piece.faces) == 4
    for key in ("1-2", "2-3", "3-4", "4-1"):
        assert key in piece.faces
        face = piece.faces[key]
        assert isinstance(face, SearchFace)
        assert face.reduced_descriptor.shape == (CONTROL_POINTS_PER_FACE, 2)
        assert face.face_type in ("macho", "hembra", "lisa", "desconocida")


def test_load_profiled_pieces_empty_db(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no pieza_perfiles records, load_profiled_pieces returns empty list."""
    db_file = tmp_path / "empty.db"
    monkeypatch.setattr("src.piece_search.DB_PATH", db_file)

    from src.piece_search import load_profiled_pieces

    db_file.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_file)
    conn.execute("""
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
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
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
        estado TEXT DEFAULT 'AVAILABLE'
    )
    """)
    conn.commit()
    conn.close()

    pieces = load_profiled_pieces()
    assert pieces == []


def test_load_profiled_pieces_filters_profile_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """load_profiled_pieces must filter by profile_version."""
    db_file = tmp_path / "filter.db"
    monkeypatch.setattr("src.piece_search.DB_PATH", db_file)
    monkeypatch.setattr("src.piece_search.PROFILE_VERSION", "test_v2")

    from src.piece_search import load_profiled_pieces

    db_file.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_file)
    conn.execute("""
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
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
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
        estado TEXT DEFAULT 'AVAILABLE'
    )
    """)
    faces_dummy = json.dumps(
        {
            "1-2": {
                "tipo": "macho",
                "points": [],
                "reduced_descriptor": [],
                "metrics": {},
            },
        }
    )
    corners = json.dumps([[0, 0], [1, 0], [1, 1], [0, 1]])

    row_data = (
        1,
        "test",
        "o1",
        0,
        "dummy.png",
        "[]",
        "[0,0,1,1]",
        0.9,
        100,
        "{}",
        "piece",
        "AVAILABLE",
    )
    conn.execute(
        """
    INSERT INTO sam3_piezas (id, nombre_puzzle, origen, piece_index, ruta_imagen,
        contour_json, bounding_box_json, confidence_score, area_pixels,
        posicion_original, text_prompt, estado)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        row_data,
    )

    conn.execute(
        """
    INSERT INTO pieza_perfiles (pieza_id, pipeline, nombre_puzzle, origen, piece_index,
        corner_points_json, corner_points_crop_json, corner_points_source_json,
        corner_points_normalized_json, face_order, faces_json, piece_kind,
        profile_status, quality_flags_json, profile_version)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            1,
            "sam3",
            "test",
            "o1",
            0,
            corners,
            corners,
            corners,
            corners,
            "1-2-3-4-1",
            faces_dummy,
            "interior",
            "valid",
            "[]",
            "test_v1",
        ),
    )

    conn.execute(
        """
    INSERT INTO pieza_perfiles (pieza_id, pipeline, nombre_puzzle, origen, piece_index,
        corner_points_json, corner_points_crop_json, corner_points_source_json,
        corner_points_normalized_json, face_order, faces_json, piece_kind,
        profile_status, quality_flags_json, profile_version)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            1,
            "sam3",
            "test",
            "o1",
            0,
            corners,
            corners,
            corners,
            corners,
            "1-2-3-4-1",
            faces_dummy,
            "interior",
            "valid",
            "[]",
            "test_v2",
        ),
    )

    conn.commit()
    conn.close()

    pieces = load_profiled_pieces(profile_version="test_v2")
    assert len(pieces) == 1


def test_load_profiled_pieces_excludes_invalid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid profiles must be excluded."""
    db_file = tmp_path / "exclude.db"
    monkeypatch.setattr("src.piece_search.DB_PATH", db_file)

    from src.piece_search import load_profiled_pieces

    db_file.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_file)
    conn.execute("""
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
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
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
        estado TEXT DEFAULT 'AVAILABLE'
    )
    """)
    faces_dummy = json.dumps(
        {
            "1-2": {
                "tipo": "macho",
                "points": [],
                "reduced_descriptor": [],
                "metrics": {},
            },
        }
    )
    corners = json.dumps([[0, 0], [1, 0], [1, 1], [0, 1]])

    for row_id in (1, 2):
        conn.execute(
            """
        INSERT INTO sam3_piezas (id, nombre_puzzle, origen, piece_index, ruta_imagen,
            contour_json, bounding_box_json, confidence_score, area_pixels,
            posicion_original, text_prompt, estado)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                row_id,
                "test",
                "o1",
                row_id,
                f"dummy{row_id}.png",
                "[]",
                "[0,0,1,1]",
                0.9,
                100,
                "{}",
                "piece",
                "AVAILABLE",
            ),
        )

    conn.execute(
        """
    INSERT INTO pieza_perfiles (pieza_id, pipeline, nombre_puzzle, origen, piece_index,
        corner_points_json, corner_points_crop_json, corner_points_source_json,
        corner_points_normalized_json, face_order, faces_json, piece_kind,
        profile_status, quality_flags_json, profile_version)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            1,
            "sam3",
            "test",
            "o1",
            1,
            corners,
            corners,
            corners,
            corners,
            "1-2-3-4-1",
            faces_dummy,
            "interior",
            "valid",
            "[]",
            "piece_profile_v2",
        ),
    )

    conn.execute(
        """
    INSERT INTO pieza_perfiles (pieza_id, pipeline, nombre_puzzle, origen, piece_index,
        corner_points_json, corner_points_crop_json, corner_points_source_json,
        corner_points_normalized_json, face_order, faces_json, piece_kind,
        profile_status, quality_flags_json, profile_version)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            2,
            "sam3",
            "test",
            "o1",
            2,
            corners,
            corners,
            corners,
            corners,
            "1-2-3-4-1",
            faces_dummy,
            "interior",
            "invalid",
            "[]",
            "piece_profile_v2",
        ),
    )

    conn.commit()
    conn.close()

    pieces = load_profiled_pieces()
    assert len(pieces) == 1
    assert pieces[0].pieza_id == 1


# ============================================================
# Phase 2: Direction mapping
# ============================================================


def test_direction_face_mapping_complete() -> None:
    """All four cardinal directions must map to a face key."""
    assert set(DIRECTION_TO_CENTER_FACE.keys()) == {"N", "E", "S", "W"}
    assert set(DIRECTION_TO_CENTER_FACE.values()) == {"1-2", "2-3", "3-4", "4-1"}


def test_opposite_direction_mapping() -> None:
    """Opposite directions must be symmetrical."""
    for d, opp in OPPOSITE_DIRECTION.items():
        assert OPPOSITE_DIRECTION[opp] == d


def test_face_for_direction_returns_expected_face() -> None:
    """face_for_direction must return the correct face."""
    piece = _make_piece(10)
    assert face_for_direction(piece, "N") is piece.faces["1-2"]
    assert face_for_direction(piece, "E") is piece.faces["2-3"]
    assert face_for_direction(piece, "S") is piece.faces["3-4"]
    assert face_for_direction(piece, "W") is piece.faces["4-1"]


def test_candidate_face_for_direction_uses_opposite() -> None:
    """A candidate's matching face is in the opposite direction."""
    piece = _make_piece(11)
    assert candidate_face_for_direction(piece, "N") is piece.faces["3-4"]
    assert candidate_face_for_direction(piece, "S") is piece.faces["1-2"]


# ============================================================
# Phase 3: Topological filter
# ============================================================


def test_macho_matches_hembra() -> None:
    assert compatible_face_types("macho", "hembra") is True


def test_lisa_matches_lisa() -> None:
    assert compatible_face_types("lisa", "lisa") is True


def test_macho_does_not_match_macho() -> None:
    assert compatible_face_types("macho", "macho") is False


def test_unknown_allowed_with_flag() -> None:
    ua = "desconocida"
    assert compatible_face_types(ua, ua, allow_unknown=True) is True
    assert compatible_face_types("macho", ua, allow_unknown=True) is True


def test_unknown_rejected_without_flag() -> None:
    ua = "desconocida"
    assert compatible_face_types(ua, ua, allow_unknown=False) is False


def test_filter_candidates_by_face_type() -> None:
    """Only candidates with compatible face type should pass."""
    center = _make_piece(
        100,
        face_types={
            "1-2": "macho",
            "2-3": "lisa",
            "3-4": "hembra",
            "4-1": "macho",
        },
    )
    candidates = [
        _make_piece(
            200,
            face_types={
                "1-2": "hembra",
                "2-3": "macho",
                "3-4": "macho",
                "4-1": "hembra",
            },
        ),
        _make_piece(
            201,
            face_types={
                "1-2": "macho",
                "2-3": "macho",
                "3-4": "hembra",
                "4-1": "macho",
            },
        ),
    ]
    # candidate face is at opposite direction (S) → "3-4"
    # 200 has "3-4": "macho" (no), 201 has "3-4": "hembra" (yes)
    result = filter_candidates_by_face_type(center, candidates, "N")
    assert len(result) == 1
    assert result[0].pieza_id == 201


# ============================================================
# Phase 4: Curve inversion and RMSE
# ============================================================


def test_invert_face_curve_preserves_shape_length() -> None:
    """Inverted curve must have same number of points."""
    points = np.column_stack([np.linspace(0, 1, 36), np.zeros(36)])
    inverted = invert_face_curve(points)
    assert inverted.shape == points.shape


def test_invert_face_curve_flips_y() -> None:
    """Y values must be inverted."""
    points = np.column_stack([np.linspace(0, 1, 36), np.linspace(0, 0.2, 36)])
    points[-1] = [1.0, 0.0]
    inverted = invert_face_curve(points)
    assert inverted[0, 1] == pytest.approx(0.0)
    assert inverted[-1, 1] == pytest.approx(0.0)
    assert inverted[0, 0] == pytest.approx(0.0)
    assert inverted[-1, 0] == pytest.approx(1.0)
    # After reversal, inverted[5] corresponds to points[30] (35-5)
    orig_idx = len(points) - 1 - 5
    assert inverted[5, 1] == pytest.approx(-points[orig_idx, 1], abs=1e-10)


def test_geometry_rmse_zero_for_identical_curves() -> None:
    x = np.linspace(0, 1, 36)
    a = np.column_stack([x, np.sin(x * np.pi) * 0.1])
    a[0] = [0, 0]
    a[-1] = [1, 0]
    b = a.copy()
    assert geometry_rmse(a, b) == pytest.approx(0.0)


def test_geometry_rmse_increases_for_shifted_curve() -> None:
    a = np.column_stack([np.linspace(0, 1, 36), np.zeros(36)])
    b = np.column_stack([np.linspace(0, 1, 36), np.ones(36) * 0.1])
    b[0] = [0, 0]
    b[-1] = [1, 0]
    assert geometry_rmse(a, b) > 0


def test_trim_curve_extremes_removes_start_and_end() -> None:
    points = np.column_stack([np.linspace(0, 1, 10), np.zeros(10)])

    trimmed = trim_curve_extremes(points, trim_samples=2)

    assert trimmed.shape == (6, 2)
    assert trimmed[0, 0] == pytest.approx(points[2, 0])
    assert trimmed[-1, 0] == pytest.approx(points[-3, 0])


def test_trimmed_geometry_rmse_ignores_endpoint_noise() -> None:
    x = np.linspace(0, 1, CONTROL_POINTS_PER_FACE)
    a = np.column_stack([x, np.sin(x * np.pi * 2.0) * 0.1])
    b = a.copy()
    b[:4, 1] = 1.0
    b[-4:, 1] = -1.0

    plain = geometry_rmse(a, b)
    trimmed = trimmed_geometry_rmse(a, b, trim_samples=4)

    assert trimmed < plain
    assert trimmed == pytest.approx(0.0)


def test_score_face_pair_contains_metadata() -> None:
    """CandidateScore must include face types and geometric RMSE."""
    center_face = _make_face("1-2", "macho", seed=0)
    cand_face = _make_face("3-4", "hembra", seed=1)
    score = score_face_pair(center_face, cand_face)
    assert score.center_face_key == "1-2"
    assert score.candidate_face_key == "3-4"
    assert score.center_face_type == "macho"
    assert score.candidate_face_type == "hembra"
    assert score.geometry_rmse >= 0


# ============================================================
# Phase 5: DTW
# ============================================================


def test_constrained_dtw_identical_zero() -> None:
    a = np.column_stack([np.linspace(0, 1, 36), np.zeros(36)])
    b = a.copy()
    d = constrained_dtw_distance(a, b)
    assert d == pytest.approx(0.0)


def test_constrained_dtw_window_limits_alignment() -> None:
    """DTW with narrow window must compute a finite distance."""
    x = np.linspace(0, 1, 36)
    a = np.column_stack([x, np.sin(x * np.pi) * 0.1])
    b = np.column_stack([x, np.cos(x * np.pi) * 0.1])
    d = constrained_dtw_distance(a, b, window_ratio=0.1)
    assert d > 0
    assert np.isfinite(d)


def test_dtw_not_used_in_default_score() -> None:
    """score_face_pair must not use DTW by default."""
    cf = _make_face("1-2", "macho", seed=0)
    cdf = _make_face("3-4", "hembra", seed=1)
    score = score_face_pair(cf, cdf)
    assert score.geometry_dtw is None


def test_dtw_can_be_reported_as_diagnostic() -> None:
    """When compute_dtw=True, DTW must be present in ranking."""
    center = _make_piece(100)
    candidates = [_make_piece(200)]
    scores = rank_candidates_for_direction(
        center,
        candidates,
        "N",
        top_k=3,
        compute_dtw=True,
    )
    if scores:
        assert scores[0].geometry_dtw is not None


# ============================================================
# Phase 6: Ranking
# ============================================================


def test_rank_excludes_center_piece() -> None:
    """The center piece must not appear in its own ranking."""
    center = _make_piece(100)
    candidates = [
        _make_piece(100),  # same as center
        _make_piece(200),
        _make_piece(201),
    ]
    scores = rank_candidates_for_direction(center, candidates, "N")
    ids = {s.candidate_piece_id for s in scores}
    assert 100 not in ids


def test_rank_orders_by_score_total() -> None:
    """Ranked candidates must be sorted by ascending score_total."""
    center = _make_piece(
        100,
        face_types={
            "1-2": "macho",
            "2-3": "lisa",
            "3-4": "lisa",
            "4-1": "lisa",
        },
    )
    candidates = [_make_piece(200 + i) for i in range(3)]
    scores = rank_candidates_for_direction(center, candidates, "N")
    for i in range(len(scores) - 1):
        assert scores[i].score_total <= scores[i + 1].score_total


def test_visual_signals_do_not_affect_ranking_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Visual diagnostics must not change the shape-first score_total."""
    center = _make_piece(100)
    cand = _make_piece(200)

    monkeypatch.setattr(
        "src.piece_search.compute_visual_scores",
        lambda *_args, **_kwargs: (999.0, 999.0, 999.0),
    )

    from src.piece_search import rank_candidates_with_visual_signals

    scores = rank_candidates_with_visual_signals(center, [cand], "N")

    assert len(scores) == 1
    assert scores[0].score_total == pytest.approx(scores[0].geometry_rmse)
    assert scores[0].luminance_distance == pytest.approx(999.0)
    assert scores[0].color_distance == pytest.approx(999.0)
    assert scores[0].texture_distance == pytest.approx(999.0)


def test_dtw_can_be_used_for_ranking_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When requested, score_total must use DTW instead of RMSE."""
    center = _make_piece(100)
    cand = _make_piece(200)

    monkeypatch.setattr(
        "src.piece_search.compute_visual_scores",
        lambda *_args, **_kwargs: (0.0, 0.0, 0.0),
    )
    monkeypatch.setattr(
        "src.piece_search.constrained_dtw_distance",
        lambda *_args, **_kwargs: 42.0,
    )

    from src.piece_search import rank_candidates_with_visual_signals

    scores = rank_candidates_with_visual_signals(
        center,
        [cand],
        "N",
        use_dtw_for_ranking=True,
    )

    assert len(scores) == 1
    assert scores[0].geometry_dtw == pytest.approx(42.0)
    assert scores[0].score_total == pytest.approx(42.0)


def test_rank_applies_needs_review_penalty() -> None:
    """needs_review candidates must have higher score than equivalent valid ones."""
    center = _make_piece(100)

    valid_cand = _make_piece(200, profile_status="valid")
    review_cand = _make_piece(201, profile_status="needs_review")

    valid_scores = rank_candidates_for_direction(center, [valid_cand], "N")
    review_scores = rank_candidates_for_direction(center, [review_cand], "N")

    if valid_scores and review_scores:
        assert review_scores[0].score_total > valid_scores[0].score_total


def test_rank_returns_top_k() -> None:
    """Rank must return at most top_k results."""
    center = _make_piece(100)
    candidates = [_make_piece(200 + i) for i in range(10)]
    scores = rank_candidates_for_direction(center, candidates, "N", top_k=3)
    assert len(scores) <= 3


def test_rank_all_cardinal_directions_has_four_keys() -> None:
    """rank_all_cardinal_directions must return exactly 4 directions."""
    center = _make_piece(100)
    candidates = [_make_piece(200 + i) for i in range(5)]
    rankings = rank_all_cardinal_directions(center, candidates)
    assert set(rankings.keys()) == {"N", "E", "S", "W"}


# ============================================================
# Phase 7: Visual signals (profile shapes)
# ============================================================


def test_luminance_profile_shape() -> None:
    """Luminance profile must have the requested number of points."""
    band = np.random.uniform(0, 255, (20, 8, 3)).astype(np.float32)
    from src.piece_search import luminance_profile

    prof = luminance_profile(band, n_points=36)
    assert prof.shape == (36,)


def test_color_profile_lab_shape() -> None:
    """Lab colour profile must have (n_points, 3) shape."""
    band = np.random.uniform(0, 255, (20, 8, 3)).astype(np.float32)
    from src.piece_search import color_profile_lab

    prof = color_profile_lab(band, n_points=36)
    assert prof.shape == (36, 3)


def test_texture_profile_gradient_shape() -> None:
    """Texture profile must have the requested number of points."""
    band = np.random.uniform(0, 255, (20, 8, 3)).astype(np.float32)
    from src.piece_search import texture_profile_gradient

    prof = texture_profile_gradient(band, n_points=36)
    assert prof.shape == (36,)


# ============================================================
# Phase 9: 3x3 layout
# ============================================================


def test_layout_3x3_contains_center() -> None:
    """The 3x3 layout must include a cell for the center."""
    center = _make_piece(100)
    center.ruta_imagen = str(Path("nonexistent.png"))
    canvas = compose_3x3_layout(center, {}, {})
    assert canvas.shape == (940, 940, 3)  # 3*300 + 4*10 = 940


def test_layout_3x3_allows_missing_candidates() -> None:
    """The 3x3 layout must not crash when candidates are missing."""
    center = _make_piece(100)
    center.ruta_imagen = str(Path("nonexistent.png"))
    ranked: dict[str, list[CandidateScore]] = {"N": [], "E": [], "S": [], "W": []}
    canvas = compose_3x3_layout(center, ranked, {})
    assert canvas.shape == (940, 940, 3)


# ============================================================
# Phase 10: Quantitative validation
# ============================================================


def test_score_margin_computation() -> None:
    """score_margin must return the difference between top two scores."""
    scores = [
        CandidateScore(
            candidate_piece_id=1,
            candidate_piece_index=0,
            center_face_key="1-2",
            candidate_face_key="3-4",
            center_face_type="macho",
            candidate_face_type="hembra",
            geometry_rmse=0.1,
            score_total=1.0,
        ),
        CandidateScore(
            candidate_piece_id=2,
            candidate_piece_index=1,
            center_face_key="1-2",
            candidate_face_key="3-4",
            center_face_type="macho",
            candidate_face_type="hembra",
            geometry_rmse=0.2,
            score_total=2.0,
        ),
    ]
    assert score_margin(scores) == pytest.approx(1.0)


def test_ambiguous_when_margin_low() -> None:
    """A low margin between top candidates must produce 'ambiguous' decision."""
    scores = [
        CandidateScore(
            candidate_piece_id=1,
            candidate_piece_index=0,
            center_face_key="1-2",
            candidate_face_key="3-4",
            center_face_type="macho",
            candidate_face_type="hembra",
            geometry_rmse=0.1,
            score_total=1.01,
        ),
        CandidateScore(
            candidate_piece_id=2,
            candidate_piece_index=1,
            center_face_key="1-2",
            candidate_face_key="3-4",
            center_face_type="macho",
            candidate_face_type="hembra",
            geometry_rmse=0.11,
            score_total=1.02,
        ),
    ]
    decision = classify_decision(scores, margin_threshold=0.1)
    assert decision == "ambiguous"


def test_no_candidate_status() -> None:
    """Empty list must produce 'no_candidate' decision."""
    assert classify_decision([]) == "no_candidate"
