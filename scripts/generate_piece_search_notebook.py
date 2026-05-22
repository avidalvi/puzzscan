"""Generate the piece search exploration notebook.

Output: notebooks/piece_search_exploration.ipynb
"""

# mypy: disable-error-code=no-untyped-call

import nbformat as nbf

nb = nbf.v4.new_notebook()

cells: list[nbf.NotebookNode] = []

# Title
cells.append(
    nbf.v4.new_markdown_cell(
        """# Piece Search Exploration

This notebook validates the piece search engine: given a center piece with four
profiled faces, it finds the top-3 compatible candidates for each cardinal direction.

## Controls

- `PUZZLE`: puzzle name (e.g. "ciudad")
- `ORIGEN`: source image stem (e.g. "piezas_1") — None for all
- `CENTER_PIEZA_ID`: database ID of the center piece (from `sam3_piezas`)
- `PROFILE_VERSION`: profile version to load
- `TOP_K`: number of top candidates per direction
- `COMPUTE_DTW`: whether to compute DTW diagnostic
- `USE_DTW_FOR_RANKING`: rank by DTW instead of RMSE
- `ALLOW_NEEDS_REVIEW`: include `needs_review` profiles
- `ALLOW_UNKNOWN_FACE`: allow `desconocida` face type in matching

Run cells sequentially.
"""
    )
)

# Setup
cells.append(
    nbf.v4.new_code_cell(
        """%matplotlib inline
import json
import os
import sqlite3
import sys
import importlib
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

plt.close("all")

if Path.cwd().name == "notebooks":
    os.chdir("..")

sys.path.insert(0, str(Path.cwd().resolve()))

import src.piece_search as piece_search

importlib.reload(piece_search)

from src.piece_search import (
    CONTROL_POINTS_PER_FACE,
    GEOMETRY_TRIM_SAMPLES,
    SearchPiece,
    candidate_face_for_direction,
    compose_3x3_layout,
    face_for_direction,
    invert_face_curve,
    load_profiled_pieces,
    load_piece_rgba,
    rank_all_cardinal_directions,
    rank_candidates_with_visual_signals,
    score_margin,
    summarize_rankings,
)

# === CONFIG ===
PUZZLE = "ciudad"
ORIGEN = None
CENTER_PIEZA_ID = 836
PROFILE_VERSION = "piece_profile_v2"
TOP_K = 3
COMPUTE_DTW = True
USE_DTW_FOR_RANKING = True
ALLOW_NEEDS_REVIEW = True
ALLOW_UNKNOWN_FACE = True

DB_PATH = "data/puzzscan_v2.db"
GEOMETRY_TRIM_X = GEOMETRY_TRIM_SAMPLES / max(1, CONTROL_POINTS_PER_FACE - 1)
print(f"Geometry comparison: trim={GEOMETRY_TRIM_SAMPLES} samples per endpoint")
"""
    )
)

# Cell 1: Load profiles
cells.append(
    nbf.v4.new_code_cell(
        """# Cell 1: Load profiled pieces

print("[1/10] Loading profiled pieces ...")
pieces = load_profiled_pieces(
    puzzle=PUZZLE,
    origen=ORIGEN,
    profile_version=PROFILE_VERSION,
    allow_needs_review=ALLOW_NEEDS_REVIEW,
)

print(f"Loaded {len(pieces)} pieces")
status_counts: dict[str, int] = {}
for p in pieces:
    status_counts[p.profile_status] = status_counts.get(p.profile_status, 0) + 1
print(f"By profile_status: {status_counts}")

if pieces:
    print()
    print("First 5 pieces:")
    for p in pieces[:5]:
        print(f"  id={p.pieza_id} idx={p.piece_index} "
              f"status={p.profile_status} kind={p.piece_kind}")

print("[1/10] Done")
"""
    )
)

# Cell 2: Select center piece
cells.append(
    nbf.v4.new_code_cell(
        """# Cell 2: Select center piece

print("[2/10] Selecting center piece ...")

def show_piece_info(piece: SearchPiece) -> None:
    print(f"Center piece: id={piece.pieza_id} index={piece.piece_index}")
    print(f"  Status: {piece.profile_status}")
    print(f"  Kind: {piece.piece_kind}")
    print(f"  Image: {piece.ruta_imagen}")
    print(f"  Faces:")
    for fk in ("1-2", "2-3", "3-4", "4-1"):
        face = piece.faces.get(fk)
        if face:
            print(f"    {fk}: type={face.face_type} "
                  f"conf={face.metrics.get('classification_confidence', 0):.2f} "
                  f"rmse={face.metrics.get('reconstruction_rmse', 0):.4f}")

center = None
for p in pieces:
    if p.pieza_id == CENTER_PIEZA_ID:
        center = p
        break

if center is None and pieces:
    for p in pieces:
        if p.profile_status in ("valid", "needs_review"):
            face_types = {
                k: p.faces[k].face_type
                for k in ("1-2", "2-3", "3-4", "4-1") if k in p.faces
            }
            unknown = sum(1 for ft in face_types.values() if ft == "desconocida")
            if unknown == 0:
                center = p
                break
    if center is None:
        center = pieces[0]
        print("WARNING: no valid/needs_review piece fully profiled, "
              "using first available")

if center is None:
    raise ValueError("No pieces loaded — run profiling first")

CENTER_PIEZA_ID = center.pieza_id
show_piece_info(center)

# Display the center piece image
rgba = load_piece_rgba(center)
if rgba is not None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(rgba)
    ax.set_title(f"Center: id={center.pieza_id} idx={center.piece_index}")
    ax.axis("off")
    plt.tight_layout()
    plt.show()
    plt.close(fig)

print("[2/10] Done")
"""
    )
)

# Cell 3: Show center faces
cells.append(
    nbf.v4.new_code_cell(
        """# Cell 3: Normalized curves of center piece faces

print("[3/10] Plotting face curves ...")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes_flat = axes.flatten()

for i, fk in enumerate(("1-2", "2-3", "3-4", "4-1")):
    ax = axes_flat[i]
    face = center.faces.get(fk)
    if face is None or len(face.points) == 0:
        ax.set_title(f"{fk}: no data")
        ax.axis("off")
        continue
    curve = face.points
    control = face.reduced_descriptor
    ax.plot(curve[:, 0], curve[:, 1], "b-", linewidth=1.5, label="Curve")
    if len(control) > 0:
        ax.plot(
            control[:, 0],
            control[:, 1],
            "ko",
            markersize=3,
            label="Control points",
        )
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.4)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.25)
    rmse = face.metrics.get("reconstruction_rmse", 0)
    ax.set_title(f"{fk} — {face.face_type} (RMSE={rmse:.4f})")
    ax.set_xlabel("X (parametric)")
    ax.set_ylabel("Y (normalised)")
    ax.legend(fontsize=8)

fig.suptitle("Center piece — normalised face curves", fontsize=14)
plt.tight_layout()
plt.show()
plt.close(fig)

print("[3/10] Done")
"""
    )
)

# Cell 4: Direction/face table
cells.append(
    nbf.v4.new_code_cell(
        """# Cell 4: Direction to face mapping

print("[4/10] Direction to face mapping ...")

from src.piece_search import DIRECTION_TO_CENTER_FACE, OPPOSITE_DIRECTION

print("Direction -> Center face -> Candidate face (opposite)")
print("-" * 55)
for direction in ("N", "E", "S", "W"):
    c_face_key = DIRECTION_TO_CENTER_FACE[direction]
    opp = OPPOSITE_DIRECTION[direction]
    cand_key = DIRECTION_TO_CENTER_FACE[opp]
    c_face = center.faces.get(c_face_key)
    c_type = c_face.face_type if c_face else "N/A"
    print(f"  {direction:5s} -> {c_face_key:5s} ({c_type:12s})  ->  {cand_key:5s}")

print()
print("NOTE: canonical corner ordering: 1=TL, 2=TR, 3=BR, 4=BL")
print("It should be validated visually for the specific puzzle convention.")

print("[4/10] Done")
"""
    )
)

# Cell 5: Rankings for N/E/S/W
cells.append(
    nbf.v4.new_code_cell(
        """# Cell 5: Rank candidates for all four cardinal directions

rank_metric = "DTW" if USE_DTW_FOR_RANKING else "RMSE"
print(f"[5/10] Ranking candidates by {rank_metric}; visual signals are diagnostics ...")

rankings = {
    direction: rank_candidates_with_visual_signals(
        center,
        pieces,
        direction,
        top_k=TOP_K,
        compute_dtw=COMPUTE_DTW,
        use_dtw_for_ranking=USE_DTW_FOR_RANKING,
        allow_unknown=ALLOW_UNKNOWN_FACE,
    )
    for direction in ("N", "E", "S", "W")
}

print("Rankings by direction")
print("=" * 60)
for direction in ("N", "E", "S", "W"):
    scores = rankings[direction]
    print()
    print("-" * 60)
    if scores:
        margin = (
            scores[1].score_total - scores[0].score_total
            if len(scores) >= 2 else None
        )
        margin_str = f"{margin:.4f}" if margin is not None else "N/A"
    else:
        margin_str = "N/A"
    print(f"Direction {direction}: {len(scores)} candidates (margin={margin_str})")
    if not scores:
        print("  No compatible candidates found")
        continue
    for rank, sc in enumerate(scores, 1):
        dtw_str = f" dtw={sc.geometry_dtw:.4f}" if sc.geometry_dtw is not None else ""
        lum_str = (
            f" lum={sc.luminance_distance:.3f}"
            if sc.luminance_distance is not None else ""
        )
        color_str = (
            f" color={sc.color_distance:.3f}"
            if sc.color_distance is not None else ""
        )
        texture_str = (
            f" texture={sc.texture_distance:.3f}"
            if sc.texture_distance is not None else ""
        )
        pen_str = f" penalties={sc.penalties}" if sc.penalties else ""
        print(f"  #{rank} piece_id={sc.candidate_piece_id} "
              f"idx={sc.candidate_piece_index} "
              f"rmse={sc.geometry_rmse:.4f}{dtw_str}"
              f"{lum_str}{color_str}{texture_str} "
              f"score({rank_metric})={sc.score_total:.4f}{pen_str}")

print("[5/10] Done")
"""
    )
)

# Cell 6: Curve overlay for top 3
cells.append(
    nbf.v4.new_code_cell(
        """# Cell 6: Curve overlay — center face vs top-3 candidates per direction

print("[6/10] Plotting curve overlays ...")

for direction in ("N", "E", "S", "W"):
    scores = rankings[direction]
    if not scores:
        print(f"  {direction}: no candidates")
        continue

    center_face = face_for_direction(center, direction)
    if center_face is None or len(center_face.reduced_descriptor) == 0:
        continue

    fig, ax = plt.subplots(figsize=(8, 5))
    c_pts = center_face.reduced_descriptor
    ax.plot(c_pts[:, 0], c_pts[:, 1], "k-", linewidth=2, label="Center", alpha=0.8)

    colors = ["tab:blue", "tab:orange", "tab:green"]
    for rank, sc in enumerate(scores[:TOP_K]):
        cand = next((p for p in pieces if p.pieza_id == sc.candidate_piece_id), None)
        if cand is None:
            continue
        cand_face = candidate_face_for_direction(cand, direction)
        if cand_face is None or len(cand_face.reduced_descriptor) == 0:
            continue
        inv = invert_face_curve(cand_face.reduced_descriptor)
        ax.plot(inv[:, 0], inv[:, 1], color=colors[rank % len(colors)],
                linewidth=1.2, linestyle="--",
                label=f"#{rank + 1} id={sc.candidate_piece_id} "
                      f"rmse={sc.geometry_rmse:.3f}")

    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.3)
    ax.axvspan(0, GEOMETRY_TRIM_X, color="gray", alpha=0.08)
    ax.axvspan(1 - GEOMETRY_TRIM_X, 1, color="gray", alpha=0.08)
    ax.set_xlim(0, 1)
    ax.set_title(f"Direction {direction} — face {center_face.face_key}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()
    plt.close(fig)

print("[6/10] Done")
"""
    )
)

# Cell 7: Visual signals
cells.append(
    nbf.v4.new_code_cell(
        """# Cell 7: Visual signals (luminance, colour, texture)

print("[7/10] Checking visual signals ...")

n_with_images = sum(1 for p in pieces if Path(p.ruta_imagen).exists())
print(f"Pieces with accessible images: {n_with_images}/{len(pieces)}")

print("[7/10] Done")
"""
    )
)

# Cell 8: Center + best candidate per direction
cells.append(
    nbf.v4.new_code_cell(
        """# Cell 8: Joined overlay - center piece with best candidate in each direction

print("[8/10] Showing joined overlays ...")

FACE_CORNER_INDICES = {
    "1-2": (0, 1),
    "2-3": (1, 2),
    "3-4": (2, 3),
    "4-1": (3, 0),
}


def face_outward_normal(piece, face_key):
    ia, ib = FACE_CORNER_INDICES[face_key]
    a = piece.corner_points_crop[ia]
    b = piece.corner_points_crop[ib]
    edge = b - a
    edge_len = np.linalg.norm(edge)
    if edge_len < 1e-9:
        return np.array([0.0, -1.0])
    unit = edge / edge_len
    normal = np.array([-unit[1], unit[0]])
    mid = (a + b) / 2
    centroid = piece.corner_points_crop.mean(axis=0)
    if np.dot(normal, centroid - mid) > 0:
        normal = -normal
    return normal


def alpha_blend_rgba(canvas, rgba, x, y):
    h, w = rgba.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(canvas.shape[1], x + w), min(canvas.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return
    sx0, sy0 = x0 - x, y0 - y
    patch = rgba[sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0)]
    alpha = patch[..., 3:4].astype(np.float32) / 255.0
    canvas[y0:y1, x0:x1] = (
        patch[..., :3].astype(np.float32) * alpha
        + canvas[y0:y1, x0:x1].astype(np.float32) * (1 - alpha)
    ).astype(np.uint8)


def compose_joined_overlay(
    center_piece,
    candidate_piece,
    direction,
    margin=160,
):
    center_rgba = load_piece_rgba(center_piece)
    cand_rgba = load_piece_rgba(candidate_piece)
    if center_rgba is None or cand_rgba is None:
        return None

    center_face = face_for_direction(center_piece, direction)
    cand_face = candidate_face_for_direction(candidate_piece, direction)
    if center_face is None or cand_face is None:
        return None

    c_ia, c_ib = FACE_CORNER_INDICES[center_face.face_key]
    k_ia, k_ib = FACE_CORNER_INDICES[cand_face.face_key]
    c_a = center_piece.corner_points_crop[c_ia]
    c_b = center_piece.corner_points_crop[c_ib]
    k_a = candidate_piece.corner_points_crop[k_ia]
    k_b = candidate_piece.corner_points_crop[k_ib]

    c_edge_len = np.linalg.norm(c_b - c_a)
    k_edge_len = np.linalg.norm(k_b - k_a)
    if c_edge_len < 1e-9 or k_edge_len < 1e-9:
        return None

    scale = c_edge_len / k_edge_len
    cand_centroid = candidate_piece.corner_points_crop.mean(axis=0)
    cand_dist = abs(np.cross(k_b - k_a, cand_centroid - k_a)) / k_edge_len
    target_centroid = (
        (c_a + c_b) / 2
        + face_outward_normal(center_piece, center_face.face_key) * cand_dist * scale
    )

    src = np.float32([k_b, k_a, cand_centroid])
    dst = np.float32([c_a, c_b, target_centroid])
    matrix = cv2.getAffineTransform(src, dst)

    ch, cw = center_rgba.shape[:2]
    kh, kw = cand_rgba.shape[:2]
    cand_corners = np.float32([[0, 0], [kw, 0], [kw, kh], [0, kh]]).reshape(-1, 1, 2)
    transformed = cv2.transform(cand_corners, matrix).reshape(-1, 2)
    all_pts = np.vstack([transformed, [[0, 0], [cw, 0], [cw, ch], [0, ch]]])
    min_xy = np.floor(all_pts.min(axis=0)).astype(int) - margin
    max_xy = np.ceil(all_pts.max(axis=0)).astype(int) + margin
    canvas_w, canvas_h = (max_xy - min_xy).astype(int)

    offset = -min_xy
    matrix_offset = matrix.copy()
    matrix_offset[:, 2] += offset

    warped = cv2.warpAffine(
        cand_rgba,
        matrix_offset,
        (canvas_w, canvas_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 245
    alpha_blend_rgba(canvas, warped, 0, 0)
    alpha_blend_rgba(canvas, center_rgba, int(offset[0]), int(offset[1]))
    return canvas


def candidate_join_transform(center_piece, candidate_piece, direction):
    center_face = face_for_direction(center_piece, direction)
    cand_face = candidate_face_for_direction(candidate_piece, direction)
    if center_face is None or cand_face is None:
        return None

    c_ia, c_ib = FACE_CORNER_INDICES[center_face.face_key]
    k_ia, k_ib = FACE_CORNER_INDICES[cand_face.face_key]
    c_a = center_piece.corner_points_crop[c_ia]
    c_b = center_piece.corner_points_crop[c_ib]
    k_a = candidate_piece.corner_points_crop[k_ia]
    k_b = candidate_piece.corner_points_crop[k_ib]

    c_edge_len = np.linalg.norm(c_b - c_a)
    k_edge_len = np.linalg.norm(k_b - k_a)
    if c_edge_len < 1e-9 or k_edge_len < 1e-9:
        return None

    scale = c_edge_len / k_edge_len
    cand_centroid = candidate_piece.corner_points_crop.mean(axis=0)
    cand_dist = abs(np.cross(k_b - k_a, cand_centroid - k_a)) / k_edge_len
    target_centroid = (
        (c_a + c_b) / 2
        + face_outward_normal(center_piece, center_face.face_key) * cand_dist * scale
    )

    src = np.float32([k_b, k_a, cand_centroid])
    dst = np.float32([c_a, c_b, target_centroid])
    return cv2.getAffineTransform(src, dst)

for direction in ("N", "E", "S", "W"):
    scores = rankings[direction]
    if not scores:
        print(f"  {direction}: no candidates")
        continue
    best = scores[0]
    cand = next((p for p in pieces if p.pieza_id == best.candidate_piece_id), None)
    if cand is None:
        continue

    joined = compose_joined_overlay(center, cand, direction)
    if joined is None:
        print(f"  {direction}: could not compose overlay")
        continue

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(joined)
    ax.set_title(
        f"{direction}: center id={center.pieza_id} + "
        f"candidate id={best.candidate_piece_id} rmse={best.geometry_rmse:.3f}"
    )
    ax.axis("off")

    plt.tight_layout()
    plt.show()
    plt.close(fig)

print("[8/10] Done")
"""
    )
)

# Cell 9: joined layout
cells.append(
    nbf.v4.new_code_cell(
        """# Cell 9: Joined 4-direction layout

print("[9/10] Building joined 4-direction layout ...")

piece_map = {p.pieza_id: p for p in pieces}

center_rgba = load_piece_rgba(center)
if center_rgba is None:
    raise ValueError(f"Could not load center image: {center.ruta_imagen}")

candidate_entries = []
all_corners = [[0, 0], [center_rgba.shape[1], 0],
               [center_rgba.shape[1], center_rgba.shape[0]],
               [0, center_rgba.shape[0]]]

for direction in ("N", "E", "S", "W"):
    scores = rankings[direction]
    if not scores:
        print(f"  {direction}: no candidates")
        continue
    best = scores[0]
    cand = piece_map.get(best.candidate_piece_id)
    if cand is None:
        continue
    cand_rgba = load_piece_rgba(cand)
    matrix = candidate_join_transform(center, cand, direction)
    if cand_rgba is None or matrix is None:
        print(f"  {direction}: could not transform candidate")
        continue

    kh, kw = cand_rgba.shape[:2]
    corners = np.float32([[0, 0], [kw, 0], [kw, kh], [0, kh]]).reshape(-1, 1, 2)
    transformed = cv2.transform(corners, matrix).reshape(-1, 2)
    all_corners.extend(transformed.tolist())
    candidate_entries.append((direction, best, cand_rgba, matrix))

all_pts = np.array(all_corners, dtype=np.float32)
margin = 220
min_xy = np.floor(all_pts.min(axis=0)).astype(int) - margin
max_xy = np.ceil(all_pts.max(axis=0)).astype(int) + margin
canvas_w, canvas_h = (max_xy - min_xy).astype(int)
offset = -min_xy

canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 245

for direction, best, cand_rgba, matrix in candidate_entries:
    matrix_offset = matrix.copy()
    matrix_offset[:, 2] += offset
    warped = cv2.warpAffine(
        cand_rgba,
        matrix_offset,
        (canvas_w, canvas_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    alpha_blend_rgba(canvas, warped, 0, 0)

alpha_blend_rgba(canvas, center_rgba, int(offset[0]), int(offset[1]))

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(canvas)
ax.set_title("Joined layout - center + best candidates", fontsize=14)
ax.axis("off")
plt.tight_layout()
plt.show()
plt.close(fig)

print("[9/10] Done")
"""
    )
)

# Cell 10: Risk/quality summary
cells.append(
    nbf.v4.new_code_cell(
        """# Cell 10: Summary of risks and quality

print("[10/10] Summary ...")

summary = summarize_rankings(rankings)

print("Decision summary")
print("=" * 60)
for direction, info in summary.items():
    decision = info["decision"]
    n = info["n_candidates"]
    margin = info.get("margin")
    margin_str = f"{margin:.4f}" if margin is not None else "N/A"
    print(f"  {direction}: {n} candidates, margin={margin_str}, decision={decision}")

print()
print("Quality notes:")
for p in pieces:
    if p.quality_flags:
        print(f"  piece_id={p.pieza_id} flags={p.quality_flags}")

valid = sum(1 for p in pieces if p.profile_status == "valid")
review = sum(1 for p in pieces if p.profile_status == "needs_review")
invalid = sum(1 for p in pieces if p.profile_status == "invalid")
print(f"\\nProfile summary: valid={valid} needs_review={review} invalid={invalid}")
print(f"Center: id={CENTER_PIEZA_ID} status={center.profile_status} "
      f"kind={center.piece_kind}")

print("[10/10] Done")
"""
    )
)

nb.cells = cells

nbf.write(nb, "notebooks/piece_search_exploration.ipynb")
print("Notebook generated: notebooks/piece_search_exploration.ipynb")
