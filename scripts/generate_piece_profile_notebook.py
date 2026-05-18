"""Generate the piece profile exploration notebook.

Output: notebooks/piece_profile_exploration.ipynb
"""

# mypy: disable-error-code=no-untyped-call

import nbformat as nbf

nb = nbf.v4.new_notebook()

cells: list[nbf.NotebookNode] = []

# Title
cells.append(
    nbf.v4.new_markdown_cell(
        """# Piece Profile Exploration

This notebook lets you visually inspect the geometric profile of a single puzzle piece.

## Controls

- `PIEZA_ID`: database ID of the piece (from `sam3_piezas`)
- `PUZZLE`: puzzle name (e.g. "ciudad")
- `ORIGEN`: source image stem (e.g. "piezas_1")
- `POINTS_PER_FACE`: resolution of normalized curves

Run cells sequentially. Each cell visualises one step of the pipeline.
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
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

plt.close("all")

if Path.cwd().name == "notebooks":
    os.chdir("..")

sys.path.insert(0, str(Path.cwd().resolve()))

from src.piece_profile import (
    POINTS_PER_FACE,
    PROFILE_VERSION,
    build_piece_profile,
    classify_face,
    control_point_descriptor,
    contour_to_continuous_curve,
    detect_corners,
    ensure_clockwise,
    extract_outer_contour,
    fourier_descriptor,
    load_alpha_mask,
    normalize_face_curve,
    order_corners_canonical,
    profile_to_db_dict,
    reconstruct_curve_from_control_points,
    reconstruct_curve_from_fourier,
    smooth_puzzle_contour,
    split_contour_into_faces,
    validate_mask,
)

# === CONFIG ===
PIEZA_ID = None
PUZZLE = "ciudad"
ORIGEN = "piezas_1"
POINTS_PER_FACE = POINTS_PER_FACE
N_REPRESENTATIVE_PIECES = 5
MAX_CONTOUR_POINTS_FOR_NOTEBOOK = 2000

DB_PATH = "data/puzzscan_v2.db"

# Load piece row
conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row

if PIEZA_ID is None:
    row = conn.execute(
        "SELECT * FROM sam3_piezas "
        "WHERE nombre_puzzle = ? AND origen = ? "
        "ORDER BY piece_index LIMIT 1",
        (PUZZLE, ORIGEN),
    ).fetchone()
else:
    row = conn.execute(
        "SELECT * FROM sam3_piezas "
        "WHERE id = ? AND nombre_puzzle = ? AND origen = ?",
        (PIEZA_ID, PUZZLE, ORIGEN),
    ).fetchone()

available = conn.execute(
    "SELECT id, piece_index, area_pixels, confidence_score, ruta_imagen "
    "FROM sam3_piezas "
    "WHERE nombre_puzzle = ? AND origen = ? "
    "ORDER BY piece_index",
    (PUZZLE, ORIGEN),
).fetchall()
conn.close()

if row is None:
    raise ValueError(
        f"No SAM 3 pieces found for puzzle={PUZZLE} origen={ORIGEN}. "
        "Run the SAM 3 extraction first or change PUZZLE/ORIGEN."
    )


def representative_rows_by_area(rows, n=5):
    if not rows:
        return []
    ordered = sorted(rows, key=lambda item: (item["area_pixels"], item["piece_index"]))
    positions = np.linspace(0, len(ordered) - 1, min(n, len(ordered)))
    selected = []
    seen_ids = set()
    for pos in positions:
        candidate = ordered[int(round(pos))]
        if candidate["id"] not in seen_ids:
            selected.append(candidate)
            seen_ids.add(candidate["id"])
    return selected


representative_rows = representative_rows_by_area(
    available,
    N_REPRESENTATIVE_PIECES,
)

first_ids = [
    {"id": item["id"], "index": item["piece_index"]}
    for item in available[:10]
]
representative_ids = [
    {"id": item["id"], "index": item["piece_index"], "area": item["area_pixels"]}
    for item in representative_rows
]

print(f"Loaded piece id={row['id']} index={row['piece_index']}: {row['ruta_imagen']}")
print(f"Available pieces for {PUZZLE}/{ORIGEN}: {len(available)}")
print(f"First IDs: {first_ids}")
print(f"Representative IDs by area quantile: {representative_ids}")
print(f"Contour points: {len(json.loads(row['contour_json']))}")
"""
    )
)

# Step 1: PNG + alpha
cells.append(
    nbf.v4.new_code_cell(
        """# Step 1: Compare alpha vs pipeline mask for representative pieces

img_path = Path(row["ruta_imagen"])
img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
if img is None:
    raise FileNotFoundError(f"Cannot read image: {img_path}")
print(f"Image shape: {img.shape}")

mask = load_alpha_mask(img_path)

columns = ["Piece RGB", "Alpha", "Pipeline mask"]
fig, axes = plt.subplots(
    len(representative_rows),
    len(columns),
    figsize=(4 * len(columns), 3.2 * len(representative_rows)),
)
if len(representative_rows) == 1:
    axes = np.expand_dims(axes, axis=0)

for row_idx, piece_row in enumerate(representative_rows):
    piece_path = Path(piece_row["ruta_imagen"])
    piece_img = cv2.imread(str(piece_path), cv2.IMREAD_UNCHANGED)
    if piece_img is None:
        continue

    if piece_img.shape[2] >= 4:
        rgb = cv2.cvtColor(piece_img[:, :, :3], cv2.COLOR_BGR2RGB)
        alpha = piece_img[:, :, 3]
        raw_mask = (alpha > 0).astype(np.uint8) * 255
    else:
        rgb = cv2.cvtColor(piece_img, cv2.COLOR_BGR2RGB)
        alpha = np.zeros(piece_img.shape[:2], dtype=np.uint8)
        raw_mask = alpha

    pipeline_mask = load_alpha_mask(piece_path)
    images = [rgb, alpha, pipeline_mask]

    for col_idx, (ax, title, image) in enumerate(zip(axes[row_idx], columns, images)):
        if col_idx == 0:
            ax.imshow(image)
        elif image is not None:
            ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.axis("off")
        if row_idx == 0:
            ax.set_title(title)

    raw_area = int(raw_mask.sum() / 255)
    pipeline_area = int(pipeline_mask.sum() / 255) if pipeline_mask is not None else 0
    axes[row_idx, 0].set_ylabel(
        f"id={piece_row['id']}\\nidx={piece_row['piece_index']}\\n"
        f"area={piece_row['area_pixels']}\\nraw={raw_area}\\n"
        f"pipe={pipeline_area}",
        rotation=0,
        ha="right",
        va="center",
        labelpad=42,
    )

for ax in axes.flat:
    ax.axis("off")
plt.tight_layout()
plt.show()
plt.close(fig)
"""
    )
)

# Step 2: Geometry preparation
cells.append(
    nbf.v4.new_code_cell(
        """# Step 2: Prepare geometry for the same representative pieces

def prepare_piece_geometry(piece_row):
    piece_path = Path(piece_row["ruta_imagen"])
    piece_img = cv2.imread(str(piece_path), cv2.IMREAD_UNCHANGED)
    piece_mask = load_alpha_mask(piece_path)
    result = {
        "row": piece_row,
        "path": piece_path,
        "img": piece_img,
        "mask": piece_mask,
        "quality": None,
        "raw_contour": None,
        "continuous_contour": None,
        "contour": None,
        "corner_list": [],
        "corner_method": "not_run",
        "corners_canonical": None,
        "faces": None,
        "centroid": None,
        "profile": None,
        "smoothing_rmse_px": None,
        "sampling_rmse_by_face": {},
        "error": None,
    }

    if piece_mask is None:
        result["error"] = "missing mask"
        return result

    result["quality"] = validate_mask(piece_mask)
    raw = extract_outer_contour(piece_mask)
    if raw is None:
        result["error"] = "missing contour"
        return result

    raw = ensure_clockwise(raw)
    continuous = contour_to_continuous_curve(raw)
    smooth = smooth_puzzle_contour(continuous)
    if len(smooth) > MAX_CONTOUR_POINTS_FOR_NOTEBOOK:
        smooth = smooth_puzzle_contour(
            continuous,
            n_points=MAX_CONTOUR_POINTS_FOR_NOTEBOOK,
        )
    continuous_aligned = contour_to_continuous_curve(continuous, n_points=len(smooth))
    result["smoothing_rmse_px"] = float(
        np.sqrt(np.mean(np.sum((continuous_aligned - smooth) ** 2, axis=1)))
    )

    result["raw_contour"] = raw
    result["continuous_contour"] = continuous
    result["contour"] = smooth

    corner_list, method = detect_corners(smooth)
    result["corner_list"] = corner_list
    result["corner_method"] = method

    if len(corner_list) >= 4:
        corners_arr = np.array([[c.x, c.y] for c in corner_list], dtype=np.float64)
        corners_canonical = order_corners_canonical(corners_arr)
        result["corners_canonical"] = corners_canonical

        corner_idx = []
        for c in corners_canonical:
            dists = np.sum((smooth - c) ** 2, axis=1)
            corner_idx.append(int(np.argmin(dists)))
        result["faces"] = split_contour_into_faces(smooth, corner_idx)

        ys, xs = np.where(piece_mask > 0)
        result["centroid"] = np.array([float(np.mean(xs)), float(np.mean(ys))])

    try:
        result["profile"] = build_piece_profile(dict(piece_row))
    except Exception as exc:
        result["error"] = str(exc)

    return result


piece_geometries = [
    prepare_piece_geometry(piece_row)
    for piece_row in representative_rows
]

for item in piece_geometries:
    piece_row = item["row"]
    quality = item["quality"]
    quality_status = quality.status if quality is not None else "missing"
    raw_len = len(item["raw_contour"]) if item["raw_contour"] is not None else 0
    smooth_len = len(item["contour"]) if item["contour"] is not None else 0
    smoothing_rmse = item["smoothing_rmse_px"]
    smoothing_text = (
        f"{smoothing_rmse:.2f}px"
        if smoothing_rmse is not None
        else "n/a"
    )
    print(
        f"id={piece_row['id']} idx={piece_row['piece_index']} "
        f"quality={quality_status} raw={raw_len} smooth={smooth_len} "
        f"smoothing_rmse={smoothing_text} "
        f"corners={len(item['corner_list'])} method={item['corner_method']}"
    )
"""
    )
)

# Step 3: Contour
cells.append(
    nbf.v4.new_code_cell(
        """# Step 3: Raw contour vs smoothed puzzle contour

for item in piece_geometries:
    piece_row = item["row"]
    piece_mask = item["mask"]
    raw_contour = item["raw_contour"]
    contour = item["contour"]
    fig, ax = plt.subplots(figsize=(10, 7))
    if piece_mask is None or raw_contour is None or contour is None:
        ax.set_title(f"id={piece_row['id']} contour unavailable")
        ax.axis("off")
        plt.show()
        plt.close(fig)
        continue

    ax.imshow(piece_mask, cmap="gray")
    ax.plot(
        raw_contour[:, 0],
        raw_contour[:, 1],
        color="tab:orange",
        linewidth=0.8,
        alpha=0.45,
        label="Raw contour",
    )
    ax.plot(
        contour[:, 0],
        contour[:, 1],
        color="red",
        linewidth=1.3,
        alpha=0.95,
        label="Smoothed puzzle contour",
    )
    ax.set_title(
        f"id={piece_row['id']} idx={piece_row['piece_index']} "
        f"raw={len(raw_contour)} smooth={len(contour)} "
        f"smoothing RMSE={item['smoothing_rmse_px']:.2f}px"
    )
    ax.legend()
    ax.axis("off")
    plt.tight_layout()
    plt.show()
    plt.close(fig)
"""
    )
)

# Step 4: Corners
cells.append(
    nbf.v4.new_code_cell(
        """# Step 4: Corner detection on the smoothed puzzle contour

for item in piece_geometries:
    piece_row = item["row"]
    piece_mask = item["mask"]
    contour = item["contour"]
    corners_canonical = item["corners_canonical"]
    fig, ax = plt.subplots(figsize=(10, 7))

    if piece_mask is None or contour is None or corners_canonical is None:
        ax.set_title(f"id={piece_row['id']} corners unavailable")
        ax.axis("off")
        plt.show()
        plt.close(fig)
        continue

    ax.imshow(piece_mask, cmap="gray")
    ax.plot(contour[:, 0], contour[:, 1], "b-", linewidth=0.8, alpha=0.6)

    labels = ["1", "2", "3", "4"]
    for i, (x, y) in enumerate(corners_canonical):
        ax.plot(x, y, "ro", markersize=8)
        ax.text(x + 3, y + 3, labels[i], fontsize=12, color="red", weight="bold")

    ax.set_title(
        f"id={piece_row['id']} idx={piece_row['piece_index']} "
        f"method={item['corner_method']}"
    )
    ax.axis("off")
    plt.tight_layout()
    plt.show()
    plt.close(fig)
"""
    )
)

# Step 5: Faces
cells.append(
    nbf.v4.new_code_cell(
        """# Step 5: Face split for the same representative pieces

colors = {"1-2": "red", "2-3": "green", "3-4": "blue", "4-1": "orange"}
for item in piece_geometries:
    piece_row = item["row"]
    piece_mask = item["mask"]
    faces = item["faces"]
    corners_canonical = item["corners_canonical"]
    fig, ax = plt.subplots(figsize=(10, 7))

    if piece_mask is None or faces is None or corners_canonical is None:
        ax.set_title(f"id={piece_row['id']} faces unavailable")
        ax.axis("off")
        plt.show()
        plt.close(fig)
        continue

    ax.imshow(piece_mask, cmap="gray")
    for key, pts in faces.items():
        ax.plot(pts[:, 0], pts[:, 1], color=colors[key], linewidth=2, label=key)

    for x, y in corners_canonical:
        ax.plot(x, y, "ko", markersize=5)

    ax.set_title(f"id={piece_row['id']} idx={piece_row['piece_index']}")
    ax.legend()
    ax.axis("off")
    plt.tight_layout()
    plt.show()
    plt.close(fig)
"""
    )
)

# Step 6: Normalized curves
cells.append(
    nbf.v4.new_code_cell(
        """# Step 6: Normalized face curves for the same representative pieces

face_keys = ["1-2", "2-3", "3-4", "4-1"]
for item in piece_geometries:
    piece_row = item["row"]
    faces = item["faces"]
    corners_canonical = item["corners_canonical"]
    centroid = item["centroid"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes_flat = axes.flatten()

    for col_idx, key in enumerate(face_keys):
        ax = axes_flat[col_idx]
        if faces is None or corners_canonical is None or centroid is None:
            ax.axis("off")
            continue

        curve = normalize_face_curve(
            faces[key],
            corners_canonical[col_idx],
            corners_canonical[(col_idx + 1) % 4],
            centroid,
        )
        classification = classify_face(curve)
        ax.plot(curve[:, 0], curve[:, 1], "b-", linewidth=1.5)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.4)
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.25)

        ax.set_title(key)
        ax.set_ylabel("Y perpendicular")
        ax.set_xlabel("X normalized (parametric)")
        ax.text(
            0.5,
            0.9,
            classification.face_type,
            transform=ax.transAxes,
            ha="center",
            fontsize=9,
        )

    fig.suptitle(f"id={piece_row['id']} idx={piece_row['piece_index']}")
    plt.tight_layout()
    plt.show()
    plt.close(fig)
"""
    )
)

# Step 7: Control-point reconstruction
cells.append(
    nbf.v4.new_code_cell(
        """# Step 7: Control-point descriptor for the same representative pieces

face_keys = ["1-2", "2-3", "3-4", "4-1"]
for item in piece_geometries:
    piece_row = item["row"]
    faces = item["faces"]
    corners_canonical = item["corners_canonical"]
    centroid = item["centroid"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes_flat = axes.flatten()

    for col_idx, key in enumerate(face_keys):
        ax = axes_flat[col_idx]
        if faces is None or corners_canonical is None or centroid is None:
            ax.axis("off")
            continue

        curve = normalize_face_curve(
            faces[key],
            corners_canonical[col_idx],
            corners_canonical[(col_idx + 1) % 4],
            centroid,
        )
        descriptor = control_point_descriptor(curve)
        control = np.array(descriptor).reshape((-1, 2))
        reconstructed = reconstruct_curve_from_control_points(descriptor)
        rmse = float(np.sqrt(np.mean(np.sum((curve - reconstructed) ** 2, axis=1))))
        item["sampling_rmse_by_face"][key] = rmse

        ax.plot(curve[:, 0], curve[:, 1], "b-", linewidth=1.4)
        ax.plot(reconstructed[:, 0], reconstructed[:, 1], "r--", linewidth=1.2)
        ax.plot(control[:, 0], control[:, 1], "ko", markersize=3)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.25)
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.25)

        ax.set_title(key)
        ax.set_ylabel("Y perpendicular")
        ax.set_xlabel("X normalized (parametric)")
        ax.text(
            0.5,
            0.9,
            f"RMSE={rmse:.3f}",
            transform=ax.transAxes,
            ha="center",
            fontsize=9,
        )

    fig.suptitle(f"id={piece_row['id']} idx={piece_row['piece_index']}")
    plt.tight_layout()
    plt.show()
    plt.close(fig)
"""
    )
)

# Step 8: Full profile
cells.append(
    nbf.v4.new_code_cell(
        """# Step 8: Full profile summary for the same representative pieces

for item in piece_geometries:
    piece_row = item["row"]
    profile = item["profile"]
    print("=" * 72)
    print(
        f"id={piece_row['id']} idx={piece_row['piece_index']} "
        f"path={piece_row['ruta_imagen']}"
    )
    if profile is None:
        print(f"Profile unavailable: {item['error']}")
        continue
    print(f"Profile version: {profile.profile_version}")
    print(f"Status: {profile.profile_status}")
    print(f"Piece kind: {profile.piece_kind}")
    print(f"Quality flags: {profile.quality_flags}")
    smoothing_rmse = item["smoothing_rmse_px"]
    if smoothing_rmse is not None:
        print(f"Smoothing RMSE: {smoothing_rmse:.2f}px")
    if item["sampling_rmse_by_face"]:
        sampling = {
            key: round(value, 4)
            for key, value in item["sampling_rmse_by_face"].items()
        }
        print(f"Sampling RMSE by face: {sampling}")
    for key, face in profile.faces.items():
        print(
            f"  {key}: type={face.face_type} "
            f"conf={face.metrics['classification_confidence']:.2f} "
            f"rmse={face.metrics['reconstruction_rmse']:.4f}"
        )
"""
    )
)

nb.cells = cells

nbf.write(nb, "notebooks/piece_profile_exploration.ipynb")
print("Notebook generated: notebooks/piece_profile_exploration.ipynb")
