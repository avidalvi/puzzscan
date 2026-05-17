"""SAM 3 image segmentation pipeline for puzzle pieces.

Extracts individual puzzle pieces from source images using Meta's SAM 3
model with text prompts ("puzzle piece"). Runs zero-shot — no training
or labeled dataset required.

Public API mirrors src/detector.py for interchangeability:
    process_image(filepath_str: str) -> None
"""

import glob
import os
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import torch
from dotenv import load_dotenv
from PIL import Image

from sam3.model.sam3_image_processor import Sam3Processor
from utils.logger import logger

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration (loaded from .env)
# ---------------------------------------------------------------------------
CONFIDENCE_THRESHOLD: float = float(os.getenv("SAM3_CONFIDENCE_THRESHOLD", "0.5"))
TEXT_PROMPT: str = os.getenv("SAM3_TEXT_PROMPT", "puzzle piece")

# ---------------------------------------------------------------------------
# Lazy model singleton — loaded once, reused for all images in the batch
# ---------------------------------------------------------------------------
_model: object = None
_bpe_path: str | None = None


class _PieceDetection(NamedTuple):
    """Post-processed SAM detection data ready for ordered output."""

    mask: np.ndarray
    box: list[float]
    score: float
    contour: list[list[int]]
    area: int
    center_x: int
    center_y: int


def _get_bpe_path() -> str:
    """Returns the path to the local BPE vocabulary file."""
    bpe_path = Path("assets/bpe_simple_vocab_16e6.txt.gz").resolve()
    return str(bpe_path)


def load_model() -> object:
    """Load (or return cached) SAM 3 model with CUDA mixed-precision setup.

    Returns:
        The SAM 3 model instance (cached singleton).
    """
    global _model, _bpe_path  # noqa: PLW0603
    if _model is not None:
        return _model

    # Enable TF32 for Ampere+ GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # bfloat16 autocast for the whole session
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()  # type: ignore[no-untyped-call]

    from sam3 import build_sam3_image_model

    _bpe_path = _get_bpe_path()
    logger.info(f"Loading SAM 3 model (bpe_path={_bpe_path})")
    _model = build_sam3_image_model(bpe_path=_bpe_path)
    logger.info("SAM 3 model loaded successfully")
    return _model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_contour(mask: np.ndarray) -> list[list[int]]:
    """Extract the largest contour from a binary mask.

    Args:
        mask: Binary mask of shape (H, W) dtype uint8.

    Returns:
        List of [x, y] coordinate pairs for the contour polygon,
        or empty list if no contour found.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    largest = max(contours, key=cv2.contourArea)
    # Squeeze to (N, 2) and convert to plain Python ints
    pts = largest.squeeze()
    if pts.ndim == 1:
        pts = pts[np.newaxis, :]
    result: list[list[int]] = pts.tolist()
    return result


def _crop_with_alpha(
    img_bgr: np.ndarray,
    mask: np.ndarray,
    contour_list: list[list[int]] | None = None,
) -> np.ndarray:
    """Crop a piece from the source image, rotate it via PCA, and apply Alpha.

    Args:
        img_bgr: Source image in BGR format (H, W, 3).
        mask: Binary mask (H, W) dtype uint8, values 0 or 255.
        contour_list: Polygon contour to calculate PCA rotation.

    Returns:
        RGBA crop of the piece (h, w, 4) with transparent background.
    """
    if contour_list is None:
        contour_list = _extract_contour(mask)

    if not contour_list:
        return np.zeros((1, 1, 4), dtype=np.uint8)

    # Calculate PCA rotation angle
    pts = np.array(contour_list, dtype=np.float64)
    mean, eigenvectors = cv2.PCACompute(pts, mean=None)  # type: ignore[call-overload]
    angle = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0]) * 180 / np.pi
    rot_angle = 90 - angle

    # Bounding box with padding
    x, y, w, h = cv2.boundingRect(np.array(contour_list, dtype=np.int32))
    pad = max(w, h)
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(img_bgr.shape[1], x + w + pad), min(img_bgr.shape[0], y + h + pad)

    roi_img = img_bgr[y1:y2, x1:x2]
    roi_mask = mask[y1:y2, x1:x2]

    if roi_img.size == 0:
        return np.zeros((1, 1, 4), dtype=np.uint8)

    # Rotate ROI
    center = (roi_img.shape[1] // 2, roi_img.shape[0] // 2)
    M_rot = cv2.getRotationMatrix2D(center, rot_angle, 1.0)
    rot_img = cv2.warpAffine(roi_img, M_rot, (roi_img.shape[1], roi_img.shape[0]))
    rot_mask = cv2.warpAffine(roi_mask, M_rot, (roi_mask.shape[1], roi_mask.shape[0]))

    # Tight crop the rotated piece
    rot_contours, _ = cv2.findContours(
        rot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not rot_contours:
        return np.zeros((1, 1, 4), dtype=np.uint8)
    best_rot_c = max(rot_contours, key=cv2.contourArea)
    rx, ry, rw, rh = cv2.boundingRect(best_rot_c)

    final_img = rot_img[ry : ry + rh, rx : rx + rw]
    final_mask = rot_mask[ry : ry + rh, rx : rx + rw]

    if final_img.size == 0:
        return np.zeros((1, 1, 4), dtype=np.uint8)

    b, g, r = cv2.split(final_img)
    rgba = cv2.merge([b, g, r, final_mask])
    return rgba


def _build_debug_image(
    img_bgr: np.ndarray,
    masks: list[np.ndarray],
    boxes: list[list[float]] | None = None,
    labels: list[int] | None = None,
) -> np.ndarray:
    """Overlay all piece masks on the source image for visual debugging.

    Args:
        img_bgr: Source image in BGR (H, W, 3).
        masks: List of binary uint8 masks (H, W), one per piece.
        boxes: Optional list of bounding boxes [x0, y0, x1, y1] for each piece.
        labels: Optional labels to draw over each piece.

    Returns:
        Annotated BGR image with semi-transparent coloured overlays.
    """
    rng = np.random.default_rng(42)
    overlay = img_bgr.copy().astype(np.float32)

    # 1. Paint semi-transparent masks
    for mask in masks:
        color = rng.integers(0, 255, size=3).tolist()
        colored = np.zeros_like(img_bgr, dtype=np.float32)
        colored[mask > 0] = color
        overlay_weighted: np.ndarray = cv2.addWeighted(
            overlay, 1.0, colored, 0.4, 0
        )
        overlay = overlay_weighted

    # 2. Convert back to uint8 to draw shapes
    final_img = overlay.astype(np.uint8)

    # 3. Draw bounding boxes and text
    for i, mask in enumerate(masks):
        if boxes and i < len(boxes):
            box = boxes[i]
            x0, y0, x1, y1 = [int(v) for v in box]
            cv2.rectangle(final_img, (x0, y0), (x1, y1), (0, 255, 0), 2)

        ys, xs = np.where(mask > 0)
        if len(ys):
            cx, cy = int(xs.mean()), int(ys.mean())
            label = labels[i] if labels and i < len(labels) else i + 1
            cv2.putText(
                final_img,
                str(label),
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (255, 255, 255),
                3,
            )

    return final_img


def _sort_detections_top_left_to_bottom_right(
    detections: list[_PieceDetection],
) -> list[_PieceDetection]:
    """Sort detections in stable row-major order using their source positions."""
    if not detections:
        return []

    heights = [detection.box[3] - detection.box[1] for detection in detections]
    row_threshold = max(1.0, float(np.median(heights)) * 0.5)
    rows: list[list[_PieceDetection]] = []

    for detection in sorted(detections, key=lambda item: item.center_y):
        if not rows:
            rows.append([detection])
            continue

        row_center = float(np.mean([item.center_y for item in rows[-1]]))
        if abs(detection.center_y - row_center) <= row_threshold:
            rows[-1].append(detection)
        else:
            rows.append([detection])

    ordered: list[_PieceDetection] = []
    for row in rows:
        ordered.extend(sorted(row, key=lambda detection: detection.box[0]))
    return ordered


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def process_image(filepath_str: str) -> None:
    """Segment puzzle pieces in a source image using SAM 3.

    Mirrors the public API of src/detector.py. Saves cropped pieces as
    PNGs with alpha channel and inserts records into sam3_piezas table.

    Args:
        filepath_str: Path to a source puzzle image (e.g. data/ciudad/piezas_1.jpg).
    """
    from src.sam3_db import (
        clear_sam3_origen,
        insert_sam3_pieza,
        update_sam3_debug_path,
        update_sam3_ruta_imagen,
    )

    path = Path(filepath_str)
    nombre_puzzle = path.parent.name
    origen = path.stem

    logger.info(f"Processing SAM 3 | puzzle='{nombre_puzzle}' origen='{origen}'")

    # Idempotency: clear previous records for this origen
    clear_sam3_origen(nombre_puzzle, origen)

    # Output directories
    out_dir = Path(f"output/{nombre_puzzle}/piezas_sam3/{origen}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load image (PIL for SAM 3, OpenCV for post-processing)
    try:
        pil_image = Image.open(path)
        img_bgr = cv2.imread(str(path))
        if img_bgr is None:
            raise OSError(f"cv2.imread returned None for {path}")
    except Exception as exc:
        logger.error(f"Cannot read image '{path}': {exc}")
        return

    # SAM 3 inference
    try:
        model = load_model()
        processor = Sam3Processor(model, confidence_threshold=CONFIDENCE_THRESHOLD)
        inference_state = processor.set_image(pil_image)
        processor.reset_all_prompts(inference_state)
        inference_state = processor.set_text_prompt(
            state=inference_state, prompt=TEXT_PROMPT
        )
    except Exception as exc:
        logger.error(f"SAM 3 inference failed for '{path}': {exc}")
        return

    masks_tensor = inference_state["masks"]  # (N, 1, H, W)
    boxes_tensor = inference_state["boxes"]  # (N, 4)
    scores_tensor = inference_state["scores"]  # (N,)

    n_pieces = len(scores_tensor)
    if n_pieces == 0:
        logger.warning(
            f"No puzzle pieces detected in '{path}' "
            f"(prompt='{TEXT_PROMPT}', threshold={CONFIDENCE_THRESHOLD})"
        )
        return

    logger.info(f"Detected {n_pieces} pieces in '{origen}'")

    # Convert tensors to numpy and normalize model output to a spatial order.
    detections: list[_PieceDetection] = []
    first_pieza_id: int | None = None

    for i in range(n_pieces):
        mask = masks_tensor[i, 0].cpu().numpy().astype(np.uint8) * 255  # (H, W)
        box = boxes_tensor[i].cpu().tolist()  # [x0,y0,x1,y1]
        score = float(scores_tensor[i].cpu())

        # Derived features
        contour = _extract_contour(mask)
        area = int((mask > 0).sum())

        ys, xs = np.where(mask > 0)
        cx = int(xs.mean()) if len(xs) else 0
        cy = int(ys.mean()) if len(ys) else 0

        detections.append(
            _PieceDetection(
                mask=mask,
                box=box,
                score=score,
                contour=contour,
                area=area,
                center_x=cx,
                center_y=cy,
            )
        )

    ordered_detections = _sort_detections_top_left_to_bottom_right(detections)
    logger.info("Sorted detections from top-left to bottom-right")

    masks_np: list[np.ndarray] = []
    boxes_list: list[list[float]] = []
    labels: list[int] = []

    for i, detection in enumerate(ordered_detections, start=1):
        mask = detection.mask

        # Insert into DB (ruta_imagen updated below once file is saved)
        pieza_id = insert_sam3_pieza(
            nombre_puzzle=nombre_puzzle,
            origen=origen,
            piece_index=i,
            ruta_imagen="",  # placeholder — updated after save
            contour_json=detection.contour,
            bounding_box_json=detection.box,
            confidence_score=detection.score,
            area_pixels=detection.area,
            posicion_original={"x": detection.center_x, "y": detection.center_y},
            text_prompt=TEXT_PROMPT,
        )

        # Save cropped PNG with alpha
        rgba = _crop_with_alpha(img_bgr, mask, detection.contour)
        out_path = out_dir / f"pieza_{i}.png"
        cv2.imwrite(str(out_path), rgba)
        update_sam3_ruta_imagen(pieza_id, str(out_path))

        logger.debug(
            f"  Piece {i}: id={pieza_id} score={detection.score:.3f} "
            f"area={detection.area}px saved='{out_path}'"
        )

        masks_np.append(mask)
        boxes_list.append(detection.box)
        labels.append(i)
        if first_pieza_id is None:
            first_pieza_id = pieza_id

    # Build and save debug overlay
    debug_img = _build_debug_image(img_bgr, masks_np, boxes_list, labels)
    debug_path = Path(f"output/{nombre_puzzle}/piezas_sam3") / f"{origen}_debug.jpg"
    cv2.imwrite(str(debug_path), debug_img)

    if first_pieza_id is not None:
        update_sam3_debug_path(first_pieza_id, str(debug_path))

    logger.info(
        f"Finished '{origen}': {n_pieces} pieces extracted. "
        f"Debug image saved to '{debug_path}'"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.sam3_db import init_sam3_db

    init_sam3_db()
    load_model()

    images = sorted(glob.glob("data/ciudad/piezas_*.jpg"))
    if not images:
        logger.warning("No source images found in data/ciudad/")
    else:
        logger.info(f"Processing {len(images)} image(s): {images}")
        for img_path in images:
            process_image(img_path)
