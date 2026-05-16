import cv2
import numpy as np
from pathlib import Path
import math
import sqlite3
from utils.logger import logger
from src.db import init_db, clear_origen, insert_pieza

TARGET_AREA = 10000.0  # Normalized area for pieces
NUM_CONTOUR_POINTS = 500



def get_dominant_colors(image, mask, k=2):
    """Extracts dominant colors using K-Means and returns them along with average luminosity."""
    pixels = image[mask > 0]
    if len(pixels) == 0:
        return [], 0.0
        
    pixels = np.float32(pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    centers = np.uint8(centers)
    # Convert BGR to RGB for JSON storage
    rgb_centers = [ [int(c[2]), int(c[1]), int(c[0])] for c in centers ]
    
    # Luminosity
    gray_pixels = cv2.cvtColor(np.array([centers]), cv2.COLOR_BGR2GRAY)[0]
    avg_luminosity = float(np.mean(gray_pixels))
    
    return rgb_centers, avg_luminosity

def process_image(filepath_str: str):
    path = Path(filepath_str)
    nombre_puzzle = path.parent.name
    origen = path.stem
    
    logger.info(f"Processing puzzle: {nombre_puzzle}, origen: {origen}")
    
    # Idempotency
    clear_origen(nombre_puzzle, origen)
    
    output_dir = Path(f"output/{nombre_puzzle}/piezas/{origen}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    img = cv2.imread(str(path))
    if img is None:
        logger.error(f"Could not read {path}")
        return
        
    annotated_img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Binarize with Otsu
    _, solid_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find final contours on the solid mask
    final_contours, _ = cv2.findContours(solid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter by area (e.g. > 1000)
    valid_contours = [c for c in final_contours if cv2.contourArea(c) > 1000]
    
    if not valid_contours:
        logger.warning("No valid pieces found in image.")
        return
        
    # Scale Normalization
    areas = [cv2.contourArea(c) for c in valid_contours]
    median_area = np.median(areas)
    scale_factor = math.sqrt(TARGET_AREA / median_area) if median_area > 0 else 1.0
    logger.debug(f"Median Area: {median_area:.1f}. Scale factor: {scale_factor:.3f}")
    
    img_scaled = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor)
    mask_scaled = cv2.resize(solid_mask, (0, 0), fx=scale_factor, fy=scale_factor)
    scaled_contours, _ = cv2.findContours(mask_scaled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_scaled_contours = []
    for c in scaled_contours:
        area = cv2.contourArea(c)
        # Assuming TARGET_AREA is 10000, filtering small noise
        if area < TARGET_AREA * 0.4:
            continue
        # Si el área es un 50% mayor de lo normal, son piezas fusionadas/pegadas.
        if area > TARGET_AREA * 1.5:
            logger.warning(f"Descartando contorno sobredimensionado (posibles piezas pegadas). Área: {area:.1f}")
            continue
        valid_scaled_contours.append(c)
    
    for c in valid_scaled_contours:
        # Original coordinates
        M = cv2.moments(c)
        if M['m00'] == 0: continue
        cx_scaled = int(M['m10']/M['m00'])
        cy_scaled = int(M['m01']/M['m00'])
        
        # Scale back to original coordinates for traceability
        cx_orig = int(cx_scaled / scale_factor)
        cy_orig = int(cy_scaled / scale_factor)
        
        # PCA for rotation
        data_pts = np.empty((len(c), 2), dtype=np.float64)
        data_pts[:, 0] = c[:, 0, 0]
        data_pts[:, 1] = c[:, 0, 1]
        mean, eigenvectors = cv2.PCACompute(data_pts, mean=None)
        angle = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0]) * 180 / np.pi
        
        # We want the principal axis vertical => rotate by (90 - angle)
        rot_angle = 90 - angle
        
        # Bounding box of the piece to extract ROI
        x, y, w, h = cv2.boundingRect(c)
        pad = max(w, h)
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(img_scaled.shape[1], x + w + pad), min(img_scaled.shape[0], y + h + pad)
        
        roi_img = img_scaled[y1:y2, x1:x2]
        roi_mask = mask_scaled[y1:y2, x1:x2]
        
        if roi_img.size == 0: continue
        
        # Rotate ROI
        center = (roi_img.shape[1]//2, roi_img.shape[0]//2)
        M_rot = cv2.getRotationMatrix2D(center, rot_angle, 1.0)
        
        # Use a slightly larger bounds to avoid clipping corners? For now assume padding is enough
        rot_img = cv2.warpAffine(roi_img, M_rot, (roi_img.shape[1], roi_img.shape[0]))
        rot_mask = cv2.warpAffine(roi_mask, M_rot, (roi_mask.shape[1], roi_mask.shape[0]))
        
        # Tightly crop the rotated piece
        rot_contours, _ = cv2.findContours(rot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not rot_contours: continue
        best_rot_c = max(rot_contours, key=cv2.contourArea)
        rx, ry, rw, rh = cv2.boundingRect(best_rot_c)
        
        final_img = rot_img[ry:ry+rh, rx:rx+rw]
        final_mask = rot_mask[ry:ry+rh, rx:rx+rw]
        
        if final_img.size == 0: continue
        
        # Add Alpha channel
        b, g, r = cv2.split(final_img)
        final_rgba = cv2.merge([b, g, r, final_mask])
        
        # Colors
        colors, lum = get_dominant_colors(final_img, final_mask)
        
        # Insert to DB
        pieza_id = insert_pieza(
            nombre_puzzle=nombre_puzzle,
            origen=origen,
            ruta_imagen="", # Will update after ID is known
            posicion_original={"x": cx_orig, "y": cy_orig},
            factor_escala=scale_factor,
            colores_dominantes=colors,
            luminosidad=lum
        )
        
        # Save image
        out_filepath = output_dir / f"pieza_{pieza_id}.png"
        cv2.imwrite(str(out_filepath), final_rgba)
        
        # Update DB with path
        conn = sqlite3.connect("data/puzzscan_v2.db")
        conn.execute("UPDATE piezas SET ruta_imagen = ? WHERE id = ?", (str(out_filepath), pieza_id))
        conn.commit()
        conn.close()
        
        # Annotate original image
        cv2.putText(annotated_img, str(pieza_id), (cx_orig, cy_orig), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        cv2.circle(annotated_img, (cx_orig, cy_orig), 5, (0, 255, 0), -1)

    annotated_path = output_dir.parent / f"{origen}_anotada.jpg"
    cv2.imwrite(str(annotated_path), annotated_img)
    logger.info(f"Finished {origen}. Annotated saved to {annotated_path}")

if __name__ == "__main__":
    init_db()
    # Process all piezas files in data/ciudad
    import glob
    for f in glob.glob("data/ciudad/piezas_*.jpg"):
        process_image(f)
