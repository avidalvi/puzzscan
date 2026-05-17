import nbformat as nbf

nb = nbf.v4.new_notebook()

# Markdown cell
text = """\
# Comparativa de Segmentación: OpenCV vs SAM 3

Este notebook compara visualmente y cuantitativamente la extracción de piezas usando el pipeline tradicional basado en OpenCV vs el nuevo modelo Zero-Shot SAM 3.
"""

# Code cell
code = """\
import os
import sqlite3
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image

# Ejecutar desde el root del proyecto
if Path.cwd().name == "notebooks":
    os.chdir("..")

# Configuración
PUZZLE = "ciudad"
ORIGEN = "piezas_1"
DB_PATH = "data/puzzscan_v2.db"

# Cargar imagen original
img_path = Path(f"data/{PUZZLE}/{ORIGEN}.jpg")
img_bgr = cv2.imread(str(img_path))
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(img_rgb)
ax.set_title(f"Imagen Original: {ORIGEN}")
ax.axis("off")
plt.show()
"""

code2 = """\
# Consultar OpenCV
conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
opencv_piezas = conn.execute("SELECT * FROM piezas WHERE nombre_puzzle=? AND origen=?", (PUZZLE, ORIGEN)).fetchall()

# Consultar SAM 3
sam3_piezas = conn.execute("SELECT * FROM sam3_piezas WHERE nombre_puzzle=? AND origen=?", (PUZZLE, ORIGEN)).fetchall()
conn.close()

print(f"Piezas detectadas con OpenCV: {len(opencv_piezas)}")
print(f"Piezas detectadas con SAM 3: {len(sam3_piezas)}")
"""

code3 = """\
# Mostrar comparativa visual
fig, axes = plt.subplots(1, 2, figsize=(20, 10))

# OpenCV
opencv_debug_path = Path(f"output/{PUZZLE}/piezas/{ORIGEN}_anotada.jpg")
if opencv_piezas and opencv_debug_path.exists():
    cv_debug = cv2.cvtColor(cv2.imread(str(opencv_debug_path)), cv2.COLOR_BGR2RGB)
    axes[0].imshow(cv_debug)
axes[0].set_title(f"OpenCV (Piezas: {len(opencv_piezas)})")
axes[0].axis("off")

# SAM 3
if sam3_piezas and sam3_piezas[0]["debug_image_path"]:
    sam3_debug = cv2.cvtColor(cv2.imread(sam3_piezas[0]["debug_image_path"]), cv2.COLOR_BGR2RGB)
    axes[1].imshow(sam3_debug)
axes[1].set_title(f"SAM 3 (Piezas: {len(sam3_piezas)})")
axes[1].axis("off")

plt.tight_layout()
plt.show()
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text),
    nbf.v4.new_code_cell(code),
    nbf.v4.new_code_cell(code2),
    nbf.v4.new_code_cell(code3)
]

nbf.write(nb, 'notebooks/compare_segmentation.ipynb')
print("Notebook generado correctamente.")
