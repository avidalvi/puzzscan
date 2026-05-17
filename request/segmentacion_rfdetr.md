# PRD: Segmentación de Piezas de Puzzle con IA

## 1. Objetivo Principal
Implementar un sistema de segmentación de instancias de piezas individuales de puzzle utilizando modelos de IA como alternativa al pipeline actual basado en OpenCV (`src/detector.py`). Ambos sistemas coexistirán en paralelo, permitiendo evaluar cuál ofrece mejores resultados en la extracción y delimitación de piezas.

## 2. Contexto del Proyecto Actual

### 2.1. Estructura de Código Existente
```
src/
├── db.py          # Módulo de base de datos (init_db, clear_origen, insert_pieza)
└── detector.py    # Pipeline OpenCV (process_image)
utils/
└── logger.py      # Logger centralizado (loguru)
data/
├── puzzscan_v2.db # Base de datos SQLite activa
└── ciudad/        # Imágenes fuente (piezas_*.jpg)
output/
└── ciudad/piezas/ # PNGs recortados con canal Alpha
tests/             # Vacío (sin tests aún)
```

### 2.2. Pipeline OpenCV Actual (`src/detector.py`)
1. Lee una imagen fuente (`data/{puzzle}/piezas_{origen}.jpg`).
2. Binariza con Otsu → `findContours` → filtra por área.
3. Normaliza escala (mediana → `TARGET_AREA = 10000`).
4. Descarta contornos fuera de rango `[0.4x, 1.5x]` de `TARGET_AREA`.
5. Rota por PCA (eje principal vertical).
6. Recorta con canal Alpha (fondo transparente).
7. Extrae colores dominantes (K-Means, k=2) y luminosidad.
8. Inserta en BBDD y guarda PNG en `output/{puzzle}/piezas/{origen}/pieza_{id}.png`.
9. Genera imagen anotada con IDs: `output/{puzzle}/piezas/{origen}_anotada.jpg`.
- **Idempotencia:** Ejecuta `clear_origen(nombre_puzzle, origen)` antes de procesar para borrar registros previos del mismo origen.
- **Entrada pública:** `process_image(filepath_str: str)`.

### 2.3. Esquema de BBDD Actual (Tabla `piezas`)
| Columna | Tipo | Descripción |
|---|---|---|
| `id` | INTEGER PK AUTOINCREMENT | ID único |
| `nombre_puzzle` | TEXT NOT NULL | Nombre del puzzle (ej. "ciudad") |
| `origen` | TEXT NOT NULL | Imagen fuente (ej. "piezas_1") |
| `ruta_imagen` | TEXT NOT NULL | Ruta al PNG recortado |
| `posicion_original` | TEXT NOT NULL | JSON `{x, y}` centro en imagen original |
| `factor_escala` | REAL NOT NULL | Factor de normalización aplicado |
| `colores_dominantes` | TEXT NOT NULL | JSON con colores RGB dominantes |
| `luminosidad` | REAL NOT NULL | Luminosidad media |
| `estado` | TEXT DEFAULT 'AVAILABLE' | Estado de la pieza |
| `fecha_creacion` | TIMESTAMP | Auto-generado |

### 2.4. Variables de Entorno (`.env`)
Actualmente se usa `python-dotenv`. Las nuevas variables necesarias para este módulo:
```env
SAM3_CONFIDENCE_THRESHOLD=0.5
SAM3_TEXT_PROMPT=puzzle piece
```

---

## 3. Fases del Proyecto

### Fase 1: Investigación y Análisis de Viabilidad (Completado)

#### Análisis de Librerías y Modelos

Se han evaluado tres enfoques para la segmentación de instancias de piezas de puzzle:

| Modelo | Tipo | Entrenamiento Necesario | Segmentación por Texto | Librería |
|---|---|---|---|---|
| **SAM 3** (Meta) | Foundation Model | **No** (zero-shot) | **Sí** (nativo) | `sam3` |
| **RF-DETR Seg** (Roboflow) | Detector + Seg Head | Sí (fine-tuning) | No | `rfdetr` + `supervision` |
| **YOLO26-seg** (Ultralytics) | Detector + Seg | Sí (fine-tuning) | No | `ultralytics` |

#### Decisión Estratégica: SAM 3 como Librería Principal

**SAM 3** (Segment Anything Model 3, Meta, noviembre 2025) introduce **Promptable Concept Segmentation (PCS)**: segmentación de instancias dirigida por texto en lenguaje natural. Esto permite buscar `"puzzle piece"` y obtener todas las piezas segmentadas con máscaras exactas a nivel de píxel **sin necesidad de entrenar, anotar datasets ni hacer fine-tuning**.

**Ventajas frente a RF-DETR y YOLO-Seg:**
- **Zero-shot:** No requiere dataset etiquetado ni entrenamiento. Se descarga el checkpoint de HuggingFace y funciona directamente.
- **Texto como prompt:** Se busca `"puzzle piece"` y SAM 3 detecta y segmenta todas las instancias automáticamente.
- **Máscaras de altísima calidad:** SAM 3 fue diseñado específicamente para segmentación de precisión, con resolución de máscara superior a modelos detectores.
- **Presence Head:** Arquitectura con cabezal de presencia que mejora la distinción entre objetos similares muy cercanos (ideal para piezas de puzzle agrupadas).

**Jerarquía de Prioridades:**
1. **Prioridad:** SAM 3 (`facebookresearch/sam3`) — zero-shot con prompt `"puzzle piece"`.
2. **Plan B:** RF-DETR Seg (`roboflow/rf-detr`) — si SAM 3 es demasiado pesado o lento para el flujo de trabajo.
3. **Plan C:** YOLO26-seg (`ultralytics`) — si se necesita velocidad extrema en tiempo real.

#### Referencia Técnica: SAM 3 (Librería Principal)
- **Paquete:** `sam3` (Meta, open-source vía GitHub: `pip install git+https://github.com/facebookresearch/sam3.git`).
- **Dependencias adicionales:** `torch`, `Pillow`, `matplotlib`, `scikit-learn`, `opencv-python`.
- **Requisitos:** GPU con soporte CUDA (Ampere+ recomendado para bf16).
- **Modelo:** Se descarga automáticamente desde HuggingFace. Requiere pasar `bpe_path` (vocabulario BPE incluido en el paquete) para text prompts.
- **API de inferencia (del notebook oficial `references/sam3_image_predictor_example.ipynb`):**
  1. `build_sam3_image_model(bpe_path=bpe_path)` → construye el modelo.
  2. `Sam3Processor(model, confidence_threshold=0.5)` → procesador con umbral.
  3. `processor.set_image(image)` → carga la imagen y retorna `inference_state`.
  4. `processor.reset_all_prompts(inference_state)` → limpia prompts previos.
  5. `processor.set_text_prompt(state=inference_state, prompt="puzzle piece")` → segmenta por texto.
- **Resultado de inferencia (`inference_state`):**
  - `inference_state["masks"]` → tensor binario `(N, 1, H, W)` con las máscaras de cada pieza.
  - `inference_state["boxes"]` → tensor `(N, 4)` con bounding boxes en formato `[x0, y0, x1, y1]`.
  - `inference_state["scores"]` → tensor `(N,)` con scores de confianza.
  - `inference_state["masks_logits"]` → logits sin umbralizar.
- **Visualización incluida:** `from sam3.visualization_utils import plot_results` → `plot_results(image, inference_state)`.
- **PoC con SAM 3 (basado en el notebook oficial):**
  ```python
  import os
  import torch
  from PIL import Image

  import sam3
  from sam3 import build_sam3_image_model
  from sam3.model.sam3_image_processor import Sam3Processor
  from sam3.visualization_utils import plot_results

  # Activar precisión mixta para GPUs Ampere+
  torch.backends.cuda.matmul.allow_tf32 = True
  torch.backends.cudnn.allow_tf32 = True
  torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

  # Construir modelo (requiere bpe_path para text prompts)
  sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
  bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
  model = build_sam3_image_model(bpe_path=bpe_path)

  # Cargar imagen y crear procesador
  image = Image.open("data/ciudad/piezas_1.jpg")
  processor = Sam3Processor(model, confidence_threshold=0.5)
  inference_state = processor.set_image(image)

  # Segmentar por texto (reset obligatorio antes de cada prompt)
  processor.reset_all_prompts(inference_state)
  inference_state = processor.set_text_prompt(
      state=inference_state, prompt="puzzle piece"
  )

  # Resultados
  masks = inference_state["masks"]     # (N, 1, H, W) máscaras binarias
  boxes = inference_state["boxes"]     # (N, 4) bounding boxes
  scores = inference_state["scores"]   # (N,) confianza

  print(f"Detectadas {len(scores)} piezas de puzzle")

  # Visualización integrada
  plot_results(image, inference_state)
  ```

#### Referencia Técnica: RF-DETR (Plan B)
- **Paquete:** `rfdetr` + `supervision`.
- **Modelos de segmentación:** `RFDETRSegMedium` (`from rfdetr import RFDETRSegMedium`).
- **Resultado:** Objeto `sv.Detections` con:
  - `detections.mask` → `(N, H, W)` booleano.
  - `detections.class_id`, `detections.confidence`.
  - `detections.metadata["source_image"]`.
- **Requiere:** Fine-tuning con dataset etiquetado (COCO JSON). Clase `puzzle_piece`.
- **PoC:**
  ```python
  import supervision as sv
  from rfdetr import RFDETRSegMedium

  model = RFDETRSegMedium()
  detections = model.predict("data/ciudad/piezas_1.jpg", threshold=0.5)

  # Máscaras: detections.mask.shape → (N, H, W)
  annotated = sv.MaskAnnotator().annotate(
      detections.metadata["source_image"], detections
  )
  sv.plot_image(annotated)
  ```

#### Referencia Técnica: YOLO26-seg (Plan C)
- **Paquete:** `ultralytics`.
- **Modelo nano:** `yolo26n-seg.pt` (`YOLO('yolo26n-seg.pt')`).
- **Resultado:** `result.masks.data` `(N,H,W)`, `result.masks.xy` (polígonos píxeles), `result.boxes.conf`, `result.boxes.xywh`.
- **Requiere:** Fine-tuning con dataset YOLO-seg format. Ultralytics incluye `convert_segment_masks_to_yolo_seg()`.
- **PoC:**
  ```python
  from ultralytics import YOLO

  model = YOLO("yolo26n-seg.pt")
  results = model.predict("data/ciudad/piezas_1.jpg", stream=True)

  for result in results:
      result.masks.data   # (N, H, W)
      result.masks.xy     # polígonos en píxeles
      result.boxes.conf   # confianza
  ```
- **Fine-tuning:** `model.train(data="dataset.yaml", epochs=100, imgsz=640)`.

### Fase 2: Implementación Técnica
- **Modelo Principal (SAM 3):**
  - **Sin entrenamiento:** No se necesita dataset, anotaciones ni fine-tuning. El modelo funciona zero-shot con el prompt `"puzzle piece"`.
  - **Hardware:** Requiere GPU con CUDA. Para GPUs Ampere+ se activa bf16 para mayor velocidad.
  - **Checkpoint:** Se descarga automáticamente desde HuggingFace. No es necesario almacenar pesos localmente en `data/models/` (se cachean en el directorio estándar de HuggingFace).
- **Plan B/C (RF-DETR, YOLO26-seg): Solo si SAM 3 falla:**
  - Requieren fine-tuning con dataset etiquetado.
  - Formato: COCO JSON (RF-DETR) o YOLO-seg `.txt` (Ultralytics).
  - Evaluar salidas de OpenCV como pre-anotaciones para acelerar la creación del dataset.
  - Los archivos `.pt` resultantes se almacenan en `data/models/` (añadir a `.gitignore`).
- **Integración No Invasiva (Arquitectura):**
  - Crear el módulo en **`src/sam3_extractor.py`** (mismo nivel que `detector.py`).
  - **API pública compatible:** Exponer `process_image(filepath_str: str)` con la misma firma que el pipeline OpenCV, para que ambos sean intercambiables.
  - **Configuración:** Prompt de texto y umbral de confianza desde `.env` usando `python-dotenv`.
  - **Logging:** Usar obligatoriamente `from utils.logger import logger`. **Prohibido `print()`**.
  - **Manejo de Errores:** Si una imagen falla, está corrupta o devuelve 0 piezas, emitir `logger.warning()`/`logger.error()` y continuar sin detener la ejecución del lote.
  - **Restricción Estricta:** No alterar `src/detector.py` ni `src/db.py`.
- **Post-Procesamiento de Imágenes:**
  - **Recortes Transparentes (Alpha):** Usar la máscara (`inference_state["masks"][i]`) para recortar cada pieza y guardarla como `.png` con fondo transparente (canal Alpha), idéntico al formato que genera OpenCV.
  - **Imagen de Depuración:** Generar y guardar la imagen original completa con todas las máscaras superpuestas para verificación visual.
- **Estructura de Salida:**
  ```
  output/{nombre_puzzle}/piezas_sam3/{origen}/
  ├── pieza_{id}.png          # Recorte individual con Alpha
  └── ...
  output/{nombre_puzzle}/piezas_sam3/
  └── {origen}_debug.jpg      # Imagen completa con máscaras superpuestas
  ```
- **Idempotencia:** Antes de procesar, borrar todos los registros previos de la tabla SAM 3 para ese `nombre_puzzle` + `origen` (mismo patrón que `clear_origen`).
- **Calidad de Código:** Todo el código nuevo debe pasar `ruff`, `mypy --strict` y `detect-secrets` (hooks de `pre-commit`).
- **Gestión de Entorno:** Añadir dependencias usando **`uv add`**:
  - Producción: `sam3`, `torch`, `Pillow`.
  - Desarrollo: `uv add --dev pytest`.
- **Testing (Obligatorio):** Tests unitarios en `tests/test_sam3_extractor.py`:
  - Test de carga de configuración desde `.env`.
  - Test de inferencia con imagen sintética/mock.
  - Test de generación de PNG con canal Alpha.
  - Test de inserción y limpieza idempotente en BBDD.
  - Test de manejo de errores (imagen corrupta, 0 detecciones).
  - *(Los notebooks quedan exentos de tests.)*

### Fase 3: Persistencia de Datos (Esquema de Base de Datos)
- **Base de Datos:** Crear una nueva tabla `sam3_piezas` en `data/puzzscan_v2.db`.
- **Esquema de Tabla:**

| Columna | Tipo | Descripción |
|---|---|---|
| `id` | INTEGER PK AUTOINCREMENT | ID único |
| `nombre_puzzle` | TEXT NOT NULL | Nombre del puzzle (ej. "ciudad") |
| `origen` | TEXT NOT NULL | Imagen fuente (ej. "piezas_1") |
| `piece_index` | INTEGER NOT NULL | Índice de la pieza dentro de la imagen |
| `ruta_imagen` | TEXT NOT NULL | Ruta al PNG recortado con Alpha |
| `contour_json` | TEXT NOT NULL | Polígono exacto (lista de coordenadas `[[x,y],...]`) |
| `bounding_box_json` | TEXT NOT NULL | Caja delimitadora `[x0, y0, x1, y1]` |
| `confidence_score` | REAL NOT NULL | Confianza del modelo (0.0–1.0) |
| `area_pixels` | INTEGER NOT NULL | Área interna de la máscara en píxeles |
| `posicion_original` | TEXT NOT NULL | JSON `{x, y}` centro en imagen original |
| `text_prompt` | TEXT NOT NULL | Prompt utilizado (ej. "puzzle piece") |
| `debug_image_path` | TEXT | Ruta a la imagen con máscaras superpuestas |
| `estado` | TEXT DEFAULT 'AVAILABLE' | Estado de la pieza (AVAILABLE / PLACED) |
| `fecha_creacion` | TIMESTAMP DEFAULT CURRENT_TIMESTAMP | Auto-generado |

- **Restricción Estricta:** La tabla `piezas` (OpenCV) queda bloqueada contra cualquier modificación.

### Fase 4: Evaluación y Comparativa (Métricas y Visualización)
- **Ejecución en Paralelo:** Procesar el mismo conjunto de imágenes (`data/ciudad/piezas_*.jpg`) con ambos pipelines de forma independiente.
- **Métricas a Evaluar:**
  1. **Tasa de Recuperación:** Piezas detectadas correctamente vs total real (sin fusiones ni omisiones).
  2. **Precisión de Contornos (IoU):** Fidelidad de la máscara respecto a la forma geométrica real.
  3. **Robustez:** Resistencia a brillos, sombras y piezas muy cercanas (debilidad conocida de OpenCV/Otsu).
  4. **Rendimiento Computacional:** Tiempo de inferencia (ms/imagen).
- **Visualización Analítica:** Notebook interactivo (`notebooks/compare_segmentation.ipynb`) para visualizar lado a lado: imagen original → salida OpenCV → salida SAM 3.

---

## 4. Entregables Esperados
1. **Módulo de Inferencia IA:** `src/sam3_extractor.py` con `process_image()`, manejo de errores, generación de PNGs transparentes e imágenes de debug.
2. **Función de BBDD:** `init_sam3_db()` para crear la tabla `sam3_piezas` (puede ir en `src/db.py` o en módulo propio).
3. **Batería de Tests:** `tests/test_sam3_extractor.py`.
4. **Dependencias:** `pyproject.toml` actualizado vía `uv add`.
5. **Configuración:** Variables `SAM3_*` documentadas y añadidas a `.env.example`.
6. **Notebook Comparativo:** `notebooks/compare_segmentation.ipynb`.

---

## 5. Conclusiones Arquitectónicas

### Detección vs Segmentación
- **Segmentación de Instancias es Obligatoria:** Las bounding boxes son insuficientes para piezas de puzzle con formas altamente irregulares (pestañas, huecos).
- **Contornos Exactos:** Requisito vital para el futuro algoritmo de encaje geométrico.

### Estrategia de Implementación
- **Prioridad:** SAM 3 (`facebookresearch/sam3`) con prompt `"puzzle piece"` — zero-shot, sin entrenamiento.
- **Plan B:** RF-DETR Seg (`roboflow/rf-detr`) con `RFDETRSegMedium` — requiere fine-tuning.
- **Plan C:** YOLO26-seg (`ultralytics`) con `yolo26n-seg.pt` — requiere fine-tuning, máxima velocidad.
- **Principio de No Invasión:** Cero modificaciones al código OpenCV existente. Los pipelines son módulos independientes e intercambiables.