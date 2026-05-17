# Plan Tecnico: Obtencion del Perfil Geometrico de Pieza

> **PRD de referencia:** `docs/PRD_obtencion_perfil_pieza.md`
> **Fecha:** 2026-05-17

---

## Resumen

Implementar una fase nueva que toma las piezas PNG ya extraidas, usa su canal
alpha como mascara principal, obtiene el contorno exterior, detecta cuatro
esquinas, divide la pieza en caras `1-2-3-4-1`, genera curvas normalizadas con
numero fijo de puntos por cara, clasifica cada cara y guarda el perfil en
SQLite.

Esta fase no hace matching contra huecos. Produce datos geometricos estables,
auditables y comparables para fases posteriores.

---

## Decisiones Tecnicas Iniciales

- Fuente principal de mascara: canal alpha del PNG en `ruta_imagen`.
- Fallback: `contour_json` de `sam3_piezas` solo si el alpha no existe o no es
  valido.
- Numero inicial de puntos por cara: `128`.
- Descriptor reducido inicial: coeficientes Fourier de baja frecuencia.
- Orden canonico: contorno horario, puntos `1=superior izquierda`,
  `2=superior derecha`, `3=inferior derecha`, `4=inferior izquierda`.
- Signo: `Y positivo` apunta hacia el exterior de la pieza.
- Tabla nueva: `pieza_perfiles`.
- Version inicial de perfil: `piece_profile_v1`.

---

## Estructura Propuesta

```text
src/piece_profile.py
src/piece_profile_db.py
scripts/generate_piece_profile_notebook.py
tests/test_piece_profile.py
tests/test_piece_profile_db.py
notebooks/piece_profile_exploration.ipynb
```

No se debe modificar la semantica de `sam3_piezas`; esta fase consume sus
salidas y genera una tabla nueva.

---

## T1: Configuracion y Constantes

**Objetivo:** Centralizar parametros del perfil para que el notebook y el
pipeline usen los mismos valores.

- [ ] **T1.1** Definir constantes en `src/piece_profile.py`:

  ```python
  PROFILE_VERSION = "piece_profile_v1"
  POINTS_PER_FACE = 128
  FOURIER_COEFFICIENTS = 16
  MIN_FACE_POINTS = 16
  MIN_CLASSIFICATION_CONFIDENCE = 0.70
  MAX_RECONSTRUCTION_RMSE = 0.03
  ```

- [ ] **T1.2** Permitir override por variables `.env` si se considera util:

  ```text
  PIECE_PROFILE_POINTS_PER_FACE=128
  PIECE_PROFILE_FOURIER_COEFFICIENTS=16
  ```

- [ ] **T1.3** Usar `from utils.logger import logger`; no usar `print()` fuera de
  notebooks.

**Criterio de aceptacion:** los parametros viven en un unico sitio y son
importables desde tests, scripts y notebook.

---

## T2: Persistencia en SQLite

**Objetivo:** Crear una tabla nueva para perfiles sin romper las tablas
existentes.

- [ ] **T2.1** Crear `src/piece_profile_db.py`.
- [ ] **T2.2** Implementar `init_piece_profile_db()`.
- [ ] **T2.3** Implementar `clear_piece_profile(pieza_id, profile_version)`.
- [ ] **T2.4** Implementar `insert_piece_profile(...)`.
- [ ] **T2.5** Implementar consulta de piezas SAM 3 disponibles:

  ```python
  fetch_sam3_pieces(nombre_puzzle: str | None = None, origen: str | None = None)
  ```

- [ ] **T2.6** Crear tabla:

  ```sql
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
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      UNIQUE(pieza_id, profile_version)
  );
  ```

**Criterio de aceptacion:** la tabla se crea de forma idempotente y una segunda
ejecucion para la misma pieza/version reemplaza o recrea el perfil sin duplicar.

---

## T3: Lectura de Mascara desde Alpha

**Objetivo:** Obtener una mascara binaria fiable desde el PNG recortado.

- [ ] **T3.1** Implementar:

  ```python
  load_alpha_mask(image_path: Path) -> np.ndarray
  ```

- [ ] **T3.2** Validar que la imagen existe y tiene canal alpha.
- [ ] **T3.3** Binarizar alpha:

  ```text
  mask = alpha > 0
  ```

  o usar umbral configurable si aparecen bordes semitransparentes.

- [ ] **T3.4** Limpiar mascara con operaciones morfologicas suaves si hace
  falta:

  - cerrar pequenos huecos;
  - conservar la componente principal;
  - no deformar machos/hembras.

- [ ] **T3.5** Implementar fallback desde `contour_json`:

  ```python
  mask_from_contour(contour_json, crop_shape, bounding_box_json) -> np.ndarray
  ```

  Este fallback debe quedar marcado en `quality_flags_json`.

**Criterio de aceptacion:** una pieza PNG con alpha produce una mascara `uint8`
binaria con una sola componente principal.

---

## T4: Validacion de Mascara y Contorno

**Objetivo:** Separar piezas perfilables de piezas dudosas antes de extraer
curvas.

- [ ] **T4.1** Implementar:

  ```python
  validate_mask(mask: np.ndarray) -> ProfileQuality
  ```

- [ ] **T4.2** `ProfileQuality` debe incluir:

  - `status`: `valid`, `needs_review`, `invalid`;
  - `flags`: lista de strings;
  - metricas: area, componentes, bbox, toca bordes, etc.

- [ ] **T4.3** Checks minimos:

  - alpha existe;
  - area minima;
  - una componente principal;
  - no toca excesivamente el borde del recorte;
  - contorno exterior extraible.

- [ ] **T4.4** Definir flags:

  ```text
  missing_alpha
  fallback_contour_json
  multiple_components
  touches_crop_border
  contour_not_found
  ```

**Criterio de aceptacion:** una mascara invalida no rompe el pipeline; se guarda
perfil `invalid` o se reporta con flags claros segun se decida.

---

## T5: Extraccion y Normalizacion de Contorno

**Objetivo:** Obtener un contorno exterior ordenado y canonico.

- [ ] **T5.1** Implementar:

  ```python
  extract_outer_contour(mask: np.ndarray) -> np.ndarray
  ```

- [ ] **T5.2** Usar `cv2.findContours(..., cv2.RETR_EXTERNAL, ...)`.
- [ ] **T5.3** Seleccionar la componente/contorno de mayor area.
- [ ] **T5.4** Convertir a array `N x 2`.
- [ ] **T5.5** Implementar:

  ```python
  ensure_clockwise(contour: np.ndarray) -> np.ndarray
  ```

- [ ] **T5.6** Validar el sentido con area firmada.

**Criterio de aceptacion:** el contorno resultante es cerrado conceptualmente,
estable, y siempre queda en sentido horario.

---

## T6: Deteccion de Esquinas Principales

**Objetivo:** Detectar las cuatro esquinas reales de la pieza, no los picos de
machos o valles de hembras.

- [ ] **T6.1** Implementar metodo principal por curvatura:

  ```python
  detect_corner_candidates(contour: np.ndarray) -> list[CornerCandidate]
  ```

- [ ] **T6.2** Suavizar el contorno antes de calcular curvatura.
- [ ] **T6.3** Calcular curvatura local con ventana configurable.
- [ ] **T6.4** Filtrar candidatos convexos.
- [ ] **T6.5** Seleccionar cuatro candidatos maximizando:

  - separacion por distancia sobre contorno;
  - consistencia espacial;
  - no caer en zonas concavas.

- [ ] **T6.6** Implementar fallback con `cv2.approxPolyDP` adaptativo.
- [ ] **T6.7** Implementar:

  ```python
  order_corners_canonical(corners: np.ndarray) -> np.ndarray
  ```

  con orden `1,2,3,4`.

- [ ] **T6.8** Guardar metricas de confianza:

  - `corner_count`;
  - distancia minima entre esquinas;
  - ratio de longitudes de caras;
  - metodo usado: `curvature` o `approx_poly_dp`.

**Criterio de aceptacion:** para piezas limpias se detectan cuatro esquinas y se
pueden visualizar correctamente en el notebook.

---

## T7: Division en Caras

**Objetivo:** Convertir el contorno en cuatro tramos ordenados.

- [ ] **T7.1** Asociar cada esquina canonica al indice mas cercano del contorno.
- [ ] **T7.2** Implementar:

  ```python
  split_contour_into_faces(contour, corner_indices) -> dict[str, np.ndarray]
  ```

- [ ] **T7.3** Las claves deben ser:

  ```text
  1-2
  2-3
  3-4
  4-1
  ```

- [ ] **T7.4** Validar `min_face_points >= 16`.
- [ ] **T7.5** Validar `face_length_ratio` contra mediana.

**Criterio de aceptacion:** las cuatro caras cubren el contorno completo y cada
tramo se puede dibujar de forma independiente.

---

## T8: Curvas Normalizadas por Cara

**Objetivo:** Transformar cada tramo de contorno en una curva comparable.

- [ ] **T8.1** Implementar:

  ```python
  normalize_face_curve(
      face_points: np.ndarray,
      start_corner: np.ndarray,
      end_corner: np.ndarray,
      piece_centroid: np.ndarray,
  ) -> np.ndarray
  ```

- [ ] **T8.2** Proyectar puntos sobre eje esquina-esquina:

  ```text
  X = proyeccion / longitud_cara
  ```

- [ ] **T8.3** Calcular distancia perpendicular:

  ```text
  Y = distancia_perpendicular / longitud_cara
  ```

- [ ] **T8.4** Orientar `Y` para que positivo apunte al exterior:

  - calcular normal candidata;
  - comprobar direccion relativa al centroide;
  - invertir signo si apunta hacia el interior.

- [ ] **T8.5** Ordenar puntos por `X`.
- [ ] **T8.6** Remuestrear a `POINTS_PER_FACE` con interpolacion.
- [ ] **T8.7** Forzar extremos:

  ```text
  punto inicial ~= (0, 0)
  punto final ~= (1, 0)
  ```

**Criterio de aceptacion:** cada cara produce un array de forma `(128, 2)` con
`X` en `[0,1]` y `Y` normalizado por longitud de cara.

---

## T9: Clasificacion de Caras y Tipo de Pieza

**Objetivo:** Clasificar caras y derivar si la pieza es interior, borde o
esquina.

- [ ] **T9.1** Implementar:

  ```python
  classify_face(curve: np.ndarray) -> FaceClassification
  ```

- [ ] **T9.2** Calcular metricas:

  - `max_positive`;
  - `max_negative`;
  - `signed_area`;
  - `peak_x`;
  - `valley_x`;
  - `classification_confidence`.

- [ ] **T9.3** Regla inicial:

  - `lisa`: amplitud absoluta baja y area firmada cercana a cero.
  - `macho`: maximo positivo domina sobre minimo negativo.
  - `hembra`: minimo negativo domina sobre maximo positivo.
  - `desconocida`: confianza baja o curva ambigua.

- [ ] **T9.4** Implementar:

  ```python
  derive_piece_kind(face_types: dict[str, str]) -> str
  ```

  con valores `interior`, `border`, `corner`, `unknown`.

**Criterio de aceptacion:** cada cara queda etiquetada y la pieza recibe un
`piece_kind` coherente.

---

## T10: Descriptor Reducido Fourier

**Objetivo:** Guardar un descriptor compacto sin perder la curva completa.

- [ ] **T10.1** Implementar:

  ```python
  fourier_descriptor(curve: np.ndarray, n_coefficients: int) -> list[float]
  ```

- [ ] **T10.2** Aplicar Fourier sobre la senal `Y(X)` remuestreada.
- [ ] **T10.3** Guardar coeficientes reales/imaginaros de baja frecuencia en
  JSON.
- [ ] **T10.4** Implementar reconstruccion para QA:

  ```python
  reconstruct_curve_from_fourier(descriptor, points_per_face) -> np.ndarray
  ```

- [ ] **T10.5** Calcular `reconstruction_rmse`.
- [ ] **T10.6** Marcar `high_reconstruction_error` si supera umbral.

**Criterio de aceptacion:** el descriptor reducido permite reconstruir una curva
aproximada y reporta error cuantitativo.

---

## T11: Ensamblado del Perfil

**Objetivo:** Orquestar el pipeline por pieza y producir el JSON final.

- [ ] **T11.1** Crear dataclasses o `TypedDict` para:

  - `ProfileQuality`;
  - `FaceProfile`;
  - `PieceProfile`;
  - `FaceClassification`;
  - `CornerCandidate`.

- [ ] **T11.2** Implementar:

  ```python
  build_piece_profile(piece_row: sqlite3.Row) -> PieceProfile
  ```

- [ ] **T11.3** `faces_json` debe contener por cara:

  - `tipo`;
  - `points`;
  - `reduced_descriptor`;
  - `metrics`;
  - `source_contour_indices` si estan disponibles.

- [ ] **T11.4** Guardar:

  - esquinas en crop;
  - esquinas en source;
  - esquinas normalizadas;
  - estado;
  - flags.

**Criterio de aceptacion:** una fila `sam3_piezas` produce una estructura
serializable a JSON y persistible en `pieza_perfiles`.

---

## T12: Script de Ejecucion

**Objetivo:** Poder procesar perfiles desde linea de comandos.

- [ ] **T12.1** Crear script o entrypoint:

  ```text
  scripts/profile_pieces.py
  ```

- [ ] **T12.2** Soportar parametros simples:

  ```powershell
  uv run python scripts/profile_pieces.py --puzzle ciudad --origen piezas_1
  uv run python scripts/profile_pieces.py --puzzle ciudad
  uv run python scripts/profile_pieces.py
  ```

- [ ] **T12.3** Inicializar la tabla si no existe.
- [ ] **T12.4** Procesar todas las piezas SAM 3 que cumplan filtros.
- [ ] **T12.5** Loguear resumen:

  - piezas procesadas;
  - perfiles valid;
  - needs_review;
  - invalid;
  - errores.

**Criterio de aceptacion:** el usuario puede generar perfiles despues de correr
`uv run python -m src.sam3_extractor`.

---

## T13: Notebook de Exploracion

**Objetivo:** Crear una herramienta visual para validar algoritmos antes del
matching.

- [ ] **T13.1** Crear generador:

  ```text
  scripts/generate_piece_profile_notebook.py
  ```

- [ ] **T13.2** Crear notebook:

  ```text
  notebooks/piece_profile_exploration.ipynb
  ```

- [ ] **T13.3** El notebook debe mostrar:

  - PNG de pieza;
  - alpha/mask;
  - contorno exterior;
  - esquinas `1,2,3,4`;
  - caras coloreadas;
  - curva normalizada por cara;
  - tipo de cara;
  - descriptor Fourier reconstruido;
  - error de reconstruccion;
  - flags de calidad.

- [ ] **T13.4** Debe permitir cambiar `pieza_id`, `PUZZLE`, `ORIGEN` y
  `POINTS_PER_FACE`.

**Criterio de aceptacion:** una persona puede abrir el notebook y entender por
que una pieza fue clasificada como interior/borde/esquina.

---

## T14: Pruebas Unitarias

**Objetivo:** Blindar las matematicas del perfil.

- [ ] **T14.1** Tests para alpha:

  - PNG RGBA produce mascara;
  - PNG sin alpha marca fallback/error.

- [ ] **T14.2** Tests para orientacion:

  - contorno antihorario se invierte;
  - contorno horario se conserva.

- [ ] **T14.3** Tests para orden de esquinas:

  - esquinas desordenadas se convierten en `1,2,3,4`.

- [ ] **T14.4** Tests para normalizacion:

  - una cara recta produce `Y ~= 0`;
  - una cara con saliente exterior produce `Y > 0`;
  - una cara con entrada produce `Y < 0`.

- [ ] **T14.5** Tests para remuestreo:

  - salida siempre `(POINTS_PER_FACE, 2)`;
  - `X` monotono;
  - extremos cercanos a `(0,0)` y `(1,0)`.

- [ ] **T14.6** Tests para clasificacion:

  - curva plana => `lisa`;
  - curva positiva => `macho`;
  - curva negativa => `hembra`;
  - curva ambigua => `desconocida`.

- [ ] **T14.7** Tests para DB:

  - tabla se crea;
  - insercion funciona;
  - idempotencia por `(pieza_id, profile_version)`.

**Criterio de aceptacion:** `uv run pytest tests/test_piece_profile*.py -q`
pasa sin depender del modelo SAM 3 real.

---

## T15: Pruebas de Integracion y QA con Datos Reales

**Objetivo:** Validar que el perfil funciona con piezas extraidas reales.

- [ ] **T15.1** Procesar `data/ciudad/piezas_1.jpg` con SAM 3.
- [ ] **T15.2** Ejecutar generacion de perfiles.
- [ ] **T15.3** Verificar que `pieza_perfiles` tiene una fila por pieza SAM 3.
- [ ] **T15.4** Revisar visualmente el notebook para varias piezas:

  - una interior;
  - una de borde;
  - una de esquina si existe;
  - una dudosa.

- [ ] **T15.5** Ajustar umbrales de clasificacion si muchas piezas limpias caen
  en `needs_review`.

**Criterio de aceptacion:** al menos el `90%` de piezas correctamente
segmentadas en una muestra quedan con `profile_status = valid`.

---

## T16: Documentacion

**Objetivo:** Dejar la fase operable por comandos.

- [ ] **T16.1** Actualizar `README.md` con comando corto.
- [ ] **T16.2** Actualizar `docs/scripts_manual.md` con:

  - prerrequisito: ejecutar SAM 3;
  - comando de perfilado;
  - tabla generada;
  - notebook de QA;
  - troubleshooting.

- [ ] **T16.3** Documentar formato de `faces_json` con ejemplo real.

**Criterio de aceptacion:** el usuario puede seguir docs y generar perfiles sin
leer el codigo.

---

## Orden Recomendado de Implementacion

1. T2 Persistencia.
2. T3 Alpha mask.
3. T4 Validacion.
4. T5 Contorno horario.
5. T6 Esquinas.
6. T7 Division en caras.
7. T8 Curvas normalizadas.
8. T9 Clasificacion.
9. T10 Fourier.
10. T11 Ensamblado.
11. T12 Script.
12. T13 Notebook.
13. T14 Tests.
14. T15 QA real.
15. T16 Docs.

El punto mas riesgoso es T6, deteccion de esquinas. Conviene implementarlo con
visualizaciones desde el principio, porque un error ahi contamina todo el perfil.

---

## Comandos Esperados

Generar piezas con SAM 3:

```powershell
uv run python -m src.sam3_extractor
```

Generar perfiles:

```powershell
uv run python scripts/profile_pieces.py --puzzle ciudad --origen piezas_1
```

Generar notebook:

```powershell
uv run python scripts/generate_piece_profile_notebook.py
```

Tests:

```powershell
uv run pytest tests/test_piece_profile.py tests/test_piece_profile_db.py -q
```

---

## Criterios de Finalizacion de la Fase

- Existe tabla `pieza_perfiles`.
- Cada pieza SAM 3 procesable tiene un perfil asociado.
- Cada perfil contiene cuatro caras `1-2`, `2-3`, `3-4`, `4-1`.
- Cada cara tiene `POINTS_PER_FACE` puntos normalizados.
- Cada cara tiene tipo y metricas.
- Cada perfil tiene `piece_kind`, `profile_status` y `quality_flags_json`.
- El notebook permite auditar visualmente una pieza.
- Hay tests unitarios para la geometria principal.
- La documentacion explica como ejecutar el flujo completo.
