# Plan Tecnico: Motor de Busqueda y Coincidencia de Piezas

> **PRD de referencia:** `docs/PRD/PRD_motor_busqueda.md`
> **Fecha:** 2026-05-18

---

## Resumen

Implementar la fase 3 del proyecto: un motor local de busqueda que, dada una
pieza central ya perfilada, encuentre candidatas compatibles para sus cuatro
caras originales. La primera entrega se desarrolla en notebook para validar
visualmente el proceso antes de persistir rankings o automatizar la resolucion
global.

El motor debe:

- cargar perfiles desde `pieza_perfiles`;
- filtrar candidatas por compatibilidad `macho`/`hembra`/`lisa`;
- comparar curvas parametricas normalizadas usando RMSE euclideo;
- evaluar DTW solo como diagnostico, no como score principal;
- calcular senales complementarias de luminancia/color/textura;
- mostrar top 3 candidatas por direccion;
- visualizar una composicion local `3x3`.

---

## Decisiones Tecnicas Iniciales

- Modulo principal: `src/piece_search.py`.
- Notebook generado: `notebooks/piece_search_exploration.ipynb`.
- Generador: `scripts/generate_piece_search_notebook.py`.
- Tests: `tests/test_piece_search.py`.
- Fuente de perfiles: `data/puzzscan_v2.db`, tabla `pieza_perfiles`.
- Fuente de imagenes: `sam3_piezas.ruta_imagen`.
- Score geometrico principal: RMSE euclideo punto a punto entre descriptores de
  36 puntos `(x, y)`.
- DTW: diagnostico opcional con ventana estrecha, fuera del score principal.
- Color/luminancia/textura: senales complementarias visibles en notebook.
- Sin persistencia de resultados en la primera iteracion.

---

## Estructura Propuesta

```text
src/piece_search.py
scripts/generate_piece_search_notebook.py
notebooks/piece_search_exploration.ipynb
tests/test_piece_search.py
docs/procesado_piezas.md
docs/scripts_manual.md
```

No modificar el significado de `pieza_perfiles`; esta fase lo consume.

---

## Fase 0: Preparacion y Datos de Prueba

**Objetivo:** asegurar que hay perfiles suficientes para probar el motor.

- [ ] **T0.1** Confirmar que existe `pieza_perfiles` con perfiles de
  `piece_profile_v2`.
- [ ] **T0.2** Definir pieza central por defecto en notebook:

  ```python
  CENTER_PIEZA_ID = 836
  ```

- [ ] **T0.3** Si la pieza no existe, elegir automaticamente una pieza
  `needs_review` o `valid` con cuatro caras no desconocidas.
- [ ] **T0.4** Documentar en el notebook cuantos perfiles hay por estado:

  ```text
  valid
  needs_review
  invalid
  ```

**Validacion manual:** el notebook muestra un resumen de perfiles disponibles y
selecciona una pieza central reproducible.

---

## Fase 1: Modelo de Datos en Memoria

**Objetivo:** cargar perfiles e imagenes en estructuras tipadas y faciles de
usar en matching.

- [ ] **T1.1** Crear `src/piece_search.py`.
- [ ] **T1.2** Definir dataclasses:

  ```python
  SearchFace
  SearchPiece
  FaceConstraint
  CandidateScore
  DirectionMatch
  ```

- [ ] **T1.3** Implementar `load_profiled_pieces(...)`:

  ```python
  load_profiled_pieces(
      puzzle: str | None = None,
      origen: str | None = None,
      profile_version: str = "piece_profile_v2",
      allow_needs_review: bool = True,
  ) -> list[SearchPiece]
  ```

- [ ] **T1.4** Parsear `faces_json` y exponer por cara:

  ```text
  face_key
  face_type
  points
  reduced_descriptor
  metrics
  ```

- [ ] **T1.5** Unir `pieza_perfiles` con `sam3_piezas` para obtener
  `ruta_imagen`, `estado`, `piece_index`, `origen`.
- [ ] **T1.6** Excluir perfiles `invalid`.
- [ ] **T1.7** Marcar, no ocultar, perfiles `needs_review`.

### Tests

- [ ] **test_load_profiled_pieces_empty_db**: devuelve lista vacia.
- [ ] **test_load_profiled_pieces_filters_profile_version**.
- [ ] **test_load_profiled_pieces_excludes_invalid**.
- [ ] **test_search_piece_faces_are_parsed**.

**Criterio de aceptacion:** se puede cargar una lista de piezas perfiladas con
caras, descriptores e imagen asociada.

---

## Fase 2: Convenciones de Direccion y Caras

**Objetivo:** fijar la correspondencia entre caras `1-2`, `2-3`, `3-4`, `4-1`
y direcciones cardinales.

- [ ] **T2.1** Definir tabla inicial:

  ```python
  DIRECTION_TO_CENTER_FACE = {
      "N": "1-2",
      "E": "2-3",
      "S": "3-4",
      "W": "4-1",
  }
  ```

- [ ] **T2.2** Definir cara esperada de la candidata para cada direccion:

  ```python
  OPPOSITE_DIRECTION = {
      "N": "S",
      "E": "W",
      "S": "N",
      "W": "E",
  }
  ```

- [ ] **T2.3** Crear helper `face_for_direction(piece, direction)`.
- [ ] **T2.4** Mostrar la tabla en el notebook con dibujo de referencia.
- [ ] **T2.5** Anadir advertencia: esta correspondencia debe validarse
  visualmente.

### Tests

- [ ] **test_direction_face_mapping_complete**.
- [ ] **test_opposite_direction_mapping**.
- [ ] **test_face_for_direction_returns_expected_face**.

**Criterio de aceptacion:** el notebook deja claro que cara se compara en cada
direccion.

---

## Fase 3: Filtro Topologico

**Objetivo:** reducir candidatas por tipo de cara antes de medir geometria.

- [ ] **T3.1** Implementar compatibilidad:

  ```text
  macho  <-> hembra
  hembra <-> macho
  lisa   <-> lisa
  desconocida -> permitido solo si ALLOW_UNKNOWN_FACE=True, con penalizacion
  ```

- [ ] **T3.2** Implementar:

  ```python
  compatible_face_types(a: str, b: str, allow_unknown: bool) -> bool
  ```

- [ ] **T3.3** Implementar:

  ```python
  filter_candidates_by_face_type(center_piece, candidates, direction)
  ```

- [ ] **T3.4** Reportar conteos:

  ```text
  total candidates
  after status filter
  after face type filter
  after quality filter
  ```

### Tests

- [ ] **test_macho_matches_hembra**.
- [ ] **test_lisa_matches_lisa**.
- [ ] **test_macho_does_not_match_macho**.
- [ ] **test_unknown_allowed_with_flag**.
- [ ] **test_unknown_rejected_without_flag**.

**Criterio de aceptacion:** para cada una de las cuatro caras de la pieza
central se muestran candidatas supervivientes por tipo.

---

## Fase 4: Curva Complementaria y Distancia Geometrica

**Objetivo:** comparar dos caras compatibles en el mismo sistema normalizado.

- [ ] **T4.1** Implementar inversion de curva:

  ```python
  invert_face_curve(points: np.ndarray) -> np.ndarray
  ```

  Debe:

  - invertir orden de puntos si corresponde;
  - mapear endpoints a `(0,0)` y `(1,0)`;
  - invertir signo de Y para representar la cara complementaria;
  - mantener curva parametrica, sin forzar `Y=f(X)`.

- [ ] **T4.2** Implementar distancia RMSE:

  ```python
  geometry_rmse(a: np.ndarray, b: np.ndarray) -> float
  ```

- [ ] **T4.3** Asegurar que ambos descriptores usan 36 puntos de control.
- [ ] **T4.4** Implementar scoring por una restriccion:

  ```python
  score_face_pair(center_face, candidate_face) -> CandidateScore
  ```

- [ ] **T4.5** Guardar en score:

  ```text
  center_face_key
  candidate_face_key
  center_face_type
  candidate_face_type
  geometry_rmse
  penalties
  ```

### Tests

- [ ] **test_invert_face_curve_preserves_shape_length**.
- [ ] **test_invert_face_curve_flips_y**.
- [ ] **test_geometry_rmse_zero_for_identical_complement**.
- [ ] **test_geometry_rmse_increases_for_shifted_curve**.
- [ ] **test_score_face_pair_contains_metadata**.

**Criterio de aceptacion:** dos curvas complementarias generan RMSE bajo y una
curva incorrecta genera RMSE mayor.

---

## Fase 5: Evaluacion de DTW como Diagnostico

**Objetivo:** comprobar si DTW aporta valor o si reduce la discriminacion.

- [ ] **T5.1** Implementar DTW sin dependencias externas:

  ```python
  constrained_dtw_distance(
      a: np.ndarray,
      b: np.ndarray,
      window_ratio: float = 0.05,
  ) -> float
  ```

- [ ] **T5.2** La ventana debe estar limitada:

  ```text
  5-10% de longitud del descriptor
  ```

- [ ] **T5.3** No usar DTW en `score_total` por defecto.
- [ ] **T5.4** Mostrar `geometry_dtw` en notebook cuando:

  ```python
  COMPUTE_DTW = True
  ```

- [ ] **T5.5** Crear diagnostico:

  ```text
  DTW separa top buenos de falsos positivos?
  DTW empata demasiadas hembras para un mismo macho?
  ```

### Tests

- [ ] **test_constrained_dtw_identical_zero**.
- [ ] **test_constrained_dtw_window_limits_alignment**.
- [ ] **test_dtw_not_used_in_default_score**.
- [ ] **test_dtw_can_be_reported_as_diagnostic**.

**Criterio de aceptacion:** DTW se puede calcular y visualizar, pero el ranking
principal no depende de DTW.

---

## Fase 6: Ranking de Candidatas

**Objetivo:** producir top 3 por cada direccion cardinal de la pieza central.

- [ ] **T6.1** Implementar:

  ```python
  rank_candidates_for_direction(
      center_piece: SearchPiece,
      candidates: list[SearchPiece],
      direction: str,
      top_k: int = 3,
  ) -> list[CandidateScore]
  ```

- [ ] **T6.2** Excluir la propia pieza central.
- [ ] **T6.3** Excluir candidatas `estado != AVAILABLE`, salvo modo debug.
- [ ] **T6.4** Aplicar filtro topologico.
- [ ] **T6.5** Calcular RMSE geometrico.
- [ ] **T6.6** Aplicar penalizaciones:

  ```text
  needs_review
  unknown_face
  high_sampling_rmse
  quality_flags
  ```

- [ ] **T6.7** Ordenar por `score_total` ascendente.
- [ ] **T6.8** Implementar:

  ```python
  rank_all_cardinal_directions(center_piece, candidates) -> dict[str, list[CandidateScore]]
  ```

### Tests

- [ ] **test_rank_excludes_center_piece**.
- [ ] **test_rank_orders_by_geometry_rmse**.
- [ ] **test_rank_applies_needs_review_penalty**.
- [ ] **test_rank_returns_top_k**.
- [ ] **test_rank_all_cardinal_directions_has_four_keys**.

**Criterio de aceptacion:** para una pieza central se obtienen cuatro rankings
independientes `N/E/S/W`.

---

## Fase 7: Luminancia, Color y Textura

**Objetivo:** calcular senales visuales complementarias de borde para ranking y
depuracion.

- [ ] **T7.1** Implementar extraccion de banda interior de cara:

  ```python
  sample_face_band(
      image_rgba: np.ndarray,
      face_points_crop: np.ndarray,
      inward_normal: np.ndarray,
      width_px: int = 8,
  ) -> np.ndarray
  ```

- [ ] **T7.2** Implementar perfil de luminancia:

  ```python
  luminance_profile(band: np.ndarray, n_points: int = 36) -> np.ndarray
  ```

- [ ] **T7.3** Implementar perfil de color en Lab:

  ```python
  color_profile_lab(band: np.ndarray, n_points: int = 36) -> np.ndarray
  ```

- [ ] **T7.4** Implementar textura simple:

  ```python
  texture_profile_gradient(band: np.ndarray, n_points: int = 36) -> np.ndarray
  ```

- [ ] **T7.5** Implementar distancias:

  ```python
  luminance_distance(a, b)
  color_distance_lab(a, b)
  texture_distance(a, b)
  ```

- [ ] **T7.6** Integrar senales en `CandidateScore`, separadas del score
  geometrico.
- [ ] **T7.7** Score combinado inicial:

  ```text
  score_total =
      1.00 * geometry_rmse
    + 0.20 * luminance_distance
    + 0.20 * color_distance
    + 0.10 * texture_distance
    + penalties
  ```

### Tests

- [ ] **test_luminance_profile_shape**.
- [ ] **test_color_profile_lab_shape**.
- [ ] **test_texture_profile_gradient_shape**.
- [ ] **test_visual_distances_zero_for_identical_profiles**.
- [ ] **test_score_keeps_visual_components_separate**.

**Criterio de aceptacion:** el notebook muestra distancia geometrica,
luminancia, color y textura por candidata.

---

## Fase 8: Notebook de Exploracion

**Objetivo:** validar el motor completo visualmente antes de automatizarlo.

- [ ] **T8.1** Crear `scripts/generate_piece_search_notebook.py`.
- [ ] **T8.2** Crear `notebooks/piece_search_exploration.ipynb`.
- [ ] **T8.3** Primera celda robusta:

  ```python
  if Path.cwd().name == "notebooks":
      os.chdir("..")
  sys.path.insert(0, str(Path.cwd().resolve()))
  ```

- [ ] **T8.4** Controles:

  ```python
  PUZZLE = "ciudad"
  ORIGEN = None
  CENTER_PIEZA_ID = 836
  PROFILE_VERSION = "piece_profile_v2"
  TOP_K = 3
  COMPUTE_DTW = False
  ALLOW_NEEDS_REVIEW = True
  ALLOW_UNKNOWN_FACE = True
  ```

- [ ] **T8.5** Celdas:

  1. Cargar perfiles.
  2. Seleccionar pieza central.
  3. Mostrar caras de pieza central.
  4. Mostrar tabla direccion/cara.
  5. Ranking para `N/E/S/W`.
  6. Curvas superpuestas para top 3.
  7. Senales de luminancia/color/textura.
  8. Central + mejor candidata por direccion.
  9. Vista `3x3`.
  10. Resumen de riesgos/calidad.

- [ ] **T8.6** Mostrar conteos de filtros por direccion.
- [ ] **T8.7** Mostrar candidatas no encontradas o descartadas por calidad.
- [ ] **T8.8** No modificar estado persistente.

### Validacion Notebook

- [ ] Compilar celdas con `ast.parse` ignorando magics.
- [ ] Ejecutar manualmente con `CENTER_PIEZA_ID=836`.
- [ ] Confirmar que se ven rankings para cuatro caras.
- [ ] Confirmar que la vista `3x3` tolera huecos vacios.

**Criterio de aceptacion:** el notebook permite revisar el motor local de
matching sin escribir en base de datos.

---

## Fase 9: Visualizacion de Ensamblado

**Objetivo:** dibujar piezas candidatas en contexto de forma entendible.

- [ ] **T9.1** Implementar carga de PNG RGBA.
- [ ] **T9.2** Dibujar pieza central con alpha.
- [ ] **T9.3** Dibujar candidata orientada junto a la cara comparada.
- [ ] **T9.4** En primera iteracion, permitir aproximacion visual sin
  transformacion fisica perfecta.
- [ ] **T9.5** Dibujar las curvas comparadas sobre las piezas.
- [ ] **T9.6** Implementar layout `3x3`:

  ```text
  NW  N  NE
  W   C  E
  SW  S  SE
  ```

- [ ] **T9.7** Mostrar placeholders para posiciones sin candidata.

### Tests

- [ ] **test_layout_3x3_contains_center**.
- [ ] **test_layout_3x3_allows_missing_candidates**.
- [ ] **test_rgba_loader_preserves_alpha**.

**Criterio de aceptacion:** las visualizaciones permiten inspeccion humana del
ranking.

---

## Fase 10: Validacion Cuantitativa

**Objetivo:** medir si el ranking discrimina lo suficiente.

- [ ] **T10.1** Crear notebook/tabla de diagnostico:

  ```text
  center_piece
  direction
  top_k
  geometry_rmse
  geometry_dtw
  luminance_distance
  color_distance
  texture_distance
  score_total
  ```

- [ ] **T10.2** Comparar top 1 vs top 2/top 3:

  ```text
  margin = score_2 - score_1
  ```

- [ ] **T10.3** Detectar ranking poco discriminativo:

  ```text
  margin < threshold
  ```

- [ ] **T10.4** Evaluar DTW:

  - mejora matches visualmente buenos?
  - empata demasiadas hembras?
  - reduce separacion entre top y falsos positivos?

- [ ] **T10.5** Marcar resultados como:

  ```text
  confident
  ambiguous
  no_candidate
  low_quality
  ```

### Tests

- [ ] **test_score_margin_computation**.
- [ ] **test_ambiguous_when_margin_low**.
- [ ] **test_no_candidate_status**.

**Criterio de aceptacion:** el notebook no solo muestra rankings, tambien indica
cuando una decision es ambigua.

---

## Fase 11: Documentacion

**Objetivo:** dejar el flujo explicable y reproducible.

- [ ] **T11.1** Actualizar `docs/procesado_piezas.md` con una seccion de motor
  de busqueda.
- [ ] **T11.2** Actualizar `docs/scripts_manual.md` con:

  ```powershell
  uv run python scripts/generate_piece_search_notebook.py
  ```

- [ ] **T11.3** Actualizar `README.md` con enlace al PRD/plan si procede.
- [ ] **T11.4** Documentar decision RMSE vs DTW.
- [ ] **T11.5** Documentar que color/luminancia/textura son senales
  complementarias.

**Criterio de aceptacion:** una persona puede reproducir el notebook desde cero
leyendo la documentacion.

---

## Fase 12: Checks y Cierre

**Objetivo:** asegurar calidad antes de commit.

- [ ] **T12.1** Ejecutar tests nuevos:

  ```powershell
  uv run pytest tests/test_piece_search.py -q
  ```

- [ ] **T12.2** Ejecutar tests relacionados:

  ```powershell
  uv run pytest tests/test_piece_profile.py tests/test_piece_profile_db.py -q
  ```

- [ ] **T12.3** Ejecutar lint:

  ```powershell
  uv run ruff check src tests scripts
  ```

- [ ] **T12.4** Ejecutar format:

  ```powershell
  uv run ruff format src tests scripts
  ```

- [ ] **T12.5** Ejecutar mypy si se tocaron modulos Python:

  ```powershell
  uv run mypy src tests scripts
  ```

- [ ] **T12.6** Regenerar notebook:

  ```powershell
  uv run python scripts/generate_piece_search_notebook.py
  ```

- [ ] **T12.7** Compilar celdas de notebook con `ast.parse`.
- [ ] **T12.8** Ejecutar pre-commit antes del commit:

  ```powershell
  uv run pre-commit run --all-files
  ```

**Criterio de aceptacion:** hooks y tests relevantes pasan.

---

## Orden de Implementacion Recomendado

1. Fase 1: carga de perfiles.
2. Fase 2: direcciones y caras.
3. Fase 3: filtro topologico.
4. Fase 4: RMSE geometrico.
5. Fase 6: ranking top 3.
6. Fase 8: notebook minimo.
7. Fase 9: visualizacion local.
8. Fase 7: luminancia/color/textura.
9. Fase 5: diagnostico DTW.
10. Fase 10: validacion cuantitativa.
11. Fase 11: documentacion.
12. Fase 12: checks y cierre.

---

## Riesgos Principales

- **Esquinas inestables:** pueden dominar el error geometrico.
- **Direccion/cara mal mapeada:** puede invalidar rankings completos.
- **DTW demasiado permisivo:** puede hacer que muchas hembras parezcan buenas.
- **Color enganoso:** piezas de zonas parecidas pueden tener color similar pero
  geometria incorrecta.
- **Perfiles `needs_review`:** utiles para explorar, peligrosos para decisiones
  automaticas.
- **Visualizacion ensamblada aproximada:** no debe confundirse con colocacion
  fisica exacta.

---

## Entregables

- `src/piece_search.py`
- `tests/test_piece_search.py`
- `scripts/generate_piece_search_notebook.py`
- `notebooks/piece_search_exploration.ipynb`
- Actualizaciones en documentacion
- Notebook validado con una pieza central fija y rankings para `N/E/S/W`
