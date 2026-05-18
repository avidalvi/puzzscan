# Procesado de Piezas

Este documento describe el flujo completo para convertir imagenes de piezas de
puzzle en perfiles geometricos comparables.

## Resumen

El proceso tiene tres fases principales:

1. **Segmentacion**: detectar piezas en imagenes fuente y exportar cada pieza
   como PNG transparente.
2. **Perfilado geometrico**: extraer mascara, contorno, esquinas, caras y
   curvas normalizadas.
3. **Reduccion dimensional**: representar cada cara con puntos de control
   equiespaciados por longitud de arco.

El objetivo no es resolver el puzzle todavia, sino obtener una representacion
estable de cada pieza para poder comparar encajes en fases posteriores.

## Entradas

Imagenes fuente:

```text
data/<puzzle>/<origen>.jpg
```

Ejemplo:

```text
data/ciudad/piezas_1.jpg
```

Base de datos:

```text
data/puzzscan_v2.db
```

## Fase 1: Segmentacion con SAM 3

Comando:

```powershell
uv run python -m src.sam3_extractor
```

El extractor:

- carga SAM 3;
- usa el prompt `SAM3_TEXT_PROMPT`, por defecto `puzzle piece`;
- detecta mascaras de piezas;
- ordena las detecciones de izquierda a derecha y de arriba a abajo;
- guarda cada pieza como PNG con canal alpha;
- inserta metadatos en `sam3_piezas`;
- crea una imagen debug numerada.

Salidas principales:

```text
output/<puzzle>/piezas_sam3/<origen>/pieza_<n>.png
output/<puzzle>/piezas_sam3/<origen>_debug.jpg
data/puzzscan_v2.db, table sam3_piezas
```

Campos importantes de `sam3_piezas`:

- `id`: identificador de base de datos.
- `nombre_puzzle`: nombre del puzzle.
- `origen`: imagen fuente, sin extension.
- `piece_index`: numero estable dentro de la imagen, ordenado espacialmente.
- `ruta_imagen`: PNG recortado con alpha.
- `contour_json`: contorno original de SAM 3 en coordenadas fuente.
- `bounding_box_json`: caja de la deteccion.
- `confidence_score`: confianza del modelo.

## Fase 2: Mascara de Perfil

El perfil geometrico usa el canal alpha del PNG recortado como fuente principal:

```python
mask = alpha > 0
```

No se aplica limpieza morfologica por defecto. La mascara de proceso debe
corresponder a lo que se ve en el alpha, porque el canal alpha resulto mas fiel
que una mascara binaria limpiada agresivamente.

Si el PNG no tiene alpha, el sistema puede reconstruir una mascara aproximada
desde `contour_json` y `bounding_box_json`, marcando flags de calidad.

## Fase 3: Contorno Exterior

Desde la mascara se extrae el contorno exterior mas grande:

```text
mask -> extract_outer_contour -> raw_contour
```

El contorno se fuerza a sentido horario para que el orden de caras sea estable.

Despues se convierte en una curva continua re-muestreada por longitud de arco.
Esto reduce la dependencia de la densidad irregular de puntos que produce el
raster.

## Fase 4: Suavizado del Contorno

Las piezas reales tienen bordes redondeados, pero el contorno de una mascara
pixelada contiene dientes y pequenos picos. Para reducir esos artefactos se
aplica un suavizado geometrico tipo Chaikin y se re-muestrea de nuevo por
longitud de arco.

El objetivo del suavizado es:

- quitar escalones de pixelado;
- conservar la forma global de machos y hembras;
- no desplazar demasiado el borde real;
- mantener una curva abierta/cerrada estable para detectar esquinas.

En el notebook se muestra el **smoothing RMSE**, que mide la distancia media
entre la curva continua original y la curva suavizada, en pixeles. Esta metrica
ayuda a ajustar el suavizado sin depender solo de inspeccion visual.

## Fase 5: Deteccion de Esquinas

Sobre el contorno suavizado se detectan las cuatro esquinas principales.

El pipeline actual:

- calcula curvatura local;
- busca candidatos convexos;
- selecciona cuatro candidatos separados a lo largo del contorno;
- usa `approxPolyDP` como fallback si la curvatura falla;
- ordena las esquinas como:

```text
1 = top-left
2 = top-right
3 = bottom-right
4 = bottom-left
```

Esta fase es critica. Los descriptores de cara dependen de las esquinas:

- definen donde empieza y termina cada cara;
- definen el eje base de normalizacion;
- definen la escala de la curva.

Por eso las futuras mejoras deberian prestar especial atencion al refinamiento
de esquinas.

## Fase 6: Division en Caras

Con las cuatro esquinas, el contorno se divide en cuatro caras:

```text
1-2
2-3
3-4
4-1
```

Cada cara conserva el orden real de puntos del contorno.

## Fase 7: Normalizacion de Caras

Cada cara se transforma a un sistema local:

- la esquina inicial pasa a `(0, 0)`;
- la esquina final pasa a `(1, 0)`;
- la distancia entre esquinas define la escala;
- el eje Y es perpendicular a la linea entre esquinas;
- Y positivo apunta hacia fuera de la pieza.

Importante: la cara normalizada es una **curva parametrica abierta**, no una
funcion `Y=f(X)`.

Esto es necesario porque una cara macho o hembra puede tener retrocesos en X.
Forzar `Y=f(X)` introduce picos y artefactos porque para una misma X puede
haber varios valores de Y.

## Fase 8: Clasificacion de Caras

Cada curva normalizada se clasifica como:

- `lisa`: poca amplitud respecto a la linea base;
- `macho`: saliente dominante hacia fuera;
- `hembra`: hueco dominante hacia dentro;
- `desconocida`: geometria ambigua.

La clasificacion usa amplitud, area firmada y relacion entre maximos positivos
y negativos.

Con las cuatro caras se deriva tambien el tipo de pieza:

- `interior`: sin caras lisas;
- `border`: una cara lisa;
- `corner`: dos caras lisas consecutivas;
- `unknown`: combinacion no concluyente.

## Fase 9: Descriptor Reducido

La reduccion dimensional actual usa puntos de control.

Cada cara normalizada se re-muestrea por longitud de arco y se guarda como:

```text
36 puntos (x, y) = 72 floats por cara
```

Ventajas:

- no asume periodicidad;
- conserva endpoints;
- preserva retrocesos en X;
- es facil de visualizar y depurar;
- permite comparar dos caras punto a punto;
- evita los artefactos de Fourier en los limites de curvas abiertas.

Se descarto Fourier como descriptor principal para caras abiertas porque la FFT
asume periodicidad y tiende a cerrar artificialmente la curva entre el final y
el inicio. Ese cierre introduce artefactos en los limites.

## Fase 10: Persistencia

Comando:

```powershell
uv run python scripts/profile_pieces.py --puzzle ciudad --origen piezas_1
```

Salida:

```text
data/puzzscan_v2.db, table pieza_perfiles
```

La tabla `pieza_perfiles` almacena:

- identificacion de pieza;
- version del perfil;
- puntos de esquinas en coordenadas de crop, fuente y normalizadas;
- caras normalizadas;
- descriptores reducidos;
- tipo de pieza;
- estado del perfil;
- flags de calidad.

El perfil versionado actual es:

```text
piece_profile_v1
```

Antes de insertar un perfil se elimina el anterior para el mismo
`pieza_id + profile_version`, de modo que la generacion es idempotente.

## Notebook de QA

Comando:

```powershell
uv run python scripts/generate_piece_profile_notebook.py
```

Salida:

```text
notebooks/piece_profile_exploration.ipynb
```

El notebook selecciona cinco piezas representativas por cuantiles de area. No
usa seleccion aleatoria. Esto permite revisar piezas pequenas, medianas y
grandes en el mismo recorrido.

El notebook muestra:

- RGB, alpha y mascara de proceso;
- metricas de calidad de mascara;
- contorno raw frente a contorno suavizado;
- RMSE de suavizado;
- esquinas detectadas;
- caras separadas;
- curvas normalizadas parametricas;
- puntos de control y reconstruccion;
- RMSE de muestreo;
- resumen final del perfil.

Si se modifica `src.piece_profile`, hay que reiniciar el kernel del notebook
antes de volver a ejecutar celdas, porque Jupyter puede mantener modulos
cacheados.

## Metricas de Calidad

### Smoothing RMSE

Mide el error medio entre la curva continua original y la curva suavizada, en
pixeles.

Sirve para comprobar que el suavizado elimina ruido sin desplazar la forma real
en exceso.

### Sampling RMSE

Mide el error medio entre la curva normalizada completa y la reconstruccion a
partir de sus puntos de control.

Sirve para comprobar si 36 puntos por cara son suficientes para representar la
geometria.

## Limitaciones Conocidas

- Las esquinas siguen siendo el punto mas sensible del pipeline.
- El suavizado aun puede requerir ajuste segun calidad de segmentacion.
- La clasificacion `macho`/`hembra` es heuristica.
- El descriptor de 36 puntos no resuelve aun el matching final entre piezas.
- No existe todavia correccion manual interactiva de esquinas o perfiles.

## Comandos Principales

```powershell
# Segmentacion SAM 3
uv run python -m src.sam3_extractor

# Perfilado geometrico
uv run python scripts/profile_pieces.py --puzzle ciudad --origen piezas_1

# Notebook de exploracion de perfiles
uv run python scripts/generate_piece_profile_notebook.py

# Tests de perfil
uv run pytest tests/test_piece_profile.py tests/test_piece_profile_db.py -q
```
