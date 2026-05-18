# PRD: Motor de Busqueda y Coincidencia de Piezas

## 1. Vision y Objetivo

Esta fase construye el primer motor de busqueda geometrica entre piezas ya
perfiladas. El objetivo es seleccionar una pieza central, buscar candidatas que
puedan encajar en sus posiciones vecinas y visualizar el resultado de forma
controlada en notebook antes de automatizar el proceso.

La fase parte de los perfiles generados en `pieza_perfiles`, que describen cada
cara como una curva parametrica normalizada y un descriptor reducido de puntos
de control. La busqueda debe usar primero restricciones topologicas simples
(`macho`/`hembra`/`lisa`) y despues medir distancia geometrica entre curvas
compatibles. Ademas, el motor debe preparar el uso de luminancia, color y
textura como informacion complementaria de ranking.

## 2. Problema que Resuelve

Tras extraer y perfilar piezas, el sistema todavia no sabe que piezas pueden
ser vecinas. Esta fase responde preguntas como:

- que piezas candidatas tienen una cara compatible con una cara dada;
- cuantas piezas sobreviven al filtro de tipo de cara;
- que candidata tiene la curva inversa mas parecida;
- que candidata tambien es compatible en luminancia, color y textura cerca del
  borde;
- como se verian ensambladas la pieza central y sus vecinas;
- que ocurre si una vecina no existe en la base de datos o tiene perfil de mala
  calidad.

El foco inicial no es resolver todo el puzzle, sino validar un motor local de
coincidencia alrededor de una pieza.

## 3. Alcance

### Incluido

- Crear un notebook de exploracion del motor de busqueda.
- Seleccionar una pieza central fija configurable.
- Permitir seleccion aleatoria posterior como modo opcional.
- Buscar candidatos para posiciones vecinas alrededor de la pieza central.
- Empezar con una vecindad de una pieza central y avanzar hacia una ventana
  `3x3` de hasta nueve piezas.
- Filtrar piezas candidatas por compatibilidad `macho`/`hembra`/`lisa`.
- Aplicar una, dos, tres o cuatro restricciones segun las caras vecinas ya
  conocidas.
- Comparar curvas inversas mediante una distancia geometrica.
- Calcular senales complementarias de luminancia, color y textura de borde.
- Calcular ranking de candidatas por posicion vecina.
- Mostrar las tres candidatas mas probables por busqueda local.
- Dibujar la pieza central y la mejor candidata ensambladas.
- Dibujar una composicion `3x3` con la mejor candidata encontrada en cada
  posicion disponible.
- Manejar posiciones sin candidata por falta de datos o baja calidad.

### Fuera de Alcance

- Resolver el puzzle completo.
- Detectar huecos reales en una foto del tablero.
- Corregir manualmente esquinas o perfiles.
- Actualizar estado `PLACED`/`AVAILABLE` de forma persistente.
- Generar una interfaz final de usuario.

## 4. Entradas

La fase consume piezas ya segmentadas y perfiladas:

```text
data/puzzscan_v2.db
```

Tablas requeridas:

```text
sam3_piezas
pieza_perfiles
```

Campos necesarios de `sam3_piezas`:

- `id`
- `nombre_puzzle`
- `origen`
- `piece_index`
- `ruta_imagen`
- `estado`

Campos necesarios de `pieza_perfiles`:

- `pieza_id`
- `profile_version`
- `piece_kind`
- `profile_status`
- `faces_json`
- `quality_flags_json`

Cada entrada de `faces_json` debe contener, por cara:

- tipo de cara: `lisa`, `macho`, `hembra` o `desconocida`;
- curva normalizada;
- descriptor reducido de 36 puntos de control;
- metricas de clasificacion y reconstruccion.

Para matching visual se necesitara extraer informacion adicional desde el PNG
de la pieza:

- perfil de luminancia junto a cada cara;
- perfil de color medio o mediano en una banda interior paralela a la cara;
- descriptores simples de textura en la banda de borde.

En la primera iteracion esta informacion puede calcularse en memoria dentro del
notebook sin persistencia obligatoria.

## 5. Modelo de Vecindad

La pieza central se coloca en el centro de una ventana `3x3`:

```text
NW  N  NE
W   C  E
SW  S  SE
```

La busqueda inicial de candidatos se ejecuta para las cuatro caras originales de
la pieza central. Es decir, para una pieza fija `C`, se calculan rankings
independientes para:

```text
N, E, S, W
```

Estas cuatro busquedas corresponden a las cuatro caras del perfil:

```text
1-2, 2-3, 3-4, 4-1
```

Las diagonales se podran usar despues para visualizar contexto o para completar
una composicion parcial, pero el encaje directo ocurre entre caras compartidas:

- `N` encaja con la cara superior de `C`;
- `E` encaja con la cara derecha de `C`;
- `S` encaja con la cara inferior de `C`;
- `W` encaja con la cara izquierda de `C`.

La convencion inicial de caras procede del perfil:

```text
1-2, 2-3, 3-4, 4-1
```

La implementacion debe definir explicitamente que cara corresponde a cada
direccion cardinal para evitar ambiguedad. Esa tabla de correspondencia debe
mostrarse en el notebook.

## 6. Restricciones Topologicas

Antes de medir curvas se filtra por tipo de cara:

```text
macho  <-> hembra
hembra <-> macho
lisa   <-> lisa
```

Casos no concluyentes:

- `desconocida` no debe descartarse necesariamente en modo exploracion, pero
  debe penalizarse.
- perfiles `invalid` se excluyen.
- perfiles `needs_review` se permiten, marcados con advertencia.

Una posicion candidata puede tener varias restricciones. Por ejemplo:

- En una busqueda simple contra la pieza central hay una restriccion.
- En una posicion de borde de una composicion parcial puede haber dos
  restricciones.
- En posiciones mas cerradas puede haber tres o cuatro restricciones.

El notebook debe mostrar cuantas candidatas quedan despues de cada restriccion.

## 7. Comparacion Geometrica

### 7.1. Curva Inversa

Para comparar dos caras que encajan, una curva debe invertirse para representar
la geometria complementaria.

La comparacion debe considerar:

- invertir el orden de puntos cuando corresponda;
- cambiar el signo de la componente perpendicular;
- alinear endpoints normalizados;
- mantener la curva parametrica, sin forzar `Y=f(X)`.

### 7.2. Distancia Geometrica

La primera distancia recomendada es RMSE euclidea punto a punto entre
descriptores de control:

```text
distance = sqrt(mean(||A_i - B_i||^2))
```

Donde `A_i` y `B_i` son puntos `(x, y)` de las curvas normalizadas y
compatibilizadas.

La distancia debe reportarse por cara y agregarse por candidata:

```text
score_geom = media ponderada de distancias por restriccion
```

Valores mas bajos indican mejor encaje.

### 7.3. Euclidea vs DTW

La distancia euclidea punto a punto es el baseline preferido para la primera
iteracion porque:

- los descriptores tienen el mismo numero de puntos;
- los puntos estan equiespaciados por longitud de arco;
- las curvas ya estan normalizadas entre las dos esquinas;
- el resultado es simple, rapido y facil de depurar visualmente.

DTW debe tratarse con mucho cuidado. Aunque puede tolerar pequenos desfases
causados por:

- esquinas detectadas con unos pixeles de error;
- distinto reparto local de puntos en un macho/hembra;
- pequenas diferencias de suavizado;
- deformaciones locales del contorno.

Tambien puede ser inutil o incluso perjudicial para este problema porque:

- puede sobre-alinear curvas que no deberian encajar;
- puede hacer que un macho parezca compatible con muchas hembras genericas;
- puede ocultar diferencias locales importantes entre tabs/huecos;
- puede ocultar errores de esquina o de normalizacion;
- es mas dificil de interpretar;
- requiere restricciones de ventana para no permitir deformaciones excesivas;
- puede reducir demasiado la discriminacion entre candidatas.

Decision inicial:

- usar RMSE euclidea como score principal;
- no usar DTW en el score principal;
- calcular DTW solo como diagnostico opcional en notebook;
- evaluar empiricamente si DTW aporta algo sobre RMSE;
- descartar DTW si ordena muchas hembras como buenas candidatas para un mismo
  macho sin discriminacion clara.

Si se prueba DTW, debe usarse con restricciones fuertes:

```text
dtw_window <= 5-10% de la longitud del descriptor
dtw_distance_normalized = coste_dtw / numero_de_puntos
```

Tambien se debe comparar contra un baseline negativo:

```text
DTW util si:
- mejora el ranking de matches visualmente buenos;
- no reduce la separacion entre top buenos y falsos positivos;
- no convierte muchas hembras en candidatas casi equivalentes para el mismo macho.
```

El notebook debe mostrar ambos valores cuando DTW este activado:

```text
geometry_rmse
geometry_dtw
```

### 7.4. Matching de Luminancia, Color y Textura

La geometria debe ser el filtro principal, pero color/luminancia/textura aportan
informacion para desempatar y descartar falsos positivos.

Para cada cara se propone extraer una banda interior paralela al borde:

```text
face curve -> normal interior -> banda de N pixeles dentro de la pieza
```

Senales iniciales:

- **Luminancia**: perfil 1D de intensidad, por ejemplo en espacio Lab canal `L`
  o gris.
- **Color**: perfil medio/mediano de `Lab` o `HSV` en la banda interior.
- **Textura**: gradiente local, varianza de luminancia o LBP sencillo.

Distancias sugeridas:

- luminancia: RMSE o correlacion normalizada;
- color: distancia euclidea en Lab, promediada por puntos;
- textura: diferencia de histograma o RMSE de gradiente.

Estas senales deben agregarse como informacion complementaria:

```text
score_total =
    w_geom * geometry_rmse
  + w_luma * luminance_distance
  + w_color * color_distance
  + w_texture * texture_distance
  + penalties
```

Pesos iniciales para exploracion:

```text
w_geom = 1.00
w_luma = 0.20
w_color = 0.20
w_texture = 0.10
```

El notebook debe mostrar las senales por separado antes de combinarlas. Si la
geometria es mala, color/textura no deben rescatar la candidata.

### 7.5. Penalizaciones

El score debe poder penalizar:

- tipo de cara `desconocida`;
- perfil `needs_review`;
- reconstruction/sampling RMSE alto;
- flags de calidad relevantes;
- falta de alguna restriccion esperada.

## 8. Ranking de Candidatas

Para cada posicion buscada se debe producir:

```text
top_1, top_2, top_3
```

Cada candidata debe mostrar:

- `pieza_id`;
- `piece_index`;
- `origen`;
- cara usada para encaje;
- tipos de caras comparadas;
- distancia geometrica;
- distancia DTW si esta activada;
- distancia de luminancia;
- distancia de color;
- distancia de textura;
- penalizaciones aplicadas;
- score final;
- flags de calidad.

El notebook debe permitir inspeccionar el ranking antes de visualizar la
composicion.

## 9. Visualizacion

### 9.1. Visualizacion de Matching Local

Para cada direccion cardinal:

- mostrar pieza central;
- mostrar candidata;
- dibujar caras comparadas;
- mostrar las curvas normalizadas superpuestas;
- mostrar score y distancia.

### 9.2. Visualizacion de Candidatas

Para cada posicion:

- mostrar las tres mejores candidatas;
- incluir su PNG original recortado;
- mostrar metadatos principales;
- indicar por que fueron aceptadas o penalizadas.

### 9.3. Visualizacion `3x3`

Construir una vista con:

```text
NW  N  NE
W   C  E
SW  S  SE
```

En esta version:

- `C` es la pieza fija seleccionada;
- cada posicion muestra la mejor candidata disponible;
- posiciones sin candidata se muestran vacias;
- si faltan piezas en la base de datos no se considera error;
- si una pieza tiene baja calidad, se debe marcar visualmente.

La composicion visual no necesita ser perfecta fisicamente en esta fase. El
objetivo es validar ranking y encaje local.

## 10. Notebook Inicial

El desarrollo debe empezar en notebook para iterar rapido.

Notebook propuesto:

```text
notebooks/piece_search_exploration.ipynb
```

Generador propuesto:

```text
scripts/generate_piece_search_notebook.py
```

Controles minimos:

```python
PUZZLE = "ciudad"
ORIGEN = None
CENTER_PIEZA_ID = 836
PROFILE_VERSION = "piece_profile_v1"
TOP_K = 3
ALLOW_NEEDS_REVIEW = True
ALLOW_UNKNOWN_FACE = True
```

Flujo del notebook:

1. Cargar piezas perfiladas.
2. Seleccionar pieza central.
3. Mostrar sus cuatro caras y tipos.
4. Seleccionar una direccion de busqueda.
5. Repetir la busqueda para las cuatro caras originales de la pieza central.
6. Filtrar candidatas por tipo de cara.
7. Medir distancia geometrica con curva inversa.
8. Calcular, si esta activado, DTW geometrico.
9. Calcular senales de luminancia, color y textura.
10. Mostrar top 3 candidatas por cara/direccion.
11. Dibujar pieza central + mejor candidata.
12. Construir vista `3x3` con mejores candidatas disponibles.

## 11. Persistencia

En la primera iteracion no es obligatorio persistir resultados de matching.

Si se decide persistir, crear una tabla futura:

```text
pieza_match_candidates
```

Campos sugeridos:

- `id`
- `center_pieza_id`
- `candidate_pieza_id`
- `direction`
- `center_face_key`
- `candidate_face_key`
- `geometry_distance`
- `score`
- `rank`
- `profile_version`
- `metadata_json`
- `created_at`

## 12. Criterios de Aceptacion

- El notebook carga perfiles existentes desde `pieza_perfiles`.
- Se puede seleccionar una pieza central fija.
- Para cada cara cardinal, el sistema muestra cuantas candidatas sobreviven al
  filtro `macho`/`hembra`/`lisa`.
- La distancia geometrica se calcula sobre curvas parametricas normalizadas y
  complementarias.
- La busqueda inicial produce rankings independientes para las cuatro caras
  originales de la pieza central.
- El notebook muestra RMSE euclideo y permite calcular DTW como diagnostico.
- El notebook calcula y muestra senales de luminancia, color y textura, aunque
  la geometria siga siendo el criterio principal.
- Se muestran las tres mejores candidatas por direccion.
- Se dibuja la pieza central con la mejor candidata para al menos una
  direccion.
- Se genera una vista `3x3` con las mejores candidatas encontradas y huecos
  vacios cuando no haya dato suficiente.
- El flujo tolera piezas faltantes o perfiles `needs_review`.
- El notebook no modifica estado persistente de piezas.

## 13. Riesgos y Decisiones Pendientes

### Esquinas Sensibles

Los descriptores dependen de la deteccion de esquinas. Un error pequeno en una
esquina puede cambiar la normalizacion de la cara y afectar al matching.

Mitigacion inicial:

- mostrar flags de calidad;
- permitir revisar visualmente curvas superpuestas;
- no tomar decisiones globales con una sola distancia.

### Orientacion de Caras

La correspondencia entre `1-2`, `2-3`, `3-4`, `4-1` y direcciones cardinales
debe validarse visualmente. Si la convencion no es correcta, los rankings no
seran interpretables.

### Matching Local vs Global

La mejor candidata local para una cara puede no ser la mejor solucion global
del puzzle. Esta fase solo valida matching local.

### Calidad de Segmentacion

Si SAM 3 recorta mal una pieza o genera alpha irregular, el perfil y el matching
pueden fallar. Las piezas de mala calidad deben quedar marcadas y no bloquear
el notebook.

## 14. Trabajo Futuro

- Incorporar color y textura cerca de la cara.
- Refinar esquinas usando ajuste local de caras adyacentes.
- Hacer matching multi-restriccion real sobre composiciones parciales.
- Persistir rankings y comparar historicos.
- Pasar del notebook a scripts reutilizables.
- Integrar busqueda con deteccion de huecos del tablero.
