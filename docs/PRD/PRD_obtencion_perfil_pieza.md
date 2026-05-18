# PRD: Obtencion del Perfil Geometrico de Pieza

## 1. Vision y Objetivo

Esta fase convierte cada pieza ya segmentada en una representacion geometrica
comparable: un perfil normalizado de sus cuatro caras, con curvas fieles al
contorno real y descriptores reducidos para busqueda posterior.

El objetivo no es resolver todavia el puzzle, sino crear la base geometrica que
permitira comparar una pieza disponible con los huecos detectados en el tablero.

## 2. Problema que Resuelve

Las imagenes recortadas de piezas no son suficientes para comparar encajes. El
sistema necesita saber, para cada cara de cada pieza:

- donde empieza y termina la cara;
- si la cara es lisa, macho o hembra;
- cual es la curva exacta del contorno entre esquina y esquina;
- como representar esa curva de forma compacta, reproducible y comparable.

Esta fase genera esos perfiles y los persiste en la base de datos.

## 3. Alcance

### Incluido

- Extraer el contorno exterior de cada pieza segmentada.
- Detectar las esquinas principales de la pieza.
- Dividir el contorno en cuatro caras cerradas en formato `1-2-3-4-1`.
- Convertir cada cara en una curva normalizada entre dos esquinas.
- Representar cada cara con un eje base donde la linea entre esquinas es `Y=0`.
- Clasificar cada cara como `lisa`, `macho` o `hembra`.
- Generar un perfil numerico con numero fijo de puntos por cara.
- Calcular una reduccion dimensional que represente la curva de forma compacta.
- Guardar perfiles, coordenadas y descriptores en SQLite.
- Crear notebook de exploracion y QA visual.

### Fuera de Alcance

- Matching final contra huecos del puzzle.
- Deteccion de huecos en el tablero.
- Scoring completo pieza-hueco.
- Interfaz de usuario.
- Correccion manual interactiva de esquinas o contornos.

## 4. Entradas

La fase parte de las piezas ya extraidas por los pipelines existentes:

```text
output/<nombre_puzzle>/piezas_sam3/<origen>/pieza_<n>.png
```

La fuente principal para obtener el perfil sera el canal alpha del PNG extraido.
El alpha define la mascara real de la pieza ya recortada y sera la base para
extraer el contorno exterior.

Tambien debe poder leer los metadatos existentes desde:

```text
data/puzzscan_v2.db
```

Tabla de referencia inicial:

```text
sam3_piezas
```

Campos utiles:

- `id`
- `nombre_puzzle`
- `origen`
- `piece_index`
- `ruta_imagen`
- `contour_json`
- `bounding_box_json`
- `posicion_original`

Uso de fuentes:

- `ruta_imagen`: fuente principal. Se lee el PNG y se usa su canal alpha para
  construir la mascara.
- `contour_json`: fuente secundaria para trazabilidad, comparacion o fallback si
  el PNG no tiene alpha valido.
- `bounding_box_json` y `posicion_original`: fuentes de trazabilidad para mapear
  coordenadas del recorte a la imagen origen.

## 5. Modelo Geometrico

### 5.1. Sistema de Caras

Cada pieza se modela como un contorno cerrado con cuatro esquinas principales.
Las caras se nombran en orden continuo:

```text
1-2, 2-3, 3-4, 4-1
```

Para almacenamiento y trazabilidad, el perfil completo se expresa como:

```text
1-2-3-4-1
```

La numeracion debe ser determinista para que una misma pieza produzca el mismo
perfil al reprocesarse. La orientacion inicial se definira a partir del recorte
original sin rotar y del orden del contorno.

### 5.2. Convenciones Canonicas

Para que los perfiles sean comparables entre ejecuciones, piezas y futuros
huecos, se fijan las siguientes convenciones:

- El contorno exterior se recorrera siempre en sentido horario.
- Si OpenCV devuelve el contorno en sentido antihorario, se invertira antes de
  dividirlo en caras.
- El punto `1` sera la esquina superior izquierda en coordenadas del recorte.
- El punto `2` sera la esquina superior derecha.
- El punto `3` sera la esquina inferior derecha.
- El punto `4` sera la esquina inferior izquierda.
- Las caras se definiran como `1-2`, `2-3`, `3-4`, `4-1`.
- Para cada cara, `X` avanza desde la primera esquina hacia la segunda.
- Para cada cara, `Y positivo` apunta hacia el exterior de la pieza.
- `Y negativo` apunta hacia el interior de la pieza.

La regla para ordenar esquinas sera:

1. Separar las dos esquinas superiores y las dos inferiores usando coordenada
   `y` en el recorte.
2. Ordenar cada pareja por coordenada `x`.
3. Asignar `1=superior izquierda`, `2=superior derecha`,
   `3=inferior derecha`, `4=inferior izquierda`.

Estas convenciones convierten la secuencia `1-2-3-4-1` en un contrato estable
de datos, no en una visualizacion accidental.

### 5.3. Deteccion de Esquinas

Una esquina principal es un punto convexo del contorno exterior que separa dos
caras de la pieza. No debe confundirse con los picos o valles de los machos y
hembras.

Metodo principal:

1. Suavizar ligeramente el contorno exterior para reducir ruido de mascara.
2. Calcular curvatura local a lo largo del contorno.
3. Identificar candidatos de alta curvatura convexa.
4. Descartar candidatos concavos asociados a entradas.
5. Seleccionar cuatro candidatos principales maximizando separacion sobre el
   contorno y consistencia geometrica.

Fallback:

- Usar `cv2.approxPolyDP` con tolerancia adaptativa hasta obtener cuatro
  vertices principales estables.

Validaciones minimas:

- Deben detectarse exactamente cuatro esquinas principales.
- Las esquinas deben estar sobre el contorno real.
- Cada cara resultante debe tener longitud no degenerada.
- Las cuatro caras deben cubrir el contorno completo sin saltos.
- Si no se cumplen estas condiciones, el perfil se marcara como `needs_review`
  o `invalid`.

### 5.4. Curva de una Cara

Cada cara es el tramo de contorno entre dos esquinas consecutivas.

Para cada cara:

1. Se toma la linea recta entre sus dos esquinas como eje base.
2. Esa linea se normaliza a un intervalo comun `X=[0, 1]`.
3. La desviacion perpendicular respecto al eje base se representa como `Y`.
4. La linea entre esquinas equivale a `Y=0`.
5. Los salientes, o `machos`, se representan con `Y positivo`.
6. Las entradas, o `hembras`, se representan con `Y negativo`.

Normalizacion:

```text
X = proyeccion_sobre_eje / longitud_cara
Y = distancia_perpendicular / longitud_cara
```

Esto permite comparar perfiles aunque las fotos tengan distinta escala.

Esta convencion hace que la curva de una pieza pueda compararse directamente
contra la curva invertida de un hueco.

### 5.5. Numero Fijo de Puntos

Cada cara debe remuestrearse a un numero fijo de puntos.

Decision de producto:

```text
N puntos por cara = configurable, valor inicial recomendado: 128
```

Razon:

- facilita comparar curvas con distancia punto a punto;
- permite almacenar tensores de forma fija;
- simplifica reduccion dimensional;
- mantiene suficiente detalle para entradas y salidas de puzzle.

El valor debe poder configurarse sin cambiar el esquema conceptual.

## 6. Clasificacion de Caras

Cada cara debe clasificarse como:

- `lisa`: desviacion pequena respecto a `Y=0`, tipica de borde exterior;
- `macho`: saliente predominante hacia `Y positivo`;
- `hembra`: entrada predominante hacia `Y negativo`;
- `desconocida`: si el contorno no permite una clasificacion fiable.

La clasificacion debe calcularse a partir de la curva normalizada, no solo por
bounding boxes. Deben registrarse metricas auxiliares para auditar la decision:

- amplitud maxima positiva;
- amplitud maxima negativa;
- area firmada de la curva;
- posicion del pico o valle principal;
- score de confianza de clasificacion.

### 6.1. Tipo de Pieza

A partir de la clasificacion de sus caras, cada pieza debe recibir un tipo
derivado:

- `interior`: cero caras lisas.
- `border`: una cara lisa.
- `corner`: dos caras lisas consecutivas.
- `unknown`: clasificacion ambigua, mas de dos caras lisas, caras lisas no
  consecutivas o baja confianza.

Este campo sera util para filtrar candidatos antes del matching geometrico.

## 7. Reduccion Dimensional

Ademas de guardar la curva remuestreada completa, el sistema debe calcular una
representacion reducida para busqueda eficiente.

Opciones aceptadas para la primera version:

- coeficientes de Fourier de baja frecuencia;
- PCA entrenada sobre curvas normalizadas;
- wavelets o spline simplificada.

Decision inicial recomendada:

```text
Guardar curva completa + descriptor Fourier configurable.
```

Razon:

- la curva completa conserva fidelidad y permite QA;
- Fourier da un descriptor compacto y facil de comparar;
- se puede reemplazar o complementar con PCA mas adelante.

El descriptor reducido no debe ser la unica fuente de verdad.

## 8. Persistencia

Se debe ampliar la persistencia sin romper los datos existentes.

### 8.1. Nueva Tabla Recomendada: `pieza_perfiles`

Campos propuestos:

- `id` INTEGER PRIMARY KEY AUTOINCREMENT
- `pieza_id` INTEGER NOT NULL
- `pipeline` TEXT NOT NULL, ejemplo: `sam3`
- `nombre_puzzle` TEXT NOT NULL
- `origen` TEXT NOT NULL
- `piece_index` INTEGER NOT NULL
- `corner_points_json` TEXT NOT NULL
- `face_order` TEXT NOT NULL, ejemplo: `1-2-3-4-1`
- `faces_json` TEXT NOT NULL
- `piece_kind` TEXT NOT NULL
- `profile_status` TEXT NOT NULL
- `quality_flags_json` TEXT NOT NULL
- `profile_version` TEXT NOT NULL
- `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP

Restriccion recomendada:

```sql
UNIQUE(pieza_id, profile_version)
```

Esto garantiza idempotencia por pieza y version de algoritmo.

### 8.2. Estructura de `faces_json`

Formato conceptual:

```json
{
  "1-2": {
    "tipo": "macho",
    "points": [[0.0, 0.0], [0.01, 0.02]],
    "reduced_descriptor": [0.12, -0.04, 0.01],
    "metrics": {
      "max_positive": 0.23,
      "max_negative": -0.04,
      "signed_area": 0.11,
      "classification_confidence": 0.92
    }
  }
}
```

Los JSON deben ser validos, versionados y reproducibles.

### 8.3. Coordenadas y Trazabilidad

El sistema debe guardar coordenadas en tres sistemas:

- coordenadas del recorte PNG;
- coordenadas de la imagen origen;
- coordenadas normalizadas.

Campos recomendados:

- `corner_points_crop_json`
- `corner_points_source_json`
- `corner_points_normalized_json`

Para cada cara, los puntos de `faces_json` se guardaran en coordenadas
normalizadas de la propia cara. Opcionalmente se guardaran indices al contorno
original para poder depurar que tramo genero cada curva.

### 8.4. Idempotencia

El procesamiento debe ser reproducible y no duplicar datos.

Estrategia:

1. Definir `profile_version` para versionar cambios de algoritmo.
2. Antes de insertar, borrar cualquier perfil previo de la misma
   combinacion `(pieza_id, profile_version)`.
3. Insertar el perfil nuevo con `created_at` actualizado.
4. Mantener la restriccion `UNIQUE(pieza_id, profile_version)` como proteccion
   adicional.

## 9. Modulos y Entregables

### 9.1. Codigo

Modulos esperados:

- `src/piece_profile.py`: extraccion de contorno, esquinas, caras y perfiles.
- `src/piece_profile_db.py`: persistencia de perfiles en SQLite.
- `scripts/generate_piece_profile_notebook.py`: generador de notebook de exploracion.

### 9.2. Notebook de Exploracion

Notebook esperado:

```text
notebooks/piece_profile_exploration.ipynb
```

Debe permitir visualizar:

- imagen original de la pieza;
- mascara y contorno exterior;
- esquinas detectadas;
- caras numeradas `1-2-3-4-1`;
- curva normalizada por cara;
- clasificacion `lisa/macho/hembra`;
- reconstruccion aproximada desde el descriptor reducido;
- error entre curva original y reducida.

## 10. Flujo de Procesamiento

1. Cargar pieza segmentada y metadatos desde SQLite.
2. Leer `ruta_imagen` y extraer la mascara desde el canal alpha del PNG.
3. Si el alpha no existe o no es valido, intentar fallback desde `contour_json`.
4. Validar calidad minima de mascara y contorno.
5. Extraer contorno exterior ordenado.
6. Forzar sentido horario.
7. Detectar cuatro esquinas principales.
8. Ordenar esquinas segun convencion canonica.
9. Dividir el contorno en caras.
10. Para cada cara:
   - normalizar eje esquina-esquina;
   - remuestrear a `N` puntos;
   - orientar signo de `Y` hacia el exterior de la pieza;
   - clasificar tipo de cara;
   - calcular descriptor reducido;
   - calcular metricas de calidad.
11. Calcular tipo derivado de pieza.
12. Guardar resultado en `pieza_perfiles`.
13. Generar visualizaciones de QA.

## 11. Requisitos de Calidad

- La extraccion debe ser idempotente por `pieza_id` y `profile_version`.
- El perfil no debe depender del orden accidental de puntos devuelto por OpenCV.
- Las curvas deben cerrar con continuidad en `1-2-3-4-1`.
- Las esquinas detectadas deben estar sobre el contorno real.
- El remuestreo debe conservar entradas y salidas visibles.
- La reduccion dimensional debe reportar error de reconstruccion.
- Los casos de baja confianza deben marcarse como `desconocida` o `needs_review`.

### 11.1. Validacion Previa de Mascara y Contorno

Antes de generar perfiles, cada pieza debe pasar controles minimos:

- existe canal alfa o mascara valida;
- si existe canal alpha, contiene suficientes pixeles no transparentes;
- el alpha se puede binarizar limpiamente como mascara de pieza;
- hay una sola componente principal;
- el area supera un minimo configurable;
- el contorno exterior es cerrado;
- la pieza no toca excesivamente los bordes del recorte;
- se detectan exactamente cuatro esquinas principales;
- ninguna cara tiene longitud degenerada;
- el contorno no presenta saltos o agujeros que rompan la cara.

Estados posibles:

- `valid`: perfil utilizable para matching.
- `needs_review`: perfil generado, pero con baja confianza o flags de calidad.
- `invalid`: no se puede generar un perfil fiable.

`quality_flags_json` debe registrar los motivos concretos, por ejemplo:

- `missing_alpha`
- `multiple_components`
- `touches_crop_border`
- `corner_detection_failed`
- `degenerate_face`
- `high_reconstruction_error`

## 12. Criterios de Aceptacion

1. Dada una pieza segmentada, el sistema detecta cuatro esquinas y genera cuatro
   caras en formato `1-2-3-4-1`.
2. Cada cara se guarda con una curva normalizada de `N` puntos.
3. Cada cara queda clasificada como `lisa`, `macho`, `hembra` o `desconocida`.
4. La curva completa y el descriptor reducido se guardan en SQLite.
5. El notebook muestra visualmente contorno, esquinas, caras, curvas y error de
   reduccion.
6. Al reprocesar la misma pieza no se duplican perfiles activos para la misma
   version.
7. La numeracion y orientacion de caras es estable entre ejecuciones.

### 12.1. Umbrales Iniciales Configurables

Valores iniciales para validar la primera implementacion:

- `corner_count = 4`
- `min_face_points >= 16` antes de remuestrear.
- `face_length_ratio` entre `0.5` y `1.8` respecto a la mediana de las cuatro
  caras.
- `classification_confidence >= 0.70` para aceptar `lisa`, `macho` o `hembra`.
- `reconstruction_rmse <= 0.03` para el descriptor reducido.
- `mask_component_count = 1`.
- `profile_status = valid` para al menos el `90%` de piezas correctamente
  segmentadas en una muestra de QA.

Estos umbrales deben vivir en configuracion y ajustarse mediante el notebook de
exploracion.

## 13. Riesgos y Decisiones Pendientes

### Riesgos

- Piezas mal segmentadas pueden producir esquinas falsas.
- Piezas con perspectiva fuerte pueden deformar curvas.
- La clasificacion macho/hembra depende de definir correctamente el signo de la
  normal.
- Una reduccion demasiado agresiva puede perder detalles clave del encaje.

### Decisiones Pendientes

- Valor definitivo de `N` puntos por cara tras pruebas: 64, 128 o 256.
- Descriptor reducido final: Fourier, PCA, splines o combinacion.
- Politica para piezas de borde con una o dos caras lisas.
- Umbrales de confianza para marcar una cara como `desconocida`.

## 14. No Objetivos de Esta Fase

- No se busca decidir que pieza encaja en un hueco.
- No se busca resolver el puzzle completo.
- No se busca crear una UI.
- No se busca corregir manualmente perfiles.

Esta fase termina cuando cada pieza disponible tiene un perfil geometrico
estable, auditable y listo para ser comparado en fases posteriores.
