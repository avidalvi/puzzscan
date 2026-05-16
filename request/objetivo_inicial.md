# Especificación Técnica: Objetivo Inicial (Fase 1)
**Módulo:** Extractor e Inventariador de Piezas

## 1. Propósito y Alcance
- **Objetivo**: Desarrollar el módulo fundacional que lee imágenes con múltiples piezas esparcidas, aísla cada pieza individualmente, estandariza su rotación, la recorta con fondo transparente y registra sus metadatos básicos en una base de datos SQLite.
- **Fuera de alcance temporal**: En este sprint *no* se realizará la búsqueda de coincidencias (matching) con el tablero, ni se perfilarán las matemáticas avanzadas de las curvas 1-2-3-4-1. El objetivo aquí es poblar una base de datos limpia y estructurada.

## 2. Entradas del Sistema
- **Rutas origen**: Imágenes organizadas por puzzle en `data/<nombre_puzzle>/` (ej. `data/ciudad/piezas_1.jpg`). El sistema debe ser capaz de procesar múltiples puzzles de forma independiente.
- **Suposiciones operativas**: 
  - Las piezas están sobre un fondo contrastante (preferiblemente liso/claro).
  - Las piezas no están montadas ni severamente superpuestas entre sí.
  - Las fotos se tomarán con un teléfono móvil a diferentes alturas. El sistema compensará esto matemáticamente (Normalización de Escala).

## 3. Pipeline de Visión Artificial (Algoritmo Principal)

### 3.0. Idempotencia (Ejecución desde Cero)
Para garantizar que el análisis se pueda recrear desde cero sin problemas:
- Al iniciar el procesamiento de un lote/imagen origen (ej. `piezas_1.jpg` del puzzle `ciudad`), se deben eliminar todos los registros asociados a ese `nombre_puzzle` y `origen` en la base de datos.
- Se debe limpiar/recrear el directorio de salida correspondiente (`output/<nombre_puzzle>/piezas/<origen>/`) para borrar recortes antiguos e imágenes anotadas previas.

### 3.1. Preprocesamiento de la Imagen
1. Leer imagen y convertir a escala de grises.
2. Aplicar un filtro de suavizado (Desenfoque Gaussiano) para eliminar ruido o texturas menores.
3. Aplicar **Detección de Bordes (Canny)** en lugar de umbralización simple. Esto hace al sistema resistente a variaciones de iluminación y a piezas del mismo color que el fondo.
4. Aplicar **Operaciones Morfológicas (Closing / Dilatación + Erosión)** sobre los bordes detectados para cerrar el perímetro de la pieza y generar una máscara sólida y rellenada.

### 3.2. Detección de Contornos y Normalización de Escala
1. Utilizar `cv2.findContours` (modo `RETR_EXTERNAL`) para capturar solo las siluetas exteriores, ignorando cualquier reflejo interno de las piezas.
2. Filtrar por Área: Descartar cualquier contorno que tenga un área inferior a un umbral base, eliminando polvo o pequeñas manchas.
3. **Normalización Automática de Escala**: Dado que las fotos tendrán distintas resoluciones/alturas, se asume que las piezas de un mismo puzzle tienen un área similar. Calcular la mediana del área de todas las piezas válidas en la imagen. Calcular un factor de escala para que esta mediana coincida con un "Área Estándar" del sistema (ej. 10,000 píxeles). Escalar geométricamente todos los contornos y coordenadas por este factor para igualar tamaños entre distintas fotos.

### 3.3. Estandarización Espacial (Alineación)
Para cada contorno válido y escalado:
1. Calcular el eje principal de inercia de la pieza usando **Momentos de la Imagen (`cv2.moments`)** o Análisis de Componentes Principales (PCA). A diferencia del rectángulo mínimo, esto evita que los salientes asimétricos tuerzan la alineación base de la cuadrícula.
2. Extraer el ángulo del eje principal.
3. **Rotación Segura**: Rotar vectorialmente el contorno por ese ángulo de manera que quede perfectamente vertical (o horizontal). El cálculo del Bounding Box de recorte se hará *después* de esta rotación geométrica para garantizar que ninguna esquina de la pieza quede cortada por falta de lienzo.

### 3.4. Extracción, Recorte y Trazabilidad
1. Calcular el centroide (o Bounding Box original) de la pieza en la imagen de origen. Este dato es vital para la trazabilidad.
2. Generar una copia de la imagen original anotada (`origen_anotada.jpg`), dibujando el `id` de cada pieza sobre su centroide. Esto permitirá al usuario localizar físicamente la pieza cuando el sistema se la sugiera.
3. Una vez rotada la imagen original (y su máscara), generar el recorte (Bounding Box) exacto de la pieza añadiendo un pequeño margen de seguridad (padding).
4. Aplicar un **Canal Alfa (Transparencia)** usando la máscara rotada, de manera que el fondo alrededor de la pieza sea transparente.

### 3.5. Extracción de Metadatos
- **Posición Original**: Coordenadas `(cx, cy)` originales del centroide en la foto fuente.
- **Colores Dominantes y Luminosidad**: 
  - Usar clustering **K-Means** (k=2 o 3) sobre los píxeles enmascarados para encontrar los verdaderos colores dominantes de la pieza (RGB), evitando las distorsiones de una media simple.
  - Almacenar también la luminosidad promedio de esos colores dominantes.

## 4. Arquitectura de Datos (Persistencia)
- **Base de Datos**: `data/puzzscan.db` (SQLite).
- **Esquema de la tabla `piezas`**:
  - `id` (INTEGER PRIMARY KEY AUTOINCREMENT).
  - `nombre_puzzle` (TEXT): Identificador del puzzle (ej. "ciudad").
  - `origen` (TEXT): Identificador del lote/imagen (ej. "piezas_1").
  - `ruta_imagen` (TEXT): Ruta al archivo `.png` del recorte generado.
  - `posicion_original` (TEXT): JSON con coordenadas originales `{"x": cx, "y": cy}`.
  - `factor_escala` (REAL): Factor de multiplicación aplicado para normalizar su tamaño.
  - `colores_dominantes` (TEXT): JSON array con los 2-3 colores K-Means `[[R1, G1, B1], [R2...]]`.
  - `luminosidad` (REAL): Valor promedio de luminosidad (0-255).
  - `estado` (TEXT): Estado lógico, por defecto `"AVAILABLE"`.
  - `fecha_creacion` (TIMESTAMP): Fecha de ingesta.

## 5. Salidas y Entregables del Sprint
1. **Imagen Anotada de Origen**: Un fichero `output/<nombre_puzzle>/piezas/<origen>_anotada.jpg` con los IDs dibujados sobre las piezas originales.
2. **Recortes de Imagen**: Ficheros `.png` guardados de forma ordenada en `output/<nombre_puzzle>/piezas/<origen>/pieza_<id>.png`.
3. **Base de Datos poblada**: El fichero SQLite conteniendo una fila por cada pieza detectada y su coordenada original.
4. **Módulos Python**:
   - `src/db.py`: Lógica para inicializar y gestionar SQLite.
   - `src/detector.py`: Pipeline de OpenCV detallado arriba.
   - Lógica de control para no duplicar piezas si se ejecuta el script dos veces sobre la misma imagen origen (hacer *drop* previo de ese origen o saltarlo).

## 6. Casos Límite a Contemplar
- **Piezas cortadas en el borde**: Identificar si un contorno toca el margen absoluto de la foto origen y descartarlo.
- **Sombras**: El Thresholding debe ajustarse para que las sombras proyectadas por las piezas no se confundan con la pieza en sí (afectando la forma del contorno).

## 7. Pruebas y Controles de Calidad (QA)
Para garantizar la fiabilidad del módulo, se deben implementar los siguientes controles y pruebas automatizadas:

### 7.1. Pruebas Unitarias (Unit Tests)
- **Módulo de Base de Datos**: Validar que la inserción de piezas funciona, y que la limpieza por idempotencia (`nombre_puzzle` + `origen`) elimina correctamente los registros sin afectar a otros puzzles.
- **Transformaciones Geométricas**: Validar matemáticamente que la función de alineación (Momentos/PCA) rota una figura asimétrica de prueba a su eje vertical/horizontal.

### 7.2. Pruebas de Integración (Integration Tests)
- **Pipeline Completo**: Procesar una imagen sintética o controlada (con un número conocido de piezas). Verificar que la cantidad de archivos creados y registros en BD coincide exactamente con el número esperado.
- **Test de Idempotencia**: Ejecutar el pipeline dos veces seguidas sobre la misma imagen. Verificar que no se duplican registros en BD y que la carpeta de recortes tiene la misma cantidad de archivos tras la segunda pasada.

### 7.3. Controles de Calidad de Datos (Manejo de Excepciones)
- **Control de Formato**: Comprobar que `colores_dominantes` y `contorno_simplificado` se están guardando como JSON arrays válidos.
- **Control de Fondo (Transparencia)**: Verificar aleatoriamente que los recortes PNG generados contienen un canal Alpha y que los bordes de la imagen recortada son transparentes.
- **Control de Ruido**: Asegurar que la BD no registra "falsas piezas" verificando que ningún `contorno_raw` tenga un área significativamente menor o mayor que la moda estadística de todas las piezas (para cazar errores de segmentación).
