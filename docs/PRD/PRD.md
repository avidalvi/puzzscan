# Product Requirements Document (PRD): PuzzScan

## 1. Visión y Objetivos del Producto
**Objetivo Principal**: Identificar, de entre un conjunto de piezas disponibles, la pieza exacta que va a continuación en un puzzle en construcción, y reconstruir visualmente el puzzle en una sola imagen de manera que se vea la secuencia de piezas a añadir.

**Problema que resuelve**: Automatizar la resolución y asistencia de puzzles complejos (ej. cielos nocturnos con información de color muy sutil), donde el encaje se basa estrictamente en la geometría y el color del entorno.

## 2. Requisitos de Entrada (Inputs)
El sistema debe poder recibir y procesar de forma continua tres inputs clave:
1. **Imágenes de piezas disponibles**: Múltiples piezas ubicadas en `data/piezas_x.png`.
2. **Estado actual del puzzle**: Foto del tablero en su estado actual, ubicado en `data/puzzle.png`.
3. **Imagen objetivo**: La imagen del puzzle completo a modo de referencia global.

## 3. Flujo de Trabajo (Workflows)

### Fase 1: Ingesta, Estandarización e Inventariado de Piezas
- **Separación y Extracción**: Detectar cada pieza individual del fondo, recortarla y aislarla.
- **Estandarización Espacial**: Calcular el bounding box de origen y alinear la pieza por su eje más largo de manera que su orientación sea determinista (parte superior e inferior consistentes).
- **Análisis de Caras**:
  - Clasificar cada cara como "macho" (saliente) o "hembra" (entrante).
  - Representar la topología como una secuencia continua de 5 caras `1-2-3-4-1` (cerrando el bucle).
  - Convertir el borde medio de cada cara en una **imagen lineal continua** y generar una función matemática/vector que describa sus curvas exactas para comparaciones estándar.
  - Almacenar el perfil de color/luminosidad de cada cara.
- **Base de Datos**: Guardar todas las piezas (imagen, descriptores geométricos, tipo de caras, color) con un ID único y estado (`AVAILABLE`).

### Fase 2: Análisis del Estado del Puzzle y Huecos
- Procesar `data/puzzle.png` para detectar la frontera actual (piezas ya colocadas).
- Identificar los "huecos" (slots) disponibles.
- Para cada hueco, generar la función/perfil de borde invertido (lo que falta por rellenar) siguiendo el mismo estándar lineal continuo utilizado en las piezas.

### Fase 3: Motor de Búsqueda y Coincidencia (Matching)
- Comparar el perfil del hueco detectado contra todas las piezas con estado `AVAILABLE` en la base de datos.
- Evaluar tanto el encaje geométrico (curvas exactas) como la compatibilidad de color/gradiente.
- **Scoring**: Calcular un **score de confianza** (Confidence Score) robusto para evaluar cuán probable es que la pieza encaje.

### Fase 4: Reconstrucción Visual y Resultados
- Seleccionar la mejor coincidencia.
- **Output Secuencial**: Reconstruir el puzzle visualmente en una sola imagen mostrando la secuencia recomendada de las próximas piezas a añadir.
- Mostrar gráficamente al usuario qué piezas específicas faltan por añadir.
- Reflejar en la interfaz/logs el score de confianza para la decisión tomada.
- **Historial**: Guardar el historial completo de piezas añadidas paso a paso y la evolución en imagen del estado del puzzle.

## 4. Arquitectura y Tecnologías Propuestas
* **Visión por Computador Clásica (OpenCV)**:
  * Segmentación: Binarización (`threshold`), contornos (`findContours`).
  * Análisis geométrico: Ejes principales para rotación/alineamiento de bounding boxes, extracción de bordes, descriptores de formas invariantes (ej. Distancia de Hausdorff o descriptores de Fourier de curvas).
* **Gestión de Datos y Estado**:
  * Base de datos **SQLite** (`puzzscan.db`) con una tabla para piezas (campos: `piece_id`, `contour_data`, `color_data`, `crop_image_path`, `status` [AVAILABLE, PLACED]).
  * Tabla separada para gestionar el **historial de colocaciones** de piezas.

* **Inteligencia Artificial (Opcional - Validadores)**:
  * Modelos de Visión y Lenguaje (VLM, ej. GPT-4V, Claude, Gemini) para orquestación de alto nivel o para validar la decisión del algoritmo matemático mediante la formulación: *"¿Ves un ajuste perfecto de forma entre esta pieza y este hueco?"*.

## 5. Criterios de Aceptación
1. El sistema puede procesar imágenes en crudo y registrar piezas en BD con sus vectores característicos lineales.
2. Identifica correctamente los huecos en el tablero de juego.
3. El algoritmo de comparación devuelve un ranking con scores de confianza lógicos.
4. Se genera una visualización paso a paso con las próximas piezas sugeridas y la vista global de la resolución.
5. El historial se mantiene íntegro tras sucesivas actualizaciones.
