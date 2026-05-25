# Guía para Agentes (Project Environment Guidelines)

Esta guía define las normas de desarrollo para los agentes que interactúan con
este proyecto.

## Estructura del Proyecto

- `src/`: módulos Python reutilizables y lógica principal del proyecto.
  - `sam3_extractor.py`: extracción de piezas con SAM 3 y persistencia en
    `sam3_piezas`.
  - `piece_profile.py`: extracción geométrica del perfil de una pieza.
  - `piece_profile_db.py`: persistencia de perfiles en `pieza_perfiles`.
  - `piece_search.py`: motor de búsqueda de candidatos por caras.
  - `detector.py`: pipeline OpenCV de referencia.
  - `db.py`, `sam3_db.py`: utilidades de base de datos.
- `scripts/`: entrypoints manuales y generadores de notebooks.
  - Ejecutar desde la raíz del repo con `uv run python ...`.
  - Los scripts deben delegar la lógica en `src/` siempre que sea razonable.
- `notebooks/`: notebooks de exploración generados o manuales.
  - Pueden usar `print()` y visualizaciones interactivas.
  - Si un notebook se genera desde `scripts/generate_*.py`, editar primero el
    generador y después regenerar el notebook.
- `tests/`: tests unitarios y de integración ligera con `pytest`.
- `docs/`: documentación funcional y técnica.
  - `docs/PRD/`: PRDs.
  - `docs/scripts_manual.md`: manual de ejecución de scripts.
  - `docs/procesado_piezas.md`: descripción del pipeline de procesado.
- `plans/`: planes técnicos por iniciativa o feature.
- `request/`: notas o peticiones originales del usuario.
- `data/`: entradas y base SQLite local (`data/puzzscan_v2.db`).
- `output/`: resultados generados por pipelines.
- `logs/`: logs de ejecución.
- `utils/`: utilidades compartidas, incluido el logger centralizado.

## Flujo de Datos Principal

1. `src.sam3_extractor` detecta piezas en imágenes fuente y genera PNGs con
   transparencia.
2. Los resultados se guardan en `data/puzzscan_v2.db`, tabla `sam3_piezas`.
3. `scripts/profile_pieces.py` lee `sam3_piezas`, extrae perfiles geométricos
   desde el canal alpha y escribe `pieza_perfiles`.
4. `src.piece_search` lee perfiles y busca candidatos compatibles por cara.
5. Los notebooks de exploración visualizan y validan cada fase.

## Gestión de Dependencias

- Se utiliza **`uv`** como gestor principal de dependencias.
- Comandos habituales:
  - `uv sync`
  - `uv add <paquete>`
  - `uv run python -m src.sam3_extractor`
  - `uv run python scripts/profile_pieces.py --puzzle ciudad`
- No instalar dependencias con `pip` directamente salvo que el usuario lo pida
  explícitamente.

## Convenciones de Ejecución

- Ejecutar comandos desde la raíz del repositorio:

```powershell
cd C:\proxectos\puzzscan
```

- Preferir ejecución como módulo para módulos en `src/`:

```powershell
uv run python -m src.sam3_extractor
uv run python -m src.detector
```

- Para scripts en `scripts/`, usar:

```powershell
uv run python scripts/profile_pieces.py
uv run python scripts/generate_piece_profile_notebook.py
uv run python scripts/generate_piece_search_notebook.py
```

## Convenciones de Código Python

- La lógica de negocio debe vivir en `src/`; los scripts deben ser finos.
- Usar tipos explícitos en funciones públicas y estructuras persistidas.
- Preferir `dataclass` para modelos internos simples.
- Usar `pathlib.Path` para rutas.
- Usar `json` para serializar campos complejos guardados en SQLite.
- Usar `numpy.ndarray` para geometría, curvas, máscaras e imágenes.
- Mantener constantes de versión y parámetros cerca del módulo que los usa
  (`PROFILE_VERSION`, `CONTROL_POINTS_PER_FACE`, etc.).
- Cuando cambie la semántica de un perfil persistido, incrementar
  `PROFILE_VERSION` para evitar mezclar datos antiguos con datos nuevos.
- No introducir abstracciones nuevas si una función local clara es suficiente.
- No modificar datos generados o base de datos dentro de tests unitarios salvo
  usando `tmp_path` o monkeypatching.

## Convenciones de Notebooks

- Los notebooks pueden imprimir y mostrar figuras.
- Si el notebook depende de código local que cambia a menudo, recargar el módulo
  con `importlib.reload(...)` en la celda de setup.
- Mantener la configuración al principio del notebook.
- Para notebooks generados, no editar directamente el `.ipynb` salvo ajustes
  exploratorios temporales; persistir cambios en el generador.
- Los notebooks de perfiles deben enseñar el mismo conjunto de piezas
  representativas a través de todas las fases cuando se comparan pasos.

## Convenciones de Base de Datos

- La base local principal es `data/puzzscan_v2.db`.
- Tablas principales:
  - `sam3_piezas`: piezas extraídas por SAM 3.
  - `pieza_perfiles`: perfiles geométricos por pieza y versión.
  - `piezas`: resultados del pipeline OpenCV de referencia.
- Al recalcular perfiles, reemplazar registros por `(pieza_id,
  profile_version)` para evitar duplicados.
- No asumir que la base de datos está versionada por git; tratarla como estado
  local generado.

## Logging

- **NO usar** `print()` para registrar información de ejecuciones, excepto en
  notebooks.
- Para cualquier script o módulo Python no-notebook, importar y usar el logger
  centralizado:

```python
from utils.logger import logger
```

- El logger está basado en `loguru`, muestra mensajes por consola y guarda en
  `logs/`.
- Los logs incluyen timestamp, nivel, fichero y número de línea.

## Calidad de Código, Tipado y Git Hooks

- El formateo, isort y linting general está unificado usando **`ruff`**.
- La longitud de línea es 88 caracteres.
- El control de tipos estático se realiza con **`mypy`** en modo estricto.
- El proyecto usa **`pre-commit`** antes de cada commit.
- Hooks configurados:
  - `end-of-file-fixer`
  - `trailing-whitespace`
  - `ruff`
  - `ruff-format`
  - `mypy`
  - `detect-secrets`

Antes de cualquier commit o cambio amplio, ejecutar las validaciones relevantes:

```powershell
uv run ruff check src tests scripts
uv run pytest
```

Para cambios acotados, se puede ejecutar solo el subconjunto afectado, por
ejemplo:

```powershell
uv run ruff check src\piece_search.py tests\test_piece_search.py
uv run pytest tests\test_piece_search.py -q
```

## Gestión de Secretos

- Nunca hacer hardcode de contraseñas, tokens o APIs.
- Los secretos deben cargarse usando `python-dotenv` desde `.env`.
- `.env` local no debe subirse al repositorio.
- `detect-secrets` está configurado con `.secrets.baseline`.

## Documentación

- Actualizar `docs/scripts_manual.md` cuando cambien comandos, inputs u outputs.
- Actualizar `docs/procesado_piezas.md` cuando cambie el pipeline de piezas.
- Actualizar PRDs en `docs/PRD/` cuando cambien requisitos o criterios de
  aceptación.
- Actualizar planes en `plans/` cuando cambie la implementación prevista por
  fases.

## Git y Cambios Locales

- No revertir cambios del usuario sin permiso explícito.
- Revisar `git status --short` antes de commits.
- Incluir notebooks generados solo cuando el cambio del generador también esté
  incluido.
- No commitear caches, logs, bases de datos locales o outputs pesados salvo
  petición explícita.

## Excepciones

- Notebooks: pueden usar `print()`, salidas estándar y código exploratorio.
- Scripts generadores de notebooks pueden contener strings largos de celdas,
  pero el archivo generador debe pasar `ruff`.
