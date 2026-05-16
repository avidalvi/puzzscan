# Guía para Agentes (Project Environment Guidelines)

Esta guía define las normas de desarrollo para los agentes que interactúan con este proyecto.

## Gestión de Dependencias
- Se utiliza **`uv`** como gestor principal de dependencias (`uv add`, `uv run`).

## Calidad de Código, Tipado y Git Hooks
- El formateo, isort y linting general está unificado usando **`ruff`**. La configuración de `isort` se encuentra en el bloque `[tool.ruff.lint.isort]` de `pyproject.toml`.
- El control de tipos estático se realiza con **`mypy`**, que usa modo estricto según `pyproject.toml`.
- El proyecto usa **`pre-commit`** para asegurar que todo el código cumple con las normativas antes de cada commit.
- **Hooks configurados:**
  - `end-of-file-fixer` (eol: asegura saltos de línea al final de archivo).
  - `trailing-whitespace` (elimina espacios en blanco innecesarios al final de línea).
  - `ruff` (corrección de errores y organización de imports).
  - `ruff-format` (formateo general de código).
  - `mypy` (revisión estricta de tipos en Python).
  - `detect-secrets` (escaneo de credenciales hardcodeadas).

*Antes de cualquier commit manual o cuando se desarrollen scripts nuevos, el agente debe asegurarse de que el código cumple con estas reglas.*

## Gestión de Secretos
- **Nunca hacer hardcode** de contraseñas, tokens o APIs en el código.
- Los secretos deben cargarse usando **`python-dotenv`** desde un archivo `.env` local (que no debe subirse al repositorio).
- Se ha configurado `detect-secrets` en los hooks de pre-commit con una baseline (`.secrets.baseline`) para evitar fugas.

## Logging
- **NO usar** `print()` para registrar información de las ejecuciones, excepto en **Notebooks**.
- Para cualquier script o módulo Python (no-notebook), se debe importar y utilizar el logger centralizado.
  - **Uso del logger:** `from utils.logger import logger`
- **Configuración del logger:** El logger (basado en `loguru`) está configurado para mostrar mensajes por consola y guardar en ficheros dentro de la carpeta `logs/`.
- **Formato:** Los logs incluyen automáticamente timestamp, tipo/nivel de log, nombre del fichero, y número de línea.

## Excepciones
- **Notebooks:** Los Notebooks de la carpeta `notebooks/` están exentos de usar este sistema de log centralizado, en ellos se pueden utilizar salidas estándar para análisis interactivo y no es necesario persistir el log en la carpeta `logs/`.
