# Clasificación — proyecto de ciencia de datos

Este repositorio en la rama **`clasificacion`** organiza el flujo desde datos crudos hasta modelos de clasificación.

## Estructura

| Ruta | Propósito |
|------|-----------|
| `Data_cruda/` | Datos sin procesar (no se versionan archivos pesados; ver `.gitignore`). |
| `data_preprocesada/` | Salida del preprocesamiento lista para entrenar. |
| `notebooks/preprocesamiento.py` | Script de preprocesamiento (reemplaza o complementa notebooks). |
| `models/` | Entrenamiento y evaluación de **5 modelos** de clasificación. |

## Entorno virtual

Python recomendado: **3.11 o 3.12**.

```bash
cd /ruta/al/repo
python3 -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate    # Windows
pip install --upgrade pip
pip install -r requirements.txt
```

## Uso (orden sugerido)

1. Coloca los archivos crudos en `Data_cruda/`.
2. Ejecuta el preprocesamiento (se irá documentando en `notebooks/preprocesamiento.py`).
3. Entrena y compara los cinco modelos desde `models/train_models.py`, por ejemplo:

   ```bash
   python models/train_models.py --target nombre_columna_clase
   ```

En macOS con disco **sin distinción de mayúsculas**, evita crear otra carpeta `Models` al lado de `models`: podrían confundirse.

Este README se ampliará conforme avance el proyecto.

## Rama

Todo lo anterior vive solo en **`clasificacion`**; las demás ramas del repositorio no se modifican desde aquí.
