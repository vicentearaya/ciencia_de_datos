# Clasificación — proyecto de ciencia de datos

Este repositorio en la rama **`clasificacion`** organiza el flujo desde datos crudos hasta modelos de clasificación del **nivel de adicción** al videojuego.

## Estructura

| Ruta | Propósito |
|------|-----------|
| `Data_cruda/` | Datos sin procesar (CSV/GZ grandes ignorados por git; ver `.gitignore`). |
| `data_preprocesada/` | Salida del notebook: CSV listo para modelos + `preprocessing_report.json`. |
| `notebooks/preprocesamiento.ipynb` | **EDA completo**, limpieza, transformación, visualizaciones y guardado de la tabla final. |
| `models/` | Entrenamiento de **5 modelos** de clasificación (`train_models.py`). |

## Objetivo de modelado

- **Variable de negocio:** `addiction_level` (continua 0–10 en el dataset crudo).
- **Target para clasificación:** `addiction_class` (4 clases por **cuartiles** de severidad), generada en el notebook. La columna continua **no** se incluye en los predictores (evita *data leakage* respecto a la etiqueta discretizada).

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

1. Coloca el archivo crudo en `Data_cruda/` (p. ej. `gaming_mental_health_10M_40features.csv.gz`).
2. Abre y ejecuta **todo** el notebook `notebooks/preprocesamiento.ipynb` (Jupyter o VS Code / Cursor). Al final se crean:
   - `data_preprocesada/dataset_preprocesado_clasificacion.csv`
   - `data_preprocesada/preprocessing_report.json`
3. Entrena los cinco modelos desde la raíz del repo:

   ```bash
   python models/train_models.py \
     --data data_preprocesada/dataset_preprocesado_clasificacion.csv \
     --target addiction_class
   ```

En macOS con disco **sin distinción de mayúsculas**, evita crear otra carpeta `Models` junto a `models`: podrían confundirse.

## Rama

El flujo descrito vive en la rama **`clasificacion`**.
