# Clasificación multiclase de riesgo de trastorno por gaming

Este repositorio usa el dataset real jordano [Internet Gaming Disorder and Sleep Quality among Jordanian University Students](https://zenodo.org/records/13382368) para clasificar a los usuarios en **3 niveles de riesgo de trastorno por gaming**.

## Objetivo del proyecto

El objetivo principal es construir una base metodológica para una futura plataforma donde un usuario responda un cuestionario breve y reciba una clasificación interpretable de riesgo.

La variable objetivo es `igd_label`, con tres clases:

- `0`: **Jugador sin indicadores relevantes de trastorno**
- `1`: **Jugador en riesgo de desarrollar problemas por gaming**
- `2`: **Jugador con alta probabilidad de trastorno por gaming**

Estas clases se derivan de las respuestas a los 9 ítems del instrumento IGD:

- `0 a 2` respuestas afirmativas: nivel `0`
- `3 a 5` respuestas afirmativas: nivel `1`
- `6 a 9` respuestas afirmativas: nivel `2`

## Dataset base

- Fuente: [Zenodo](https://zenodo.org/records/13382368)
- DOI: [10.5281/zenodo.13382368](https://doi.org/10.5281/zenodo.13382368)
- Archivo local: `Data_cruda/jordan_igd_sleep_quality.sav`

El flujo principal del repo trabaja con los **9 ítems `igd1` a `igd9`**, ya que son los síntomas directamente vinculados al fenómeno que la plataforma quiere medir.

## Estructura

| Ruta | Propósito |
|------|-----------|
| `Data_cruda/jordan_igd_sleep_quality.sav` | Dataset real base del proyecto. |
| `scripts/preprocess_real_dataset.py` | Lee el `.sav`, convierte los ítems `Yes/No` a `0/1` y genera el CSV final. |
| `data_preprocesada/` | Salida del preprocesamiento y clustering. |
| `models/train_models.py` | Entrena y compara **5 modelos de clasificación multiclase**. |
| `models/train_classification.py` | Alias del flujo principal de clasificación. |
| `models/run_clustering.py` | Ejecuta clustering exploratorio sobre los 9 ítems IGD. |
| `models/common.py` | Utilidades compartidas de carga y preprocesamiento. |

## 5 modelos incluidos

Se comparan cinco enfoques:

1. `logistic_regression`
2. `random_forest`
3. `extra_trees`
4. `gradient_boosting`
5. `knn`

Las métricas reportadas son:

- `accuracy`
- `balanced_accuracy`
- `f1_macro`
- `precision_macro`
- `recall_macro`

La comparación se hace mediante **validación cruzada estratificada**.

## Clustering

Además del aprendizaje supervisado, el repo incluye clustering exploratorio para identificar perfiles de respuesta dentro de los 9 ítems de IGD.

Se evalúan:

- `KMeans`
- `AgglomerativeClustering`

La selección del mejor esquema se apoya en `silhouette score`.

## Entorno virtual

Python recomendado: **3.11 o 3.12**.

```bash
cd /ruta/al/repo
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Flujo de uso

### 1. Preprocesar el dataset jordano

```bash
python scripts/preprocess_real_dataset.py
```

Salida esperada:

- `data_preprocesada/dataset_preprocesado_jordan.csv`
- `data_preprocesada/preprocessing_report_jordan.json`

### 2. Entrenar los 5 modelos de clasificación

```bash
python models/train_models.py ^
  --data data_preprocesada/dataset_preprocesado_jordan.csv ^
  --target igd_label
```

Salida esperada:

- tabla comparativa en consola
- `models/metrics_jordan_classification.json`
- un archivo `.pkl` por modelo

### 3. Ejecutar clustering exploratorio

```bash
python models/run_clustering.py ^
  --data data_preprocesada/dataset_preprocesado_jordan.csv ^
  --target igd_label ^
  --min-k 2 ^
  --max-k 6
```

Salida esperada:

- `models/clustering_report_jordan.json`
- `data_preprocesada/cluster_assignments_jordan.csv`

## Preguntas base del cuestionario

El instrumento se apoya en nueve preguntas tipo `Sí/No` sobre síntomas de gaming disorder. El reporte de preprocesamiento guarda una traducción funcional de cada una en `question_map_es`.

En términos prácticos, el cuestionario pregunta por:

- preocupación constante por volver a jugar
- necesidad de jugar más
- malestar cuando no puede jugar
- incapacidad para reducir el tiempo de juego
- uso del juego para escapar de problemas
- conflictos con otras personas por jugar
- ocultamiento del tiempo de juego
- pérdida de interés en otras actividades
- conflictos con familia, amistades o pareja por causa del juego

## Recomendación metodológica

Este repositorio ya no está orientado a inferir adicción de forma indirecta desde variables débiles. Está orientado a una **plataforma de screening digital**, donde el usuario responde un instrumento breve y el sistema devuelve un nivel de riesgo interpretable.

Ese enfoque es más consistente con los resultados obtenidos y con el uso final esperado del proyecto.
