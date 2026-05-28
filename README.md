# Clasificación multiclase de riesgo de trastorno por gaming con cuestionario genérico

Este repositorio usa el dataset real jordano [Internet Gaming Disorder and Sleep Quality among Jordanian University Students](https://zenodo.org/records/13382368) para entrenar un modelo **generic_hybrid**: una combinación entre las 9 preguntas base de síntomas de gaming disorder y un bloque corto de hábitos de juego, sueño y bienestar.

## Objetivo del proyecto

El objetivo es construir una base metodológica para una plataforma donde un usuario responda un cuestionario breve y reciba una clasificación interpretable de riesgo.

La variable objetivo es `igd_label`, con tres clases:

- `0`: **Jugador sin indicadores relevantes de trastorno**
- `1`: **Jugador en riesgo de desarrollar problemas por gaming**
- `2`: **Jugador con alta probabilidad de trastorno por gaming**

## Enfoque `generic_hybrid`

El flujo principal del repo trabaja con dos bloques:

- **Bloque 1**: 9 preguntas base sobre síntomas de gaming disorder (`igd1` a `igd9`)
- **Bloque 2**: variables genéricas y reutilizables sobre juego, redes sociales, sueño y bienestar

Se eliminaron campos demasiado específicos para una encuesta genérica, como:

- edad exacta
- ciudad
- región
- universidad
- lugar donde vive
- ingreso familiar
- carrera específica

## Dataset base

- Fuente: [Zenodo](https://zenodo.org/records/13382368)
- DOI: [10.5281/zenodo.13382368](https://doi.org/10.5281/zenodo.13382368)
- Archivo local: `Data_cruda/jordan_igd_sleep_quality.sav`

## Estructura

| Ruta | Propósito |
|------|-----------|
| `Data_cruda/jordan_igd_sleep_quality.sav` | Dataset real base del proyecto. |
| `scripts/preprocess_real_dataset.py` | Genera el dataset `generic_hybrid` listo para modelado. |
| `data_preprocesada/dataset_preprocesado_jordan_generic_hybrid.csv` | Dataset final para entrenamiento. |
| `data_preprocesada/preprocessing_report_jordan_generic_hybrid.json` | Reporte del preprocesamiento. |
| `models/train_models.py` | Entrena y compara 5 modelos de clasificación multiclase. |
| `models/train_classification.py` | Alias del flujo principal. |
| `models/run_clustering.py` | Ejecuta clustering exploratorio sobre el dataset `generic_hybrid`. |
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

## Cuestionario recomendado para la plataforma

### Bloque 1: síntomas de gaming

Usar respuestas cerradas `Sí / No`.

1. Pienso constantemente en cuándo podré volver a jugar.
2. Siento que jugar más tiempo me dejaría más satisfecho.
3. Me siento mal cuando no puedo jugar.
4. Me cuesta reducir el tiempo que paso jugando.
5. Uso los videojuegos para evitar pensar en problemas o emociones desagradables.
6. He tenido conflictos con otras personas por mi forma de jugar.
7. Oculto cuánto tiempo paso jugando.
8. He perdido interés en otras actividades por jugar.
9. He tenido problemas con familia, amistades o pareja por los videojuegos.

### Bloque 2: hábitos de juego, sueño y bienestar

Usar respuestas cerradas, no texto libre.

10. ¿Cuántas horas juegas en una semana típica?
- Menos de 1 hora por día
- Entre 1 y 3 horas por día
- Más de 3 horas por día

11. ¿Cuántas horas al día usas redes sociales?
- Entre 0 y 2 horas
- Entre 3 y 4 horas
- Más de 4 horas

12. ¿Cuál es tu principal motivo para usar internet?
- Videojuegos
- Estudiar
- Trabajo
- Redes sociales
- Más de un motivo
- Otro

13. ¿Cuánto tardas normalmente en quedarte dormido?
- Menos de 15 minutos
- Entre 15 y 30 minutos
- Entre 31 y 60 minutos
- Más de 60 minutos

14. ¿Cuántas horas duermes normalmente por noche?
- Menos de 5 horas
- Entre 5 y 6 horas
- Entre 6 y 7 horas
- Entre 7 y 8 horas
- Más de 8 horas

15. ¿Cómo evaluarías tu calidad general de sueño?
- Muy buena
- Bastante buena
- Bastante mala
- Muy mala

16. ¿Con qué frecuencia has necesitado ayuda o medicación para dormir durante el último mes?
- Nunca
- Menos de una vez por semana
- Una o dos veces por semana
- Tres o más veces por semana

17. ¿Con qué frecuencia te cuesta mantenerte despierto durante el día?
- Nunca
- Menos de una vez por semana
- Una o dos veces por semana
- Tres o más veces por semana

18. ¿Con qué frecuencia te cuesta mantener el entusiasmo para hacer tus actividades?
- Nunca
- Menos de una vez por semana
- Una o dos veces por semana
- Tres o más veces por semana

## Resultados principales del enfoque `generic_hybrid`

Las métricas del entrenamiento quedaron en:

- `models/metrics_jordan_generic_hybrid_classification.json`

Resumen:

- `logistic_regression`: `accuracy = 1.0000`
- `gradient_boosting`: `accuracy = 0.9701`
- `extra_trees`: `accuracy = 0.9586`
- `random_forest`: `accuracy = 0.9342`
- `knn`: `accuracy = 0.8713`

## Clustering

Además del aprendizaje supervisado, el repo incluye clustering exploratorio para detectar perfiles de respuesta dentro del cuestionario `generic_hybrid`.

## Flujo de uso

### 1. Preprocesar

```bash
python scripts/preprocess_real_dataset.py
```

Salida esperada:

- `data_preprocesada/dataset_preprocesado_jordan_generic_hybrid.csv`
- `data_preprocesada/preprocessing_report_jordan_generic_hybrid.json`

### 2. Entrenar los 5 modelos

```bash
python models/train_models.py
```

Salida esperada:

- tabla comparativa en consola
- `models/metrics_jordan_generic_hybrid_classification.json`
- un `.pkl` por modelo

### 3. Ejecutar clustering

```bash
python models/run_clustering.py
```

Salida esperada:

- `models/clustering_report_jordan_generic_hybrid.json`
- `data_preprocesada/cluster_assignments_jordan_generic_hybrid.csv`

## Recomendación metodológica

Este repositorio queda orientado a una **plataforma de screening digital** con un cuestionario suficientemente genérico para reutilizarlo fuera del contexto jordano, pero manteniendo un desempeño alto porque conserva las 9 preguntas centrales del trastorno por gaming.
