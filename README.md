# Gaming & Mental Health — Análisis de Datos (Tarea 1)

Proyecto de Ciencia de Datos Avanzado aplicando la metodología CRISP-DM. Se analiza la relación entre patrones de comportamiento en videojuegos y salud mental, usando un dataset sintético de 10 millones de registros.

## Pregunta analítica

> ¿Cuáles son los factores de comportamiento de juego (horas, intensidad, contexto social, hábitos) que mejor predicen niveles elevados de ansiedad, depresión y adicción en jugadores?

## Estructura del repositorio

```
ciencia_de_datos/
├── RAW/
│   └── gaming_mental_health_10M_40features.csv.gz   # Dataset original (comprimido)
├── processed/
│   └── dataset_preprocessed.csv                     # Dataset limpio (generado al ejecutar el notebook)
└── Preprocesamiento_y_analisis.ipynb                 # Notebook principal
```

## Cómo ejecutar

1. Instalar dependencias:

```bash
pip install pandas numpy matplotlib scikit-learn
```

2. Abrir y ejecutar el notebook de forma lineal:

```
Preprocesamiento_y_analisis.ipynb
```

El notebook lee el dataset desde `RAW/` y guarda el resultado procesado en `processed/dataset_preprocessed.csv`.

## Contenido del notebook

| Sección | Contenido |
|---------|-----------|
| 0 | Contexto, problemática, objetivos SMART y KPIs |
| 1–2 | Librerías, configuración y carga del dataset |
| 3–4 | Revisión inicial: tipos, missing, estadísticos, cardinalidad |
| 5–6 | Reglas de validación y limpieza de valores inválidos |
| 7 | Tratamiento de outliers (winsorización + boxplots) |
| 8–9 | Variables derivadas, imputación y codificación |
| 10 | Visualizaciones EDA |
| 12 | Data Quality Report estructurado |
| 13 | Arquitectura del pipeline (diagrama Mermaid) |
