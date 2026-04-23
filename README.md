# Gaming & Mental Health — Análisis de Datos (Tarea 1)

Proyecto de Ciencia de Datos Avanzado aplicando la metodología CRISP-DM. Se analiza la relación entre patrones de comportamiento en videojuegos y salud mental, usando un dataset sintético de 10 millones de registros.

## Pregunta analítica

> ¿Cuáles son los factores de comportamiento de juego (horas, intensidad, contexto social, hábitos) que mejor predicen niveles elevados de ansiedad, depresión y adicción en jugadores?

## Estructura del repositorio

```
ciencia_de_datos/
├── RAW/
│   └── gaming_mental_health_10M_40features.csv.gz   # Dataset original (comprimido)
├── Dataset/
│   └── dataset_preprocessed.csv                     # Dataset limpio (generado al ejecutar el notebook)
├── preprocesamiento.ipynb                            # Notebook principal ejecutable
├── Preprocesamiento_y_analisis.ipynb                 # Notebook alternativo de análisis
└── venv_ds/                                          # Entorno virtual local
```

## Ejecución paso a paso (terminal)

1. Ir a la raíz del proyecto:

```bash
cd /Users/miilwaukee/Documents/ciencia_de_datos/ciencia_de_datos
```

2. Activar el entorno virtual:

```bash
source venv_ds/bin/activate
```

3. Ejecutar todo el notebook `preprocesamiento.ipynb` de forma automática:

```bash
jupyter nbconvert --to notebook --execute preprocesamiento.ipynb --output preprocesamiento.ipynb --ExecutePreprocessor.timeout=1800
```

4. Verificar resultado esperado en consola:

```text
[NbConvertApp] Converting notebook preprocesamiento.ipynb to notebook
[NbConvertApp] Writing ... bytes to preprocesamiento.ipynb
```

5. Revisar artefactos generados:
- Notebook ejecutado con outputs: `preprocesamiento.ipynb`
- Dataset procesado: `Dataset/dataset_preprocessed.csv`

## Notas de entorno

- Si estás en una red restringida (por ejemplo, red universitaria), Git puede fallar en `443`; en ese caso usa otra red o revisa configuración de proxy.
- Si en entornos aislados aparece error de permisos de Jupyter (`~/.jupyter`), define `JUPYTER_CONFIG_DIR` y `JUPYTER_DATA_DIR` dentro del proyecto antes de ejecutar.

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
