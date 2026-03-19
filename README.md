# Preprocesamiento del Dataset

Este repositorio contiene un script para perfilar y preprocesar el dataset ubicado en [Dataset/gaming_mental_health_10M_40features.csv.gz](c:\Users\mronc\Documents\GitHub\ciencia_de_datos\Dataset\gaming_mental_health_10M_40features.csv.gz).

## Archivo principal

El script principal es [preprocess_dataset.py](c:\Users\mronc\Documents\GitHub\ciencia_de_datos\preprocess_dataset.py).

Su objetivo es:
- inspeccionar la estructura del dataset
- detectar valores inválidos o sospechosos
- generar un reporte de perfilado
- aplicar un preprocesamiento reproducible
- guardar los artefactos resultantes

## Qué se hizo

Se implementó un flujo de trabajo con estas etapas:

1. Carga del dataset desde un archivo `.csv.gz`.
2. Perfilado general:
   - cantidad de filas y columnas
   - nombres de columnas
   - separación de variables categóricas, binarias, ordinales y continuas
   - conteo de valores faltantes
   - perfil numérico con mínimos, máximos, media, desviación estándar y percentiles
   - detección de outliers usando IQR
3. Validación de reglas de negocio para detectar valores imposibles o fuera de rango.
4. Reemplazo de valores inválidos por `NaN`.
5. Tratamiento de outliers mediante winsorización por percentiles.
6. Creación de variables derivadas para análisis posterior.
7. Imputación de valores faltantes numéricos usando la mediana.
8. Codificación `one-hot` de la variable `gender`.
9. Escalado robusto opcional para variables continuas.

## Reglas de validación incluidas

El script valida, entre otras, estas condiciones:
- `sleep_hours` entre `0` y `24`
- `daily_gaming_hours` entre `0` y `24`
- `screen_time_total` entre `0` y `24`
- `bmi` entre `10` y `70`
- variables tipo score en escala `0-10`
- variables ratio en rango `0-1`
- variables ordinales como `stress_level` e `internet_quality` dentro de sus escalas esperadas

Cuando un valor no cumple la regla, se reemplaza por `NaN` y luego puede ser imputado.

## Variables derivadas

El script agrega estas variables cuando las columnas necesarias existen:
- `gaming_intensity_index`
- `online_social_ratio`
- `gaming_screen_share`
- `mental_burden_index`

## Archivos generados

Por defecto, al ejecutar el script se generan:
- `dataset_profile.json`: reporte completo del perfil del dataset
- `preprocessing_log.json`: resumen de las transformaciones aplicadas
- `dataset_preprocessed.csv`: dataset procesado

## Uso

Ejecutar con configuración por defecto:

```powershell
python .\preprocess_dataset.py
```

Ejecutar sobre una muestra para pruebas rápidas:

```powershell
python .\preprocess_dataset.py --sample-rows 100000
```

Ejecutar con escalado robusto:

```powershell
python .\preprocess_dataset.py --scale-continuous
```

Ejecutar sin guardar el CSV procesado:

```powershell
python .\preprocess_dataset.py --skip-save-processed
```

Cambiar nombre o ruta de salida:

```powershell
python .\preprocess_dataset.py --output-csv dataset_preprocessed_full.csv --report-json perfil.json --output-log-json log.json
```

## Notas

- El script no modifica el archivo original.
- El preprocesamiento está orientado a análisis exploratorio y preparación inicial para modelado.
- Si el siguiente paso es entrenar modelos, conviene mover este flujo a un `Pipeline` de `scikit-learn` para evitar fuga de información entre entrenamiento y prueba.
