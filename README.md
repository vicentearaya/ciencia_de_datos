# Predicción De Estrés Basada En Parámetros De Gaming

Repositorio enfocado exclusivamente en **regresión de `stress_level`** a partir de variables de comportamiento en videojuegos.

## Alcance

- Target: `stress_level` (continuo).
- Enfoque: solo variables de gaming.
- Tarea: regresión (no clasificación).

## Estructura

```text
ciencia_de_datos/
├── RAW/
│   └── gaming_mental_health_10M_40features.csv.gz
├── Dataset/
│   └── dataset_stress_gaming.csv
├── Models/
│   ├── regresion_stress.py
│   ├── stress_regression.pkl
│   └── stress_regression_metrics.json
├── requirements.txt
└── README.md
```

## Modelado Implementado

Script principal: `Models/regresion_stress.py`

Modelos comparados:

- `DummyRegressor` (baseline)
- `LinearRegression`
- `Ridge`
- `Lasso`
- `RandomForestRegressor`
- `SVR`
- `GradientBoostingRegressor`

Evaluación:

- Validación cruzada `KFold` en `train`
- Evaluación final en `test`
- Métricas: `MAE`, `RMSE`, `R²`

## Ejecución

```bash
python -m pip install -r requirements.txt
python Models/regresion_stress.py
```

Resultados:

- Artefacto del mejor modelo: `Models/stress_regression.pkl`
- Métricas de comparación: `Models/stress_regression_metrics.json`
