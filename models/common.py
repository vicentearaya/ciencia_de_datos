from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "data_preprocesada" / "dataset_preprocesado_jordan.csv"
TARGET_COLUMN = "igd_label"
NON_FEATURE_COLUMNS = {"igd_label_name", "yes_count"}


def load_dataset(path: Path, target: str = TARGET_COLUMN) -> tuple[pd.DataFrame, pd.Series]:
    if not path.is_file():
        raise FileNotFoundError(f"No existe el archivo de datos: {path}")

    df = pd.read_csv(path)
    if target not in df.columns:
        raise ValueError(f"La columna objetivo '{target}' no está en el dataset.")

    drop_columns = [target] + [col for col in NON_FEATURE_COLUMNS if col in df.columns]
    X = df.drop(columns=drop_columns)
    y = df[target]
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, list(X.columns)),
        ],
        sparse_threshold=0.0,
    )
