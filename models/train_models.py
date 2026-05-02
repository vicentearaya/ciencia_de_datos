"""
Entrena y compara 5 modelos de clasificación sobre datos preprocesados.

Uso (tras generar un CSV en data_preprocesada/):
  python train_models.py --data ../data_preprocesada/dataset.csv --target columna_objetivo

Los artefactos (.pkl, métricas) se ignoran por git; ajusta rutas según tu CSV real.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent


def build_models(random_state: int = 42) -> dict[str, Pipeline]:
    """Cinco clasificadores con escalado cuando aplica."""
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(max_iter=2000, random_state=random_state),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=200, random_state=random_state, n_jobs=-1
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                (
                    "clf",
                    GradientBoostingClassifier(random_state=random_state),
                ),
            ]
        ),
        "svc_rbf": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SVC(kernel="rbf", random_state=random_state),
                ),
            ]
        ),
        "knn": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=5)),
            ]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrenar 5 modelos de clasificación.")
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data_preprocesada" / "dataset.csv",
        help="CSV preprocesado con features y columna objetivo.",
    )
    parser.add_argument("--target", type=str, required=True, help="Nombre de la columna y.")
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fracción para conjunto de prueba.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
    )
    args = parser.parse_args()

    if not args.data.is_file():
        raise FileNotFoundError(f"No existe el archivo de datos: {args.data}")

    df = pd.read_csv(args.data)
    if args.target not in df.columns:
        raise ValueError(f"Columna objetivo '{args.target}' no está en el dataset.")

    X = df.drop(columns=[args.target])
    y = df[args.target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y if y.nunique() > 1 else None,
    )

    out_dir = Path(__file__).resolve().parent
    metrics_all: dict[str, dict] = {}

    for name, model in build_models(random_state=args.random_state).items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics_all[name] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
            "classification_report": classification_report(
                y_test, y_pred, zero_division=0
            ),
        }
        joblib.dump(model, out_dir / f"{name}.pkl")

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_all, f, indent=2, ensure_ascii=False)

    print(f"Métricas guardadas en {metrics_path}")
    for name, m in metrics_all.items():
        print(f"{name}: accuracy={m['accuracy']:.4f} f1_macro={m['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
