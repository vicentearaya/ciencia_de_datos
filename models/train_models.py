"""
Entrena y compara 5 modelos de clasificación para predecir el nivel de adicción
(`addiction_class`) usando `data_preprocesada/dataset_preprocesado_clasificacion.csv`.

Uso desde la raíz del repo:
  python models/train_models.py

O desde models/:
  cd models && python train_models.py

Por defecto se usa un subconjunto estratificado (ver `--max-rows`) para no cargar
el millón de filas completo; sube `--max-rows` si quieres métricas más estables.

Opcional: --data, --target, --max-rows, --test-size, --random-state

Salida: tabla comparativa en consola, `metrics.json` y un `.pkl` por modelo.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
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
    parser = argparse.ArgumentParser(
        description="Entrenar 5 modelos de clasificación para nivel de adicción."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data_preprocesada" / "dataset_preprocesado_clasificacion.csv",
        help="CSV preprocesado con features y columna objetivo.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="addiction_class",
        help="Columna objetivo (nivel de adicción).",
    )
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
    parser.add_argument(
        "--max-rows",
        type=int,
        default=200_000,
        help=(
            "Máximo de filas tras muestreo estratificado por la clase objetivo "
            "(mantiene proporciones de addiction_class). "
            "0 = usar todo el CSV en disco."
        ),
    )
    args = parser.parse_args()

    if not args.data.is_file():
        raise FileNotFoundError(f"No existe el archivo de datos: {args.data}")

    df = pd.read_csv(args.data)
    if args.target not in df.columns:
        raise ValueError(f"Columna objetivo '{args.target}' no está en el dataset.")

    X = df.drop(columns=[args.target])
    y = df[args.target]
    n_file = len(y)

    if args.max_rows > 0 and n_file > args.max_rows:
        X, _, y, _ = train_test_split(
            X,
            y,
            train_size=args.max_rows,
            random_state=args.random_state,
            stratify=y if y.nunique() > 1 else None,
        )
        print(
            f"Muestreo estratificado por '{args.target}': "
            f"{len(y):,} filas (de {n_file:,} en el archivo, "
            f"~{100 * len(y) / n_file:.1f}%)."
        )
    else:
        if args.max_rows == 0:
            print(f"Sin límite de filas: usando las {n_file:,} filas del archivo.")
        else:
            print(
                f"Filas en archivo ({n_file:,}) ≤ max-rows ({args.max_rows:,}); "
                "usando todas."
            )

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
            "f1_macro": float(
                f1_score(y_test, y_pred, average="macro", zero_division=0)
            ),
            "f1_weighted": float(
                f1_score(y_test, y_pred, average="weighted", zero_division=0)
            ),
            "classification_report": classification_report(
                y_test, y_pred, zero_division=0
            ),
        }
        joblib.dump(model, out_dir / f"{name}.pkl")

    comparison = pd.DataFrame(
        [
            {
                "modelo": name,
                "accuracy": m["accuracy"],
                "f1_macro": m["f1_macro"],
                "f1_weighted": m["f1_weighted"],
            }
            for name, m in metrics_all.items()
        ]
    ).sort_values("f1_macro", ascending=False)

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_all, f, indent=2, ensure_ascii=False)

    best = comparison.iloc[0]
    print("\n=== Tabla comparativa (ordenada por F1 macro) ===\n")
    print(
        comparison.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )
    print(
        f"\nMejor modelo por F1 macro: {best['modelo']} "
        f"(accuracy={best['accuracy']:.4f}, f1_macro={best['f1_macro']:.4f})."
    )
    print(
        f"\nInformes por clase (classification_report) y demás detalles: {metrics_path}"
    )


if __name__ == "__main__":
    main()
