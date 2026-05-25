"""
Entrena y compara 5 modelos de clasificacion multiclase para predecir `igd_label`
usando el dataset jordano preprocesado.

Clases:
  0 = Jugador sin indicadores relevantes de trastorno
  1 = Jugador en riesgo de desarrollar problemas por gaming
  2 = Jugador con alta probabilidad de trastorno por gaming
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from common import DEFAULT_DATA, TARGET_COLUMN, build_preprocessor, load_dataset

CLASS_MAP = {
    0: "Jugador sin indicadores relevantes de trastorno",
    1: "Jugador en riesgo de desarrollar problemas por gaming",
    2: "Jugador con alta probabilidad de trastorno por gaming",
}


def build_models(random_state: int = 42) -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=1,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=1,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=3,
            random_state=random_state,
        ),
        "knn": KNeighborsClassifier(
            n_neighbors=11,
            weights="distance",
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entrenar 5 modelos de clasificacion para gaming disorder."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help="CSV preprocesado con features y columna objetivo.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=TARGET_COLUMN,
        help="Columna objetivo multiclase.",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Numero de folds para validacion cruzada estratificada.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
    )
    args = parser.parse_args()

    X, y = load_dataset(args.data, target=args.target)
    preprocessor = build_preprocessor(X)
    cv = StratifiedKFold(
        n_splits=args.cv_splits,
        shuffle=True,
        random_state=args.random_state,
    )

    scoring = {
        "accuracy": make_scorer(accuracy_score),
        "balanced_accuracy": make_scorer(balanced_accuracy_score),
        "f1_macro": make_scorer(f1_score, average="macro"),
        "precision_macro": make_scorer(precision_score, average="macro", zero_division=0),
        "recall_macro": make_scorer(recall_score, average="macro", zero_division=0),
    }

    out_dir = Path(__file__).resolve().parent
    metrics_all: dict[str, dict[str, float | dict[str, str]]] = {}

    for name, classifier in build_models(random_state=args.random_state).items():
        model = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("classifier", classifier),
            ]
        )
        scores = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            n_jobs=None,
        )
        metrics_all[name] = {
            "accuracy_mean": float(scores["test_accuracy"].mean()),
            "accuracy_std": float(scores["test_accuracy"].std()),
            "balanced_accuracy_mean": float(scores["test_balanced_accuracy"].mean()),
            "balanced_accuracy_std": float(scores["test_balanced_accuracy"].std()),
            "f1_macro_mean": float(scores["test_f1_macro"].mean()),
            "f1_macro_std": float(scores["test_f1_macro"].std()),
            "precision_macro_mean": float(scores["test_precision_macro"].mean()),
            "recall_macro_mean": float(scores["test_recall_macro"].mean()),
        }

        model.fit(X, y)
        joblib.dump(model, out_dir / f"{name}_jordan_multiclass.pkl")

    comparison = pd.DataFrame(
        [{"modelo": name, **metrics} for name, metrics in metrics_all.items()]
    ).sort_values(
        ["accuracy_mean", "f1_macro_mean", "balanced_accuracy_mean"],
        ascending=False,
    )

    metrics_path = out_dir / "metrics_jordan_classification.json"
    payload = {
        "dataset": str(args.data),
        "target": args.target,
        "class_map": {str(k): v for k, v in CLASS_MAP.items()},
        "models": metrics_all,
    }
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    best = comparison.iloc[0]
    print("\n=== Tabla comparativa multiclase (ordenada por accuracy) ===\n")
    print(
        comparison.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )
    print(
        f"\nMejor modelo: {best['modelo']} "
        f"(accuracy={best['accuracy_mean']:.4f}, "
        f"balanced_accuracy={best['balanced_accuracy_mean']:.4f}, "
        f"f1_macro={best['f1_macro_mean']:.4f})."
    )
    print(f"\nMétricas detalladas: {metrics_path}")


if __name__ == "__main__":
    main()
