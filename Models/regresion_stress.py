from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "Dataset" / "dataset_stress_gaming.csv"
MODEL_PATH = ROOT / "Models" / "stress_regression.pkl"
REPORT_PATH = ROOT / "Models" / "stress_regression_metrics.json"

TARGET = "stress_level"
RANDOM_STATE = 42
TEST_SIZE = 0.20
MAX_SVR_TRAIN_SAMPLES = 20_000
MAX_RF_TRAIN_SAMPLES = 60_000
MAX_GB_TRAIN_SAMPLES = 60_000
CV_SPLITS = 5

# Features de gaming (sin variables psicologicas directas para evitar leakage).
GAMING_FEATURES = [
    "daily_gaming_hours",
    "weekly_sessions",
    "years_gaming",
    "multiplayer_ratio",
    "toxic_exposure",
    "violent_games_ratio",
    "mobile_gaming_ratio",
    "night_gaming_ratio",
    "weekend_gaming_hours",
    "friends_gaming_count",
    "online_friends",
    "streaming_hours",
    "esports_interest",
    "headset_usage",
    "microtransactions_spending",
    "competitive_rank",
    "internet_quality",
    "gaming_intensity_index",
    "online_social_ratio",
    "gaming_screen_share",
]


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    fit_sample_size: int | None = None,
) -> dict[str, float]:
    if fit_sample_size is not None and len(X_train) > fit_sample_size:
        X_fit, _, y_fit, _ = train_test_split(
            X_train,
            y_train,
            train_size=fit_sample_size,
            random_state=RANDOM_STATE,
        )
    else:
        X_fit, y_fit = X_train, y_train

    model.fit(X_fit, y_fit)
    pred = model.predict(X_test)

    return {
        "mae": float(mean_absolute_error(y_test, pred)),
        "rmse": rmse(y_test.to_numpy(), pred),
        "r2": float(r2_score(y_test, pred)),
    }


def sample_for_training(
    X_train: pd.DataFrame, y_train: pd.Series, sample_size: int | None
) -> tuple[pd.DataFrame, pd.Series]:
    if sample_size is None or len(X_train) <= sample_size:
        return X_train, y_train

    X_fit, _, y_fit, _ = train_test_split(
        X_train,
        y_train,
        train_size=sample_size,
        random_state=RANDOM_STATE,
    )
    return X_fit, y_fit


def cv_mae(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sample_size: int | None,
) -> tuple[float, float, int]:
    X_cv, y_cv = sample_for_training(X_train, y_train, sample_size)
    cv = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(
        model,
        X_cv,
        y_cv,
        scoring="neg_mean_absolute_error",
        cv=cv,
        n_jobs=1,
    )
    mae_scores = -scores
    return float(mae_scores.mean()), float(mae_scores.std()), int(len(X_cv))


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"No existe el dataset: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    missing_cols = [c for c in GAMING_FEATURES + [TARGET] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Faltan columnas requeridas: {missing_cols}")

    work_df = df[GAMING_FEATURES + [TARGET]].copy()
    work_df = work_df.dropna(subset=[TARGET]).reset_index(drop=True)

    X = work_df[GAMING_FEATURES]
    y = work_df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    models: dict[str, dict[str, Any]] = {
        "linear_regression": {
            "estimator": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", LinearRegression()),
                ]
            ),
            "fit_sample_size": None,
            "cv_sample_size": 40_000,
        },
        "dummy_mean": {
            "estimator": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", DummyRegressor(strategy="mean")),
                ]
            ),
            "fit_sample_size": None,
            "cv_sample_size": 40_000,
        },
        "ridge": {
            "estimator": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", Ridge(alpha=1.0)),
                ]
            ),
            "fit_sample_size": None,
            "cv_sample_size": 40_000,
        },
        "lasso": {
            "estimator": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", Lasso(alpha=0.001, max_iter=5000)),
                ]
            ),
            "fit_sample_size": None,
            "cv_sample_size": 40_000,
        },
        "random_forest": {
            "estimator": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        RandomForestRegressor(
                            n_estimators=120,
                            random_state=RANDOM_STATE,
                            n_jobs=1,
                        ),
                    ),
                ]
            ),
            "fit_sample_size": MAX_RF_TRAIN_SAMPLES,
            "cv_sample_size": 20_000,
        },
        "svr_rbf": {
            "estimator": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", SVR(C=10.0, epsilon=0.1, kernel="rbf")),
                ]
            ),
            "fit_sample_size": MAX_SVR_TRAIN_SAMPLES,
            "cv_sample_size": 5_000,
        },
        "gradient_boosting": {
            "estimator": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", GradientBoostingRegressor(random_state=RANDOM_STATE)),
                ]
            ),
            "fit_sample_size": MAX_GB_TRAIN_SAMPLES,
            "cv_sample_size": 20_000,
        },
    }

    results: list[dict[str, Any]] = []
    trained_estimators: dict[str, Any] = {}

    for model_name, cfg in models.items():
        estimator = cfg["estimator"]
        cv_mean_mae, cv_std_mae, cv_used_rows = cv_mae(
            model=estimator,
            X_train=X_train,
            y_train=y_train,
            sample_size=cfg["cv_sample_size"],
        )
        metrics = evaluate_model(
            model=estimator,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            fit_sample_size=cfg["fit_sample_size"],
        )
        results.append(
            {
                "model": model_name,
                "cv_mae_mean": cv_mean_mae,
                "cv_mae_std": cv_std_mae,
                "cv_rows": cv_used_rows,
                "fit_sample_size": cfg["fit_sample_size"] or int(len(X_train)),
                **metrics,
            }
        )
        trained_estimators[model_name] = estimator

    results_df = pd.DataFrame(results).sort_values("mae", ascending=True).reset_index(drop=True)
    best_row = results_df.iloc[0]
    best_model_name = str(best_row["model"])
    best_model = trained_estimators[best_model_name]

    artifact = {
        "task": "stress_regression_from_gaming_parameters",
        "target": TARGET,
        "features": GAMING_FEATURES,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "best_model_name": best_model_name,
        "metrics_test_sorted_by_mae": results_df.to_dict(orient="records"),
        "model": best_model,
    }

    with MODEL_PATH.open("wb") as f:
        pickle.dump(artifact, f)
    REPORT_PATH.write_text(
        json.dumps(
            {
                "task": artifact["task"],
                "target": artifact["target"],
                "features_count": len(artifact["features"]),
                "best_model_name": artifact["best_model_name"],
                "metrics_test_sorted_by_mae": artifact["metrics_test_sorted_by_mae"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("Comparación de modelos (ordenado por MAE):")
    print(results_df.to_string(index=False))
    print(f"\nMejor modelo: {best_model_name}")
    print(f"Artefacto guardado en: {MODEL_PATH}")
    print(f"Reporte guardado en:  {REPORT_PATH}")


if __name__ == "__main__":
    main()
