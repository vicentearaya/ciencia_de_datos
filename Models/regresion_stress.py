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
DATA_PATH = ROOT / "Dataset" / "data_preprocesada.csv"
MODEL_PATH = ROOT / "Models" / "stress_regression.pkl"
REPORT_PATH = ROOT / "Models" / "stress_regression_metrics.json"

TARGET = "stress_level"
RANDOM_STATE = 42
TEST_SIZE = 0.20
MAX_SVR_TRAIN_SAMPLES = 20_000
MAX_RF_TRAIN_SAMPLES = 60_000
MAX_GB_TRAIN_SAMPLES = 60_000
CV_SPLITS = 5
PARTITION_COL = "partition"
TRAIN_PARTITION = "train"
TEST_PARTITION = "test"
ALL_NUMERIC_SCENARIO = "all_numeric_no_leakage"
GAMING_ONLY_SCENARIO = "gaming_only"

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


def get_model_configs() -> dict[str, dict[str, Any]]:
    return {
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


def evaluate_scenario(
    scenario_name: str,
    features: list[str],
    work_df: pd.DataFrame,
) -> dict[str, Any]:
    used_partition_split = False
    if PARTITION_COL in work_df.columns:
        partition_series = (
            work_df[PARTITION_COL].astype(str).str.strip().str.lower()
        )
        train_mask = partition_series == TRAIN_PARTITION
        test_mask = partition_series == TEST_PARTITION
        if train_mask.any() and test_mask.any():
            used_partition_split = True
            X_train = work_df.loc[train_mask, features]
            y_train = work_df.loc[train_mask, TARGET]
            X_test = work_df.loc[test_mask, features]
            y_test = work_df.loc[test_mask, TARGET]
        else:
            X = work_df[features]
            y = work_df[TARGET]
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE,
            )
    else:
        X = work_df[features]
        y = work_df[TARGET]
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )

    models = get_model_configs()
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

    results_df = (
        pd.DataFrame(results).sort_values("mae", ascending=True).reset_index(drop=True)
    )
    best_row = results_df.iloc[0]
    best_model_name = str(best_row["model"])
    best_model = trained_estimators[best_model_name]

    return {
        "scenario_name": scenario_name,
        "features_used": features,
        "features_count": len(features),
        "used_partition_split": used_partition_split,
        "best_model_name": best_model_name,
        "best_metrics": {
            "mae": float(best_row["mae"]),
            "rmse": float(best_row["rmse"]),
            "r2": float(best_row["r2"]),
        },
        "metrics_test_sorted_by_mae": results_df.to_dict(orient="records"),
        "model": best_model,
    }


def build_diagnostic(scenarios: dict[str, dict[str, Any]]) -> dict[str, Any]:
    gaming = scenarios[GAMING_ONLY_SCENARIO]["best_metrics"]
    all_numeric = scenarios[ALL_NUMERIC_SCENARIO]["best_metrics"]
    mae_improvement = gaming["mae"] - all_numeric["mae"]
    r2_improvement = all_numeric["r2"] - gaming["r2"]

    if all_numeric["r2"] <= 0.02 and abs(mae_improvement) < 0.03:
        conclusion = (
            "target_dificil_o_ruidoso: incluso usando mas features numericas, "
            "el desempeno se mantiene cercano al baseline."
        )
    elif mae_improvement >= 0.05 or r2_improvement >= 0.05:
        conclusion = (
            "features_insuficientes_en_gaming_only: al ampliar features mejora "
            "de forma relevante."
        )
    else:
        conclusion = (
            "mixto: hay mejora leve con mas features, pero el target sigue siendo "
            "complejo para predecir con alta precision."
        )

    return {
        "mae_improvement_all_numeric_vs_gaming": mae_improvement,
        "r2_improvement_all_numeric_vs_gaming": r2_improvement,
        "conclusion": conclusion,
    }


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"No existe el dataset: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    available_gaming_features = [c for c in GAMING_FEATURES if c in df.columns]
    missing_gaming_features = sorted(set(GAMING_FEATURES) - set(available_gaming_features))
    if not available_gaming_features:
        raise ValueError(
            "No hay features de gaming disponibles en el dataset preprocesado."
        )
    if TARGET not in df.columns:
        raise ValueError(f"No existe la columna objetivo '{TARGET}' en {DATA_PATH}")

    leak_prone_cols = {
        TARGET,
        PARTITION_COL,
        "addiction_class",
        "addiction_level_original",
    }
    numeric_candidates = [
        c for c in df.select_dtypes(include=[np.number]).columns if c not in leak_prone_cols
    ]
    if not numeric_candidates:
        raise ValueError("No se encontraron features numericas para el escenario ampliado.")

    selected_cols = sorted(set(available_gaming_features + numeric_candidates + [TARGET]))
    if PARTITION_COL in df.columns:
        selected_cols.append(PARTITION_COL)
    work_df = df[sorted(set(selected_cols))].copy()
    work_df = work_df.dropna(subset=[TARGET]).reset_index(drop=True)
    scenarios = {
        GAMING_ONLY_SCENARIO: evaluate_scenario(
            scenario_name=GAMING_ONLY_SCENARIO,
            features=available_gaming_features,
            work_df=work_df,
        ),
        ALL_NUMERIC_SCENARIO: evaluate_scenario(
            scenario_name=ALL_NUMERIC_SCENARIO,
            features=numeric_candidates,
            work_df=work_df,
        ),
    }
    diagnostic = build_diagnostic(scenarios)

    # Mantener compatibilidad: el modelo principal guardado es el mejor del escenario gaming_only.
    best_model_name = scenarios[GAMING_ONLY_SCENARIO]["best_model_name"]
    best_model = scenarios[GAMING_ONLY_SCENARIO]["model"]
    results_df = pd.DataFrame(
        scenarios[GAMING_ONLY_SCENARIO]["metrics_test_sorted_by_mae"]
    )

    artifact = {
        "task": "stress_regression_from_gaming_parameters",
        "target": TARGET,
        "features": available_gaming_features,
        "missing_features": missing_gaming_features,
        "dataset_path": str(DATA_PATH),
        "used_partition_split": scenarios[GAMING_ONLY_SCENARIO]["used_partition_split"],
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "best_model_name": best_model_name,
        "metrics_test_sorted_by_mae": results_df.to_dict(orient="records"),
        "scenario_comparison": {
            k: {
                kk: vv
                for kk, vv in v.items()
                if kk != "model"
            }
            for k, v in scenarios.items()
        },
        "diagnostic": diagnostic,
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
                "features_used": artifact["features"],
                "missing_features": artifact["missing_features"],
                "dataset_path": artifact["dataset_path"],
                "used_partition_split": artifact["used_partition_split"],
                "best_model_name": artifact["best_model_name"],
                "metrics_test_sorted_by_mae": artifact["metrics_test_sorted_by_mae"],
                "scenario_comparison": artifact["scenario_comparison"],
                "diagnostic": artifact["diagnostic"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("Comparación de modelos (ordenado por MAE) - escenario gaming_only:")
    print(pd.DataFrame(scenarios[GAMING_ONLY_SCENARIO]["metrics_test_sorted_by_mae"]).to_string(index=False))
    print("\nComparación de modelos (ordenado por MAE) - escenario all_numeric_no_leakage:")
    print(pd.DataFrame(scenarios[ALL_NUMERIC_SCENARIO]["metrics_test_sorted_by_mae"]).to_string(index=False))
    print(f"\nDataset usado: {DATA_PATH}")
    print(
        "Split usado: "
        + (
            "columna partition (train/test)"
            if scenarios[GAMING_ONLY_SCENARIO]["used_partition_split"]
            else "train_test_split aleatorio"
        )
    )
    if missing_gaming_features:
        print(f"Features de gaming no encontradas y omitidas: {missing_gaming_features}")
    print(f"\nDiagnóstico: {diagnostic['conclusion']}")
    print(
        "ΔMAE (all_numeric vs gaming_only): "
        f"{diagnostic['mae_improvement_all_numeric_vs_gaming']:.6f}"
    )
    print(
        "ΔR2 (all_numeric vs gaming_only): "
        f"{diagnostic['r2_improvement_all_numeric_vs_gaming']:.6f}"
    )
    print(f"\nMejor modelo: {best_model_name}")
    print(f"Artefacto guardado en: {MODEL_PATH}")
    print(f"Reporte guardado en:  {REPORT_PATH}")


if __name__ == "__main__":
    main()
