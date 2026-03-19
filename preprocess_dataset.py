import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


CATEGORICAL_COLUMNS = ["gender"]
BINARY_COLUMNS = ["headset_usage"]
ORDINAL_COLUMNS = [
    "stress_level",
    "esports_interest",
    "parental_supervision",
    "internet_quality",
]
LIKERT_COLUMNS = [
    "anxiety_score",
    "depression_score",
    "social_interaction_score",
    "relationship_satisfaction",
    "addiction_level",
    "loneliness_score",
    "aggression_score",
    "happiness_score",
    "eye_strain_score",
    "back_pain_score",
]
PERCENT_COLUMNS = [
    "academic_performance",
    "work_productivity",
]
RATIO_COLUMNS = [
    "multiplayer_ratio",
    "toxic_exposure",
    "violent_games_ratio",
    "mobile_gaming_ratio",
    "night_gaming_ratio",
]
POSITIVE_COLUMNS = [
    "daily_gaming_hours",
    "weekly_sessions",
    "years_gaming",
    "caffeine_intake",
    "exercise_hours",
    "weekend_gaming_hours",
    "friends_gaming_count",
    "online_friends",
    "streaming_hours",
    "microtransactions_spending",
    "screen_time_total",
    "competitive_rank",
]

INVALID_RULES = {
    "age": ("between", 13, 100),
    "income": ("between", 0, 1_000_000),
    "daily_gaming_hours": ("between", 0, 24),
    "weekly_sessions": ("between", 0, 7 * 24),
    "years_gaming": ("between", 0, 80),
    "sleep_hours": ("between", 0, 24),
    "caffeine_intake": ("between", 0, 50),
    "exercise_hours": ("between", 0, 24),
    "stress_level": ("between", 1, 10),
    "anxiety_score": ("between", 0, 10),
    "depression_score": ("between", 0, 10),
    "social_interaction_score": ("between", 0, 10),
    "relationship_satisfaction": ("between", 0, 10),
    "academic_performance": ("between", 0, 100),
    "work_productivity": ("between", 0, 100),
    "addiction_level": ("between", 0, 10),
    "multiplayer_ratio": ("between", 0, 1),
    "toxic_exposure": ("between", 0, 1),
    "violent_games_ratio": ("between", 0, 1),
    "mobile_gaming_ratio": ("between", 0, 1),
    "night_gaming_ratio": ("between", 0, 1),
    "weekend_gaming_hours": ("between", 0, 72),
    "friends_gaming_count": ("between", 0, 10_000),
    "online_friends": ("between", 0, 100_000),
    "streaming_hours": ("between", 0, 24),
    "esports_interest": ("between", 0, 10),
    "headset_usage": ("between", 0, 1),
    "microtransactions_spending": ("between", 0, 1_000_000),
    "parental_supervision": ("between", 0, 10),
    "loneliness_score": ("between", 0, 10),
    "aggression_score": ("between", 0, 10),
    "happiness_score": ("between", 0, 10),
    "bmi": ("between", 10, 70),
    "screen_time_total": ("between", 0, 24),
    "eye_strain_score": ("between", 0, 10),
    "back_pain_score": ("between", 0, 10),
    "competitive_rank": ("between", 0, 100),
    "internet_quality": ("between", 1, 10),
}

SKEWED_COLUMNS = [
    "microtransactions_spending",
    "screen_time_total",
    "daily_gaming_hours",
    "weekend_gaming_hours",
    "caffeine_intake",
    "exercise_hours",
]


def classify_columns(df: pd.DataFrame) -> dict:
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    continuous_columns = [
        col
        for col in numeric_columns
        if col not in BINARY_COLUMNS and col not in ORDINAL_COLUMNS
    ]
    return {
        "categorical": [col for col in CATEGORICAL_COLUMNS if col in df.columns],
        "binary": [col for col in BINARY_COLUMNS if col in df.columns],
        "ordinal": [col for col in ORDINAL_COLUMNS if col in df.columns],
        "continuous": continuous_columns,
    }


def detect_invalid_values(df: pd.DataFrame) -> dict:
    invalid_summary = {}
    for column, rule in INVALID_RULES.items():
        if column not in df.columns:
            continue
        rule_type, lower, upper = rule
        if rule_type != "between":
            continue
        mask = df[column].notna() & ((df[column] < lower) | (df[column] > upper))
        invalid_summary[column] = {
            "count": int(mask.sum()),
            "lower_bound": lower,
            "upper_bound": upper,
        }
    return invalid_summary


def numeric_profile(df: pd.DataFrame) -> dict:
    profile = {}
    numeric_df = df.select_dtypes(include=[np.number])
    for column in numeric_df.columns:
        series = numeric_df[column]
        quantiles = series.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        q1 = quantiles.loc[0.25]
        q3 = quantiles.loc[0.75]
        iqr = q3 - q1
        lower_iqr = q1 - 1.5 * iqr
        upper_iqr = q3 + 1.5 * iqr
        outlier_mask = series.notna() & ((series < lower_iqr) | (series > upper_iqr))
        profile[column] = {
            "dtype": str(series.dtype),
            "missing": int(series.isna().sum()),
            "min": round(float(series.min()), 4),
            "p01": round(float(quantiles.loc[0.01]), 4),
            "p05": round(float(quantiles.loc[0.05]), 4),
            "p25": round(float(q1), 4),
            "p50": round(float(quantiles.loc[0.5]), 4),
            "p75": round(float(q3), 4),
            "p95": round(float(quantiles.loc[0.95]), 4),
            "p99": round(float(quantiles.loc[0.99]), 4),
            "max": round(float(series.max()), 4),
            "mean": round(float(series.mean()), 4),
            "std": round(float(series.std()), 4),
            "outliers_iqr": int(outlier_mask.sum()),
        }
    return profile


def categorical_profile(df: pd.DataFrame) -> dict:
    profile = {}
    for column in CATEGORICAL_COLUMNS:
        if column not in df.columns:
            continue
        counts = df[column].value_counts(dropna=False).to_dict()
        profile[column] = {
            "missing": int(df[column].isna().sum()),
            "unique": int(df[column].nunique(dropna=False)),
            "top_values": {str(k): int(v) for k, v in counts.items()},
        }
    return profile


def build_report(df: pd.DataFrame) -> dict:
    return {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": df.columns.tolist(),
        "column_groups": classify_columns(df),
        "missing_values": {column: int(value) for column, value in df.isna().sum().items()},
        "invalid_values": detect_invalid_values(df),
        "numeric_profile": numeric_profile(df),
        "categorical_profile": categorical_profile(df),
    }


def replace_invalid_with_nan(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    cleaned = df.copy()
    replacements = {}
    for column, rule in INVALID_RULES.items():
        if column not in cleaned.columns:
            continue
        _, lower, upper = rule
        mask = cleaned[column].notna() & (
            (cleaned[column] < lower) | (cleaned[column] > upper)
        )
        replacements[column] = int(mask.sum())
        cleaned.loc[mask, column] = np.nan
    return cleaned, replacements


def winsorize_columns(
    df: pd.DataFrame,
    columns: list[str],
    lower_quantile: float,
    upper_quantile: float,
) -> tuple[pd.DataFrame, dict]:
    transformed = df.copy()
    summary = {}
    for column in columns:
        if column not in transformed.columns:
            continue
        lower = transformed[column].quantile(lower_quantile)
        upper = transformed[column].quantile(upper_quantile)
        before_lower = int((transformed[column] < lower).sum())
        before_upper = int((transformed[column] > upper).sum())
        transformed[column] = transformed[column].clip(lower=lower, upper=upper)
        summary[column] = {
            "lower_clip": round(float(lower), 4),
            "upper_clip": round(float(upper), 4),
            "values_clipped_low": before_lower,
            "values_clipped_high": before_upper,
        }
    return transformed, summary


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    derived = df.copy()
    if {"daily_gaming_hours", "weekend_gaming_hours"}.issubset(derived.columns):
        derived["gaming_intensity_index"] = (
            derived["daily_gaming_hours"].fillna(0) * 5 + derived["weekend_gaming_hours"].fillna(0)
        ) / 7
    if {"online_friends", "friends_gaming_count"}.issubset(derived.columns):
        derived["online_social_ratio"] = derived["online_friends"] / (
            derived["friends_gaming_count"] + 1
        )
    if {"screen_time_total", "daily_gaming_hours"}.issubset(derived.columns):
        derived["gaming_screen_share"] = derived["daily_gaming_hours"] / (
            derived["screen_time_total"] + 1e-6
        )
    if {"anxiety_score", "depression_score", "stress_level"}.issubset(derived.columns):
        derived["mental_burden_index"] = (
            derived["anxiety_score"] + derived["depression_score"] + derived["stress_level"]
        ) / 3
    return derived


def impute_numeric_median(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    imputed = df.copy()
    numeric_columns = imputed.select_dtypes(include=[np.number]).columns
    medians = {}
    for column in numeric_columns:
        missing = int(imputed[column].isna().sum())
        if missing > 0:
            median = imputed[column].median()
            medians[column] = round(float(median), 4)
            imputed[column] = imputed[column].fillna(median)
    return imputed, medians


def one_hot_encode_gender(df: pd.DataFrame) -> pd.DataFrame:
    if "gender" not in df.columns:
        return df
    return pd.get_dummies(df, columns=["gender"], prefix="gender", dtype=int)


def scale_continuous_columns(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, dict]:
    scaled = df.copy()
    scale_summary = {}
    for column in columns:
        if column not in scaled.columns:
            continue
        series = scaled[column]
        median = series.median()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0 or pd.isna(iqr):
            continue
        scaled[column] = (series - median) / iqr
        scale_summary[column] = {
            "median": round(float(median), 4),
            "iqr": round(float(iqr), 4),
        }
    return scaled, scale_summary


def preprocess_dataframe(
    df: pd.DataFrame,
    clip_lower: float,
    clip_upper: float,
    encode_gender: bool,
    scale_continuous: bool,
) -> tuple[pd.DataFrame, dict]:
    working = df.copy()
    log = {}

    working, invalid_replacements = replace_invalid_with_nan(working)
    log["invalid_values_replaced_with_nan"] = invalid_replacements

    working, clip_summary = winsorize_columns(
        working, [col for col in SKEWED_COLUMNS if col in working.columns], clip_lower, clip_upper
    )
    log["winsorization"] = clip_summary

    working = add_derived_features(working)
    log["derived_features"] = [
        col
        for col in [
            "gaming_intensity_index",
            "online_social_ratio",
            "gaming_screen_share",
            "mental_burden_index",
        ]
        if col in working.columns
    ]

    working, imputation_summary = impute_numeric_median(working)
    log["median_imputation"] = imputation_summary

    if encode_gender:
        working = one_hot_encode_gender(working)
        log["gender_encoding"] = "one_hot"

    if scale_continuous:
        continuous_columns = classify_columns(working)["continuous"]
        working, scale_summary = scale_continuous_columns(working, continuous_columns)
        log["continuous_scaling"] = scale_summary

    log["final_shape"] = {"rows": int(working.shape[0]), "columns": int(working.shape[1])}
    return working, log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Perfila y preprocesa el dataset de salud mental y gaming."
    )
    parser.add_argument(
        "--input",
        default="Dataset/gaming_mental_health_10M_40features.csv.gz",
        help="Ruta al dataset de entrada.",
    )
    parser.add_argument(
        "--report-json",
        default="dataset_profile.json",
        help="Ruta de salida del reporte JSON.",
    )
    parser.add_argument(
        "--output-csv",
        default="dataset_preprocessed.csv",
        help="Ruta de salida del dataset procesado.",
    )
    parser.add_argument(
        "--output-log-json",
        default="preprocessing_log.json",
        help="Ruta de salida del log de preprocesamiento.",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help="Lee solo N filas si quieres una corrida rápida.",
    )
    parser.add_argument(
        "--skip-save-processed",
        action="store_true",
        help="Si se indica, no guarda el CSV preprocesado.",
    )
    parser.add_argument(
        "--skip-encode-gender",
        action="store_true",
        help="Si se indica, no aplica one-hot encoding a gender.",
    )
    parser.add_argument(
        "--scale-continuous",
        action="store_true",
        help="Si se indica, aplica robust scaling a variables continuas.",
    )
    parser.add_argument(
        "--clip-lower",
        type=float,
        default=0.01,
        help="Percentil inferior para winsorizacion.",
    )
    parser.add_argument(
        "--clip-upper",
        type=float,
        default=0.99,
        help="Percentil superior para winsorizacion.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"No existe el archivo de entrada: {input_path}")

    df = pd.read_csv(input_path, nrows=args.sample_rows)

    report = build_report(df)
    Path(args.report_json).write_text(json.dumps(report, indent=2), encoding="utf-8")

    processed_df, preprocess_log = preprocess_dataframe(
        df=df,
        clip_lower=args.clip_lower,
        clip_upper=args.clip_upper,
        encode_gender=not args.skip_encode_gender,
        scale_continuous=args.scale_continuous,
    )
    Path(args.output_log_json).write_text(
        json.dumps(preprocess_log, indent=2), encoding="utf-8"
    )

    if not args.skip_save_processed:
        processed_df.to_csv(args.output_csv, index=False)

    print(f"Input shape: {df.shape}")
    print(f"Processed shape: {processed_df.shape}")
    print(f"Profile report: {args.report_json}")
    print(f"Preprocessing log: {args.output_log_json}")
    if not args.skip_save_processed:
        print(f"Processed dataset: {args.output_csv}")


if __name__ == "__main__":
    main()
