from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pyreadstat

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "Data_cruda" / "jordan_igd_sleep_quality.sav"
DEFAULT_OUTPUT = ROOT / "data_preprocesada" / "dataset_preprocesado_jordan_generic_hybrid.csv"
DEFAULT_REPORT = ROOT / "data_preprocesada" / "preprocessing_report_jordan_generic_hybrid.json"
TARGET_COLUMN = "igd_label"
SCREENING_COLUMNS = [f"igd{i}" for i in range(1, 10)]
NOMINAL_COLUMNS = [
    "Gender",
    "city",
    "region",
    "University",
    "Major",
    "livingplace",
    "usagecause",
    "sleepQUAL",
]
GENERIC_NOMINAL_COLUMNS = [
    "usagecause",
    "sleepQUAL",
]
GENERIC_ORDINAL_COLUMNS = [
    "gamingsHgroupsnew",
    "newSHgroups",
    "psqi6",
    "psqi7",
    "psqi8",
    "psqi9",
]
GENERIC_CONTINUOUS_COLUMNS = [
    "gameinghourspermonth",
    "hoursonsocialmedia",
    "psqi2",
    "psqi4",
    "globalscorepsqi",
]
ORDINAL_COLUMNS = [
    "agegroups",
    "grade",
    "monthlybalance",
    "gamingsHgroupsnew",
    "newSHgroups",
    "psqi6",
    "psqi7",
    "psqi8",
    "psqi9",
]
CONTINUOUS_COLUMNS = [
    "age",
    "gameinghourspermonth",
    "hoursonsocialmedia",
    "psqi2",
    "psqi4",
    "globalscorepsqi",
]

IGD_QUESTIONS_ES = {
    "igd1": "Piensa constantemente en cuando podra volver a jugar en internet.",
    "igd2": "Siente que jugar mas tiempo le dejaria mas satisfecho.",
    "igd3": "Se siente mal cuando no puede jugar en internet.",
    "igd4": "No logra reducir el tiempo de juego aunque otras personas se lo pidan.",
    "igd5": "Usa los juegos en internet para evitar pensar en problemas o molestias.",
    "igd6": "Ha tenido conflictos con otras personas por las consecuencias de su forma de jugar.",
    "igd7": "Oculta a otras personas cuanto tiempo pasa jugando.",
    "igd8": "Ha perdido interes en otras actividades porque jugar es lo unico que quiere hacer.",
    "igd9": "Ha tenido conflictos con familia, amistades o pareja por causa de los juegos.",
}

CLASS_MAP = {
    0: "Jugador sin indicadores relevantes de trastorno",
    1: "Jugador en riesgo de desarrollar problemas por gaming",
    2: "Jugador con alta probabilidad de trastorno por gaming",
}
REFINED_LIKERT_CLASS_MAP = {
    0: "Riesgo bajo: respuestas mayormente ocasionales o de baja intensidad",
    1: "Riesgo moderado: presencia sostenida de senales de alerta relacionadas con gaming",
    2: "Riesgo alto: patron de respuestas compatible con compromiso problematico elevado",
}


def yes_no_to_int(series: pd.Series) -> pd.Series:
    mapped = (
        series.astype(str)
        .str.strip()
        .replace({"Yes": 1, "No": 0, "yes": 1, "no": 0, "1": 1, "0": 0})
    )
    return pd.to_numeric(mapped, errors="coerce")


def load_spss(path: Path) -> tuple[pd.DataFrame, object]:
    if not path.is_file():
        raise FileNotFoundError(f"No existe el archivo de entrada: {path}")
    df, meta = pyreadstat.read_sav(path)
    if df.empty:
        raise ValueError("El archivo SPSS no contiene datos.")
    return df, meta


def decode_labeled_column(series: pd.Series, meta: object, column: str) -> pd.Series:
    label_set_name = meta.variable_to_label.get(column)
    if not label_set_name:
        return series
    label_map = meta.value_labels.get(label_set_name, {})
    if not label_map:
        return series
    return series.map(label_map).fillna(series)


def build_contextual_dataframe(df: pd.DataFrame, meta: object) -> pd.DataFrame:
    required = NOMINAL_COLUMNS + ORDINAL_COLUMNS + CONTINUOUS_COLUMNS + ["Igdlabels"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas contextuales esperadas en el dataset: {missing}")

    output = pd.DataFrame()

    for col in NOMINAL_COLUMNS:
        output[col.lower()] = decode_labeled_column(df[col], meta, col).astype(str)

    for col in ORDINAL_COLUMNS:
        output[col.lower()] = pd.to_numeric(df[col], errors="coerce")

    for col in CONTINUOUS_COLUMNS:
        output[col.lower()] = pd.to_numeric(df[col], errors="coerce")

    output[TARGET_COLUMN] = pd.to_numeric(df["Igdlabels"], errors="coerce")

    output["digital_exposure_total"] = (
        output["gameinghourspermonth"] + output["hoursonsocialmedia"]
    )
    output["gaming_minus_social"] = (
        output["gameinghourspermonth"] - output["hoursonsocialmedia"]
    )
    output["gaming_to_social_ratio"] = output["gameinghourspermonth"] / (
        output["hoursonsocialmedia"] + 1.0
    )
    output["sleep_deficit_hours"] = (8.0 - output["psqi4"]).clip(lower=0.0)
    output["long_sleep_latency"] = (output["psqi2"] >= 30).astype(float)
    output["sleep_problem_burden"] = (
        output["psqi6"] + output["psqi7"] + output["psqi8"] + output["psqi9"]
    )
    output["gaming_sleep_risk_interaction"] = (
        output["gameinghourspermonth"] * (output["globalscorepsqi"] + 1.0)
    )

    output = output.dropna(subset=[TARGET_COLUMN]).copy()
    output[TARGET_COLUMN] = output[TARGET_COLUMN].astype(int)
    return output


def build_generic_contextual_dataframe(df: pd.DataFrame, meta: object) -> pd.DataFrame:
    required = (
        GENERIC_NOMINAL_COLUMNS
        + GENERIC_ORDINAL_COLUMNS
        + GENERIC_CONTINUOUS_COLUMNS
        + ["Igdlabels"]
    )
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas genéricas esperadas en el dataset: {missing}")

    output = pd.DataFrame()

    for col in GENERIC_NOMINAL_COLUMNS:
        output[col.lower()] = decode_labeled_column(df[col], meta, col).astype(str)

    for col in GENERIC_ORDINAL_COLUMNS:
        output[col.lower()] = pd.to_numeric(df[col], errors="coerce")

    for col in GENERIC_CONTINUOUS_COLUMNS:
        output[col.lower()] = pd.to_numeric(df[col], errors="coerce")

    output[TARGET_COLUMN] = pd.to_numeric(df["Igdlabels"], errors="coerce")

    output["digital_exposure_total"] = (
        output["gameinghourspermonth"] + output["hoursonsocialmedia"]
    )
    output["gaming_minus_social"] = (
        output["gameinghourspermonth"] - output["hoursonsocialmedia"]
    )
    output["gaming_to_social_ratio"] = output["gameinghourspermonth"] / (
        output["hoursonsocialmedia"] + 1.0
    )
    output["sleep_deficit_hours"] = (8.0 - output["psqi4"]).clip(lower=0.0)
    output["long_sleep_latency"] = (output["psqi2"] >= 30).astype(float)
    output["sleep_problem_burden"] = (
        output["psqi6"] + output["psqi7"] + output["psqi8"] + output["psqi9"]
    )
    output["gaming_sleep_risk_interaction"] = (
        output["gameinghourspermonth"] * (output["globalscorepsqi"] + 1.0)
    )

    output = output.dropna(subset=[TARGET_COLUMN]).copy()
    output[TARGET_COLUMN] = output[TARGET_COLUMN].astype(int)
    return output


def build_simulated_likert_dataframe(
    screening_df: pd.DataFrame, contextual_df: pd.DataFrame
) -> pd.DataFrame:
    df = screening_df.copy().reset_index(drop=True)
    ctx = contextual_df.reset_index(drop=True)

    risk_components = pd.DataFrame(
        {
            "gaming_hours": ctx["gameinghourspermonth"],
            "social_hours": ctx["hoursonsocialmedia"],
            "sleep_latency": ctx["psqi2"],
            "sleep_deficit": (8.0 - ctx["psqi4"]).clip(lower=0.0),
            "sleep_global": ctx["globalscorepsqi"],
            "poor_sleep_flag": (ctx["sleepqual"] == "poorsleep").astype(float),
        }
    )
    risk_index = risk_components.rank(pct=True).mean(axis=1)

    likert_df = pd.DataFrame()
    for col in SCREENING_COLUMNS:
        original = df[col].astype(int)
        simulated = pd.Series(3, index=df.index, dtype=int)
        simulated[(original == 0) & (risk_index < 0.33)] = 1
        simulated[(original == 0) & (risk_index >= 0.33) & (risk_index < 0.66)] = 2
        simulated[(original == 0) & (risk_index >= 0.66)] = 3
        simulated[(original == 1) & (risk_index < 0.33)] = 3
        simulated[(original == 1) & (risk_index >= 0.33) & (risk_index < 0.66)] = 4
        simulated[(original == 1) & (risk_index >= 0.66)] = 5
        likert_df[f"{col}_likert"] = simulated

    likert_df["simulated_risk_index"] = risk_index
    likert_df["likert_total"] = likert_df[[f"{col}_likert" for col in SCREENING_COLUMNS]].sum(axis=1)
    likert_df["igd_label_original"] = df[TARGET_COLUMN].astype(int).values
    likert_df[TARGET_COLUMN] = pd.cut(
        likert_df["likert_total"],
        bins=[8, 22, 28, 46],
        labels=[0, 1, 2],
        include_lowest=True,
    ).astype(int)
    likert_df["igd_label_name"] = likert_df[TARGET_COLUMN].map(REFINED_LIKERT_CLASS_MAP)
    return likert_df


def build_output_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in SCREENING_COLUMNS + ["Igdlabels"] if col not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas esperadas en el dataset: {missing}")

    output = pd.DataFrame()
    for col in SCREENING_COLUMNS:
        output[col] = yes_no_to_int(df[col])

    output["yes_count"] = output[SCREENING_COLUMNS].sum(axis=1)
    output[TARGET_COLUMN] = pd.to_numeric(df["Igdlabels"], errors="coerce")
    output["igd_label_name"] = output[TARGET_COLUMN].map(CLASS_MAP)
    output = output.dropna(subset=[TARGET_COLUMN]).copy()
    output[TARGET_COLUMN] = output[TARGET_COLUMN].astype(int)
    output["yes_count"] = output["yes_count"].astype(int)

    if output[SCREENING_COLUMNS + [TARGET_COLUMN]].isna().any().any():
        raise ValueError("Persisten valores nulos tras convertir los items IGD.")

    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocesa el dataset jordano de Internet Gaming Disorder y genera un "
            "CSV listo para clasificacion multiclase basada en los 9 items IGD."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Ruta al archivo .sav original.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Ruta del CSV de salida.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help="Ruta del reporte JSON.",
    )
    parser.add_argument(
        "--feature-set",
        choices=("screening", "contextual", "hybrid", "generic_hybrid", "likert_simulated"),
        default="generic_hybrid",
        help=(
            "screening = solo igd1-igd9; contextual = variables demograficas, "
            "de uso digital y sueno; hybrid = contextual + igd1-igd9; "
            "generic_hybrid = solo variables genericas de juego/sueno + igd1-igd9; "
            "likert_simulated = version exploratoria 1-5 derivada de igd1-igd9 y contexto."
        ),
    )
    args = parser.parse_args()

    df, meta = load_spss(args.input)
    screening_df = build_output_dataframe(df)
    contextual_df = build_contextual_dataframe(df, meta)
    generic_contextual_df = build_generic_contextual_dataframe(df, meta)

    if args.feature_set == "screening":
        df_model = screening_df
    elif args.feature_set == "contextual":
        df_model = contextual_df
    elif args.feature_set == "generic_hybrid":
        generic_base = generic_contextual_df.reset_index(drop=True)
        screening_base = screening_df[SCREENING_COLUMNS].reset_index(drop=True)
        df_model = pd.concat([generic_base, screening_base], axis=1)
    elif args.feature_set == "likert_simulated":
        df_model = build_simulated_likert_dataframe(screening_df, contextual_df)
    else:
        contextual_base = contextual_df.reset_index(drop=True)
        screening_base = screening_df[SCREENING_COLUMNS].reset_index(drop=True)
        df_model = pd.concat([contextual_base, screening_base], axis=1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_model.to_csv(args.output, index=False, encoding="utf-8")

    report = {
        "input_file": str(args.input),
        "source_url": "https://zenodo.org/records/13382368",
        "doi": "10.5281/zenodo.13382368",
        "n_rows": int(len(df_model)),
        "n_features": int(len(df_model.columns) - 1),
        "feature_set": args.feature_set,
        "target_column": TARGET_COLUMN,
        "feature_columns": [col for col in df_model.columns if col != TARGET_COLUMN],
        "class_map": {
            str(k): v
            for k, v in (
                REFINED_LIKERT_CLASS_MAP.items()
                if args.feature_set == "likert_simulated"
                else CLASS_MAP.items()
            )
        },
        "question_map_es": IGD_QUESTIONS_ES,
        "value_map": {"0": "No", "1": "Yes"},
        "synthetic_note": (
            "La variante likert_simulated no es observada en el dataset original. "
            "Se genera de forma exploratoria a partir de los items binarios IGD y "
            "un indice de riesgo contextual."
        ),
        "likert_total_class_cutoffs": (
            {
                "0": "9-22",
                "1": "23-28",
                "2": "29-45",
            }
            if args.feature_set == "likert_simulated"
            else {}
        ),
        "target_distribution": {
            str(int(k)): int(v)
            for k, v in df_model[TARGET_COLUMN].value_counts().sort_index().items()
        },
        "yes_count_distribution": (
            {
                str(int(k)): int(v)
                for k, v in screening_df["yes_count"].value_counts().sort_index().items()
            }
        ),
        "original_column_labels": {
            col: meta.column_names_to_labels.get(col, "")
            for col in SCREENING_COLUMNS
            + NOMINAL_COLUMNS
            + ORDINAL_COLUMNS
            + CONTINUOUS_COLUMNS
            + ["Igdlabels"]
        },
    }

    with open(args.report, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    print(f"CSV generado en: {args.output}")
    print(f"Reporte generado en: {args.report}")
    print("Target multiclase:", report["class_map"])


if __name__ == "__main__":
    main()
