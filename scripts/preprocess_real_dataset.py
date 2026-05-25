from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pyreadstat

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "Data_cruda" / "jordan_igd_sleep_quality.sav"
DEFAULT_OUTPUT = ROOT / "data_preprocesada" / "dataset_preprocesado_jordan.csv"
DEFAULT_REPORT = ROOT / "data_preprocesada" / "preprocessing_report_jordan.json"
TARGET_COLUMN = "igd_label"
SCREENING_COLUMNS = [f"igd{i}" for i in range(1, 10)]

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
    args = parser.parse_args()

    df, meta = load_spss(args.input)
    df_model = build_output_dataframe(df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_model.to_csv(args.output, index=False, encoding="utf-8")

    report = {
        "input_file": str(args.input),
        "source_url": "https://zenodo.org/records/13382368",
        "doi": "10.5281/zenodo.13382368",
        "n_rows": int(len(df_model)),
        "n_features": len(SCREENING_COLUMNS),
        "target_column": TARGET_COLUMN,
        "feature_columns": SCREENING_COLUMNS,
        "class_map": {str(k): v for k, v in CLASS_MAP.items()},
        "question_map_es": IGD_QUESTIONS_ES,
        "value_map": {"0": "No", "1": "Yes"},
        "target_distribution": {
            str(int(k)): int(v)
            for k, v in df_model[TARGET_COLUMN].value_counts().sort_index().items()
        },
        "yes_count_distribution": {
            str(int(k)): int(v)
            for k, v in df_model["yes_count"].value_counts().sort_index().items()
        },
        "original_column_labels": {
            col: meta.column_names_to_labels.get(col, "")
            for col in SCREENING_COLUMNS + ["Igdlabels"]
        },
    }

    with open(args.report, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    print(f"CSV generado en: {args.output}")
    print(f"Reporte generado en: {args.report}")
    print("Target multiclase:", report["class_map"])


if __name__ == "__main__":
    main()
