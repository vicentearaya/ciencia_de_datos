import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))

import matplotlib.pyplot as plt


RAW_DEFAULT = "Dataset/gaming_mental_health_10M_40features.csv.gz"
PROCESSED_DEFAULT = "dataset_preprocessed.csv"

PLOT_COLUMNS = [
    "age",
    "gender",
    "daily_gaming_hours",
    "sleep_hours",
    "stress_level",
    "anxiety_score",
    "depression_score",
    "addiction_level",
    "happiness_score",
    "weekend_gaming_hours",
    "microtransactions_spending",
    "screen_time_total",
    "loneliness_score",
    "social_interaction_score",
]

HEATMAP_COLUMNS = [
    "daily_gaming_hours",
    "sleep_hours",
    "stress_level",
    "anxiety_score",
    "depression_score",
    "addiction_level",
    "happiness_score",
    "loneliness_score",
    "social_interaction_score",
    "screen_time_total",
    "weekend_gaming_hours",
]

CORRELATION_TARGET = "addiction_level"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera graficos relevantes del dataset de gaming y salud mental."
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Ruta del dataset a visualizar. Si no se indica, intenta usar el procesado y luego el original.",
    )
    parser.add_argument(
        "--output-dir",
        default="plots",
        help="Directorio donde se guardaran los graficos.",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=200000,
        help="Cantidad maxima de filas a leer para las visualizaciones.",
    )
    parser.add_argument(
        "--scatter-sample",
        type=int,
        default=15000,
        help="Cantidad maxima de puntos para el scatter plot.",
    )
    return parser.parse_args()


def resolve_input_path(user_path: str | None) -> Path:
    candidates = [user_path, PROCESSED_DEFAULT, RAW_DEFAULT]
    for candidate in candidates:
        if candidate is None:
            continue
        path = Path(candidate)
        if path.exists():
            return path
    raise FileNotFoundError("No se encontro un dataset para generar graficos.")


def load_dataframe(path: Path, sample_rows: int) -> pd.DataFrame:
    return pd.read_csv(path, nrows=sample_rows, usecols=lambda c: c in PLOT_COLUMNS or c in HEATMAP_COLUMNS)


def ensure_output_dir(path: str) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def set_style() -> None:
    plt.style.use("ggplot")
    plt.rcParams.update(
        {
            "figure.figsize": (10, 6),
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def save_figure(output_dir: Path, filename: str) -> None:
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
    plt.close()


def plot_gender_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    counts = df["gender"].value_counts(dropna=False).sort_values(ascending=False)
    fig, ax = plt.subplots()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    counts.plot(kind="bar", ax=ax, color=colors[: len(counts)])
    ax.set_title("Distribucion por genero")
    ax.set_xlabel("Genero")
    ax.set_ylabel("Cantidad")
    for idx, value in enumerate(counts.values):
        ax.text(idx, value, f"{value:,}", ha="center", va="bottom", fontsize=9)
    save_figure(output_dir, "01_gender_distribution.png")


def plot_age_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots()
    ax.hist(df["age"].dropna(), bins=20, color="#33658a", edgecolor="white")
    ax.set_title("Distribucion de edad")
    ax.set_xlabel("Edad")
    ax.set_ylabel("Frecuencia")
    save_figure(output_dir, "02_age_distribution.png")


def plot_mental_health_distributions(df: pd.DataFrame, output_dir: Path) -> None:
    columns = ["anxiety_score", "depression_score", "addiction_level", "happiness_score"]
    titles = [
        "Anxiety Score",
        "Depression Score",
        "Addiction Level",
        "Happiness Score",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    colors = ["#bc4749", "#6a4c93", "#f4a259", "#4d908e"]
    for ax, column, title, color in zip(axes.flat, columns, titles, colors):
        ax.hist(df[column].dropna(), bins=20, color=color, edgecolor="white")
        ax.set_title(title)
        ax.set_xlabel(column)
        ax.set_ylabel("Frecuencia")
    save_figure(output_dir, "03_mental_health_distributions.png")


def plot_gaming_vs_addiction(df: pd.DataFrame, output_dir: Path, scatter_sample: int) -> None:
    plot_df = df[["daily_gaming_hours", "addiction_level"]].dropna()
    if len(plot_df) > scatter_sample:
        plot_df = plot_df.sample(scatter_sample, random_state=42)

    x = plot_df["daily_gaming_hours"].to_numpy()
    y = plot_df["addiction_level"].to_numpy()
    slope, intercept = np.polyfit(x, y, 1)

    fig, ax = plt.subplots()
    ax.scatter(x, y, alpha=0.2, s=16, color="#2a9d8f")
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, color="#e63946", linewidth=2)
    ax.set_title("Horas de juego diarias vs nivel de adiccion")
    ax.set_xlabel("Daily gaming hours")
    ax.set_ylabel("Addiction level")
    save_figure(output_dir, "04_gaming_vs_addiction.png")


def plot_selected_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    corr = df[HEATMAP_COLUMNS].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title("Mapa de correlacion de variables relevantes")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)

    for row in range(corr.shape[0]):
        for col in range(corr.shape[1]):
            ax.text(
                col,
                row,
                f"{corr.iloc[row, col]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )

    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    save_figure(output_dir, "05_correlation_heatmap.png")


def plot_top_correlations(df: pd.DataFrame, output_dir: Path) -> None:
    corr = df.corr(numeric_only=True)[CORRELATION_TARGET].drop(labels=[CORRELATION_TARGET])
    corr = corr.abs().sort_values(ascending=False).head(10).sort_values()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(corr.index, corr.values, color="#577590")
    ax.set_title(f"Top 10 correlaciones absolutas con {CORRELATION_TARGET}")
    ax.set_xlabel("Correlacion absoluta")
    ax.set_ylabel("Variable")
    save_figure(output_dir, "06_top_correlations_addiction.png")


def plot_stress_vs_depression(df: pd.DataFrame, output_dir: Path) -> None:
    grouped = df.groupby("stress_level", dropna=True)["depression_score"].mean().sort_index()
    fig, ax = plt.subplots()
    ax.plot(grouped.index, grouped.values, marker="o", color="#c1121f", linewidth=2)
    ax.set_title("Promedio de depresion por nivel de estres")
    ax.set_xlabel("Stress level")
    ax.set_ylabel("Depression score promedio")
    save_figure(output_dir, "07_stress_vs_depression.png")


def main() -> None:
    args = parse_args()
    input_path = resolve_input_path(args.input)
    output_dir = ensure_output_dir(args.output_dir)
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    set_style()

    df = load_dataframe(input_path, args.sample_rows)

    plot_gender_distribution(df, output_dir)
    plot_age_distribution(df, output_dir)
    plot_mental_health_distributions(df, output_dir)
    plot_gaming_vs_addiction(df, output_dir, args.scatter_sample)
    plot_selected_heatmap(df, output_dir)
    plot_top_correlations(df, output_dir)
    plot_stress_vs_depression(df, output_dir)

    print(f"Dataset usado: {input_path}")
    print(f"Filas leidas: {len(df):,}")
    print(f"Graficos guardados en: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
