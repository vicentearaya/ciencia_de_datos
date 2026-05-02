from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.cluster import (
    Birch,
    KMeans,
    MiniBatchKMeans,
    SpectralClustering,
)
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "Dataset" / "data_preprocesada.csv"
METRICS_OUT = ROOT / "Models" / "clustering_comparison.json"

RANDOM_STATE = 42
SAMPLE_N = 20_000
PCA_VAR = 0.85
K_MIN = 2
K_MAX = 8
SUBSAMPLE_SEEDS = [RANDOM_STATE + i for i in range(5)]


def build_models(n_clusters: int) -> dict[str, BaseEstimator]:
    return {
        "kmeans": KMeans(
            n_clusters=n_clusters,
            random_state=RANDOM_STATE,
            n_init="auto",
        ),
        "minibatch_kmeans": MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=RANDOM_STATE,
            batch_size=4096,
            n_init="auto",
        ),
        "birch_threshold_0p5": Birch(
            branching_factor=50,
            threshold=0.5,
            n_clusters=n_clusters,
        ),
        # 'full' suele saturar RAM en alta dimension; diag es mas estable aqui tras PCA.
        "gmm_diag": GaussianMixture(
            n_components=n_clusters,
            covariance_type="diag",
            random_state=RANDOM_STATE,
            n_init=5,
            max_iter=300,
            reg_covar=1e-3,
        ),
        "spectral_nn_15": SpectralClustering(
            n_clusters=n_clusters,
            affinity="nearest_neighbors",
            n_neighbors=15,
            assign_labels="kmeans",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
    }


def inertia_if_any(model_name: str, model: BaseEstimator) -> float | None:
    if model_name.startswith("kmeans") or model_name.startswith("minibatch_kmeans"):
        return float(getattr(model, "inertia_", float("nan")))
    return None


def choose_k_on_embedding(X_emb: np.ndarray, km_random_state: int) -> tuple[int, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for k in range(K_MIN, K_MAX + 1):
        km = KMeans(n_clusters=k, random_state=km_random_state, n_init="auto")
        labels = km.fit_predict(X_emb)
        sil = silhouette_score(X_emb, labels)
        dbi = davies_bouldin_score(X_emb, labels)
        chi = calinski_harabasz_score(X_emb, labels)
        rows.append(
            {
                "k": k,
                "silhouette": float(sil),
                "davies_bouldin": float(dbi),
                "calinski_harabasz": float(chi),
            }
        )
    k_df = pd.DataFrame(rows)
    k_df["sil_norm"] = (k_df["silhouette"] - k_df["silhouette"].min()) / (
        k_df["silhouette"].max() - k_df["silhouette"].min() + 1e-9
    )
    k_df["dbi_norm"] = (k_df["davies_bouldin"].max() - k_df["davies_bouldin"]) / (
        k_df["davies_bouldin"].max() - k_df["davies_bouldin"].min() + 1e-9
    )
    k_df["chi_norm"] = (k_df["calinski_harabasz"] - k_df["calinski_harabasz"].min()) / (
        k_df["calinski_harabasz"].max() - k_df["calinski_harabasz"].min() + 1e-9
    )
    k_df["composite_score"] = (
        0.45 * k_df["sil_norm"] + 0.30 * k_df["dbi_norm"] + 0.25 * k_df["chi_norm"]
    )
    best_k = int(k_df.sort_values("composite_score", ascending=False).iloc[0]["k"])
    return best_k, k_df


def run_once(
    df: pd.DataFrame,
    feat_cols: list[str],
    sample_seed: int,
) -> dict[str, Any]:
    X_sample = df[feat_cols].sample(n=SAMPLE_N, random_state=sample_seed).to_numpy(dtype=float)

    pca_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=PCA_VAR, random_state=RANDOM_STATE, svd_solver="full")),
        ]
    )
    X_emb = pca_pipe.fit_transform(X_sample)
    n_components = int(pca_pipe.named_steps["pca"].n_components_)

    best_k, k_search = choose_k_on_embedding(X_emb, km_random_state=sample_seed)
    if best_k < K_MIN:
        best_k = K_MIN

    results: list[dict[str, Any]] = []
    models = build_models(best_k)

    for model_name, model in models.items():
        labels = model.fit_predict(X_emb)

        sil = silhouette_score(X_emb, labels)
        dbi = davies_bouldin_score(X_emb, labels)
        chi = calinski_harabasz_score(X_emb, labels)

        uniq_pred = sorted(set(labels.astype(int).tolist()))
        n_pred_clusters = len(uniq_pred)
        pct_smallest_cluster = (
            pd.Series(labels).value_counts(normalize=True).min() if n_pred_clusters else float("nan")
        )

        row: dict[str, Any] = {
            "model": model_name,
            "k_used": int(best_k),
            "n_clusters_on_pred": int(n_pred_clusters),
            "pct_smallest_cluster_pred": float(pct_smallest_cluster),
            "silhouette_mean": float(sil),
            "davies_bouldin": float(dbi),
            "calinski_harabasz": float(chi),
        }
        inert = inertia_if_any(model_name, model)
        if inert is not None and not np.isnan(inert):
            row["inertia"] = inert

        results.append(row)

    res_df = pd.DataFrame(results)
    res_sorted = res_df.sort_values(
        ["silhouette_mean", "davies_bouldin"], ascending=[False, True]
    ).reset_index(drop=True)

    return {
        "sample_seed": int(sample_seed),
        "n_components_actual": n_components,
        "k_choice_table": k_search.to_dict(orient="records"),
        "chosen_k": int(best_k),
        "rows": res_sorted.to_dict(orient="records"),
    }


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"No existe el dataset: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    if not feat_cols:
        raise ValueError("No se encontraron columnas 'feat_*' en data_preprocesada.csv.")

    if len(df) < SAMPLE_N:
        raise ValueError(f"Se requieren al menos {SAMPLE_N} filas; hay {len(df)}.")

    runs: list[dict[str, Any]] = []
    for sample_seed in SUBSAMPLE_SEEDS:
        runs.append(run_once(df, feat_cols, sample_seed))

    all_rows: list[dict[str, Any]] = []
    for run in runs:
        for row in run["rows"]:
            all_rows.append({"sample_seed": run["sample_seed"], **row})

    long_df = pd.DataFrame(all_rows)
    summary_rows: list[dict[str, Any]] = []
    for model_name, g in long_df.groupby("model"):
        k_counts = Counter(g["k_used"].astype(int).tolist())
        k_mode = int(k_counts.most_common(1)[0][0])
        summary_rows.append(
            {
                "model": model_name,
                "runs": int(len(g)),
                "k_mode": k_mode,
                "k_used_mean": float(g["k_used"].mean()),
                "silhouette_mean_avg": float(g["silhouette_mean"].mean()),
                "silhouette_mean_std": float(g["silhouette_mean"].std(ddof=0)),
                "davies_bouldin_avg": float(g["davies_bouldin"].mean()),
                "davies_bouldin_std": float(g["davies_bouldin"].std(ddof=0)),
                "calinski_harabasz_avg": float(g["calinski_harabasz"].mean()),
                "calinski_harabasz_std": float(g["calinski_harabasz"].std(ddof=0)),
                "pct_smallest_cluster_pred_avg": float(g["pct_smallest_cluster_pred"].mean()),
                "pct_smallest_cluster_pred_min": float(g["pct_smallest_cluster_pred"].min()),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_sorted = summary_df.sort_values(
        ["silhouette_mean_avg", "davies_bouldin_avg"], ascending=[False, True]
    ).reset_index(drop=True)

    payload: dict[str, Any] = {
        "evaluation_note": (
            "Comparativa no supervisada con multiples submuestreos reproducibles; "
            "StandardScaler + PCA(varianza acumulada); sin etiquetas cluster_id; "
            "k elegido por KMeans provisional + metricas internas sobre el mismo embedding por corrida."
        ),
        "sample_n": SAMPLE_N,
        "subsample_seeds": SUBSAMPLE_SEEDS,
        "pca_variance_ratio_target": PCA_VAR,
        "runs": runs,
        "summary_sorted": summary_sorted.to_dict(orient="records"),
    }

    METRICS_OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Seeds de submuestreo:", SUBSAMPLE_SEEDS)
    for run in runs:
        print(
            f"\n--- seed={run['sample_seed']} | k={run['chosen_k']} | n_comp={run['n_components_actual']} ---"
        )
        print(pd.DataFrame(run["rows"]).to_string(index=False))

    print("\nResumen agregado (promedio sobre seeds; orden: silhouette_avg DESC, DB_avg ASC):")
    print(summary_sorted.to_string(index=False))
    print(f"\nMetricas escritas en: {METRICS_OUT}")


if __name__ == "__main__":
    main()
