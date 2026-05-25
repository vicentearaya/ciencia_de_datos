"""
Ejecuta clustering exploratorio sobre los 9 items IGD del dataset jordano para
identificar perfiles de jugadores sin usar la etiqueta `igd_label` durante el ajuste.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from common import DEFAULT_DATA, TARGET_COLUMN, build_preprocessor, load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Clustering exploratorio del dataset jordano.")
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
        help="Columna objetivo a excluir del clustering.",
    )
    parser.add_argument(
        "--min-k",
        type=int,
        default=2,
        help="Numero minimo de clusters a evaluar.",
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=6,
        help="Numero maximo de clusters a evaluar.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
    )
    args = parser.parse_args()

    if args.min_k < 2:
        raise ValueError("--min-k debe ser al menos 2.")
    if args.max_k <= args.min_k:
        raise ValueError("--max-k debe ser mayor que --min-k.")

    X, y = load_dataset(args.data, target=args.target)
    transformer = build_preprocessor(X)
    X_prepared = transformer.fit_transform(X)

    evaluations: list[dict[str, float | int | str]] = []
    best_result: dict[str, float | int | str] | None = None
    best_labels = None

    for k in range(args.min_k, args.max_k + 1):
        kmeans = KMeans(n_clusters=k, n_init=20, random_state=args.random_state)
        kmeans_labels = kmeans.fit_predict(X_prepared)
        kmeans_silhouette = silhouette_score(X_prepared, kmeans_labels)
        evaluations.append(
            {
                "algorithm": "kmeans",
                "k": k,
                "silhouette": float(kmeans_silhouette),
                "inertia": float(kmeans.inertia_),
            }
        )
        if best_result is None or kmeans_silhouette > best_result["silhouette"]:
            best_result = evaluations[-1]
            best_labels = kmeans_labels

        agglomerative = AgglomerativeClustering(n_clusters=k)
        agg_labels = agglomerative.fit_predict(X_prepared)
        agg_silhouette = silhouette_score(X_prepared, agg_labels)
        evaluations.append(
            {
                "algorithm": "agglomerative",
                "k": k,
                "silhouette": float(agg_silhouette),
            }
        )
        if best_result is None or agg_silhouette > best_result["silhouette"]:
            best_result = evaluations[-1]
            best_labels = agg_labels

    assert best_result is not None
    assert best_labels is not None

    pca = PCA(n_components=2, random_state=args.random_state)
    coords = pca.fit_transform(X_prepared)

    assignments = X.copy()
    assignments[args.target] = y.values
    assignments["cluster"] = best_labels
    assignments["pca_1"] = coords[:, 0]
    assignments["pca_2"] = coords[:, 1]

    assignments_path = args.data.parent / "cluster_assignments_jordan.csv"
    report_path = Path(__file__).resolve().parent / "clustering_report_jordan.json"
    assignments.to_csv(assignments_path, index=False, encoding="utf-8")

    crosstab = pd.crosstab(assignments["cluster"], assignments[args.target])
    report = {
        "data_file": str(args.data),
        "target_excluded": args.target,
        "n_rows": int(len(assignments)),
        "n_features": int(X.shape[1]),
        "best_model": best_result,
        "cluster_sizes": {
            str(int(k)): int(v)
            for k, v in assignments["cluster"].value_counts().sort_index().items()
        },
        "cluster_vs_label": {
            str(int(cluster)): {str(int(label)): int(value) for label, value in row.items()}
            for cluster, row in crosstab.to_dict(orient="index").items()
        },
        "pca_explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
        "evaluations": evaluations,
    }

    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    print(f"Mejor clustering: {best_result}")
    print(f"Asignaciones guardadas en: {assignments_path}")
    print(f"Reporte guardado en: {report_path}")


if __name__ == "__main__":
    main()
