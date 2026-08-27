"""
src/customer_segmentation.py
==============================

Standalone customer segmentation module (K-Means on RFM features).

This module is the canonical home for the segmentation logic. It was
originally implemented inline inside `src/train_model.py`; the reusable
pieces (`choose_k`, `label_segments`) now live here, and
`src/train_model.py` imports them from this module so there is a single
source of truth and no duplicated logic. `src/train_model.py` continues to
work exactly as before (it still calls `run_segmentation()` as part of the
combined modeling pipeline), and this module can also be run entirely on
its own.

Uses actual project columns from `data/processed/rfm_features.csv`
(`customer_id`, `recency_days`, `frequency`, `monetary`, `avg_order_value`,
`R_score`, `F_score`, `M_score`, `RFM_score`), as produced by
`src/feature_engineering.py`.

Run as:
    python -m src.customer_segmentation
"""

from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.config import DATA_PROCESSED_DIR, RANDOM_SEED, REPORTS_DIR, get_logger

logger = get_logger(__name__)

np.random.seed(RANDOM_SEED)

# Features used for clustering. These are the exact column names produced by
# src/feature_engineering.py::build_rfm_features.
SEGMENT_FEATURES = ["recency_days", "frequency", "monetary"]


def load_rfm_features() -> pd.DataFrame:
    """Load the RFM feature table produced by src/feature_engineering.py."""
    path = DATA_PROCESSED_DIR / "rfm_features.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run 'python -m src.feature_engineering' first."
        )
    return pd.read_csv(path)


def prepare_features(rfm: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, StandardScaler]:
    """Select and scale the clustering features."""
    X = rfm[SEGMENT_FEATURES].fillna(0).copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X, X_scaled, scaler


def choose_k(scaled_features: np.ndarray, k_range=range(4, 8)) -> tuple[int, dict]:
    """
    Elbow (inertia) + silhouette analysis to pick a reasonable k.

    Note: k_range starts at 4 rather than 2. Purely maximizing silhouette
    score on this dataset tends to collapse to k=2 (an "active vs. inactive"
    split), which is statistically clean but not business-actionable. We
    constrain the search to k in [4, 7] so the resulting segments map onto
    the standard RFM-style personas (Champions, Loyal, Potential Loyalists,
    New, At Risk, Hibernating) that marketing teams can act on, and still
    pick the best k within that range by silhouette score.
    """
    inertias = {}
    silhouettes = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = km.fit_predict(scaled_features)
        inertias[k] = km.inertia_
        sample_size = min(2000, len(scaled_features))
        idx = np.random.choice(len(scaled_features), sample_size, replace=False)
        silhouettes[k] = silhouette_score(scaled_features[idx], labels[idx])

    best_k = max(silhouettes, key=silhouettes.get)
    logger.info("Silhouette scores by k: %s", {k: round(v, 3) for k, v in silhouettes.items()})
    logger.info("Selected k=%d based on highest silhouette score.", best_k)
    return best_k, {"inertias": inertias, "silhouettes": silhouettes}


def label_segments(cluster_profile: pd.DataFrame) -> dict[int, str]:
    """
    Assign meaningful business labels to clusters based on their RFM
    characteristics *relative to the other clusters found* (rank-based),
    rather than by arbitrary cluster ID or a single population-wide median
    (which tends to collapse everything into just two buckets when clusters
    are unevenly sized).
    """
    k = len(cluster_profile)

    recency_rank = cluster_profile["recency_days"].rank(method="first", ascending=True)
    freq_rank = cluster_profile["frequency"].rank(method="first", ascending=False)
    monetary_rank = cluster_profile["monetary"].rank(method="first", ascending=False)

    top_third = max(1, round(k / 3))
    mid_half = max(1, round(k / 2))

    labels = {}
    for cluster_id in cluster_profile.index:
        r_rank = recency_rank[cluster_id]
        f_rank = freq_rank[cluster_id]
        m_rank = monetary_rank[cluster_id]

        recent = r_rank <= top_third
        somewhat_recent = r_rank <= mid_half
        frequent = f_rank <= top_third
        somewhat_frequent = f_rank <= mid_half
        high_value = m_rank <= top_third

        if recent and frequent and high_value:
            label = "Champions"
        elif somewhat_recent and somewhat_frequent:
            label = "Loyal Customers"
        elif recent and not somewhat_frequent:
            label = "New Customers"
        elif somewhat_recent:
            label = "Potential Loyalists"
        elif not somewhat_recent and (somewhat_frequent or m_rank <= mid_half):
            label = "At Risk"
        else:
            label = "Hibernating"

        labels[cluster_id] = label

    return labels


def profile_clusters(rfm: pd.DataFrame) -> pd.DataFrame:
    """Return mean RFM values per cluster (used to derive segment labels)."""
    return rfm.groupby("cluster")[SEGMENT_FEATURES].mean()


def plot_diagnostics(diagnostics: dict, best_k: int, rfm: pd.DataFrame) -> None:
    """Save elbow/silhouette/segment-size and RFM-characteristics figures to reports/."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ks = list(diagnostics["inertias"].keys())
    axes[0].plot(ks, list(diagnostics["inertias"].values()), marker="o")
    axes[0].set_title("Elbow Method (Inertia vs. k)")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Inertia")

    axes[1].plot(ks, list(diagnostics["silhouettes"].values()), marker="o", color="orange")
    axes[1].axvline(best_k, color="gray", linestyle="--", label=f"selected k={best_k}")
    axes[1].set_title("Silhouette Score vs. k")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].legend()

    segment_counts = rfm["segment"].value_counts()
    axes[2].barh(segment_counts.index, segment_counts.values, color="teal")
    axes[2].set_title("Customer Segment Sizes")
    axes[2].set_xlabel("Number of Customers")

    plt.tight_layout()
    seg_plot_path = REPORTS_DIR / "segmentation_analysis.png"
    plt.savefig(seg_plot_path, dpi=120)
    plt.close(fig)
    logger.info("Saved segmentation visualization to %s", seg_plot_path)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    profile_by_segment = rfm.groupby("segment")[SEGMENT_FEATURES].mean()
    profile_norm = (profile_by_segment - profile_by_segment.min()) / (
        profile_by_segment.max() - profile_by_segment.min() + 1e-9
    )
    profile_norm.plot(kind="bar", ax=ax2)
    ax2.set_title("Normalized RFM Characteristics by Segment")
    ax2.set_ylabel("Normalized value (0-1)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    rfm_plot_path = REPORTS_DIR / "segment_rfm_characteristics.png"
    plt.savefig(rfm_plot_path, dpi=120)
    plt.close(fig2)
    logger.info("Saved RFM characteristics chart to %s", rfm_plot_path)


def run_segmentation(save_outputs: bool = True) -> pd.DataFrame:
    """
    Full segmentation workflow: load RFM features, scale, choose k, fit
    K-Means, label clusters, and (optionally) persist outputs.

    Returns the RFM DataFrame enriched with `cluster` and `segment` columns.
    """
    logger.info("--- Customer Segmentation (K-Means) ---")
    rfm = load_rfm_features()

    X, X_scaled, _scaler = prepare_features(rfm)
    best_k, diagnostics = choose_k(X_scaled)

    kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_SEED, n_init=10)
    rfm["cluster"] = kmeans.fit_predict(X_scaled)

    cluster_profile = profile_clusters(rfm)
    segment_labels = label_segments(cluster_profile)
    rfm["segment"] = rfm["cluster"].map(segment_labels)

    logger.info("Segment sizes:\n%s", rfm["segment"].value_counts().to_string())

    if save_outputs:
        out_path = DATA_PROCESSED_DIR / "customer_segments.csv"
        rfm[["customer_id", "recency_days", "frequency", "monetary", "cluster", "segment"]].to_csv(
            out_path, index=False
        )
        logger.info("Saved %s", out_path)
        plot_diagnostics(diagnostics, best_k, rfm)

    return rfm


def main() -> int:
    logger.info("=" * 70)
    logger.info("OmniStyle Customer Intelligence Platform - Customer Segmentation")
    logger.info("=" * 70)
    run_segmentation(save_outputs=True)
    logger.info("Segmentation complete. See data/processed/customer_segments.csv and reports/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
