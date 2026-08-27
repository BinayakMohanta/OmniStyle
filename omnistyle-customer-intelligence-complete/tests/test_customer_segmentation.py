"""
Unit tests for src/customer_segmentation.py

Focuses on the pure, deterministic logic (label_segments) rather than
re-running full K-Means on the generated dataset, so these tests stay fast
and independent of any particular data volume.
"""

import pandas as pd

from src.customer_segmentation import label_segments


def test_label_segments_identifies_champions_cluster():
    # Cluster 0: very recent, very frequent, very high spend -> Champions
    # Cluster 1: old, infrequent, low spend -> Hibernating
    # Cluster 2: recent-ish, mid frequency, mid spend -> Loyal Customers
    # Cluster 3: very recent, low frequency -> New Customers
    cluster_profile = pd.DataFrame(
        {
            "recency_days": [5, 300, 60, 10],
            "frequency": [20, 1, 8, 2],
            "monetary": [5000, 50, 900, 150],
        },
        index=[0, 1, 2, 3],
    )

    labels = label_segments(cluster_profile)

    assert labels[0] == "Champions"
    assert labels[1] == "Hibernating"


def test_label_segments_returns_a_label_for_every_cluster():
    cluster_profile = pd.DataFrame(
        {
            "recency_days": [10, 400, 200, 50, 15, 500],
            "frequency": [15, 0, 3, 6, 12, 1],
            "monetary": [3000, 20, 300, 700, 2000, 40],
        },
        index=range(6),
    )

    labels = label_segments(cluster_profile)

    assert len(labels) == 6
    assert all(isinstance(v, str) and len(v) > 0 for v in labels.values())


def test_label_segments_handles_minimal_k():
    # Even with only two clusters, every cluster should get a valid label.
    cluster_profile = pd.DataFrame(
        {
            "recency_days": [10, 300],
            "frequency": [10, 1],
            "monetary": [2000, 50],
        },
        index=[0, 1],
    )

    labels = label_segments(cluster_profile)

    assert len(labels) == 2
    valid_labels = {
        "Champions", "Loyal Customers", "Potential Loyalists",
        "New Customers", "At Risk", "Hibernating",
    }
    assert set(labels.values()).issubset(valid_labels)
