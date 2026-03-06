# src/models/clustering.py
"""
Clusters browsing sessions using KMeans (default), GMM, or DBSCAN.
Outputs labeled clusters with human-readable interpretation.
Evaluation: Silhouette Score.
"""

import pandas as pd
import numpy as np
import os
import yaml
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

try:
    from sklearn.mixture import GaussianMixture
    HAS_GMM = True
except ImportError:
    HAS_GMM = False

FEATURES = [
    'duration_min', 'total_visits', 'unique_domains',
    'social_ratio', 'video_ratio', 'learning_ratio',
    'shopping_ratio', 'ai_tools_ratio', 'switching_rate', 'hour'
]

CLUSTER_LABELS = {
    # These are assigned after profiling cluster means
    # Pattern: (high_social, high_learning, high_video, late_hour)
}


def label_cluster(row):
    """Assign human-readable label based on cluster profile."""
    if row['social_ratio'] > 0.4:
        if row['hour'] >= 20:
            return "Late-Night Social Scroller"
        return "Social Media Browser"
    if row['learning_ratio'] > 0.35:
        return "Focused Learner"
    if row['video_ratio'] > 0.4:
        return "Video Binge Watcher"
    if row['shopping_ratio'] > 0.3:
        return "Online Shopper"
    if row['switching_rate'] > 0.6:
        return "Rapid Tab Switcher"
    if row['duration_min'] > 60:
        return "Long Deep-Work Session"
    return "General Browser"


def run_clustering(sessions_path: str, output_path: str,
                   n_clusters: int = 4, algorithm: str = "kmeans",
                   random_seed: int = 42) -> pd.DataFrame:

    print(f"Running {algorithm.upper()} clustering (k={n_clusters})...")

    df = pd.read_csv(sessions_path)

    # Keep only available feature columns
    available = [f for f in FEATURES if f in df.columns]
    X = df[available].fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Fit model ────────────────────────────────────────────────────────────
    if algorithm == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=10)
        labels = model.fit_predict(X_scaled)

    elif algorithm == "gmm" and HAS_GMM:
        model = GaussianMixture(n_components=n_clusters, random_state=random_seed)
        labels = model.fit_predict(X_scaled)

    elif algorithm == "dbscan":
        model = DBSCAN(eps=0.8, min_samples=3)
        labels = model.fit_predict(X_scaled)
        # DBSCAN may produce -1 (noise), map to cluster 0
        labels = np.where(labels == -1, 0, labels)
    else:
        print(f"  Unknown algorithm '{algorithm}', defaulting to KMeans")
        model = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=10)
        labels = model.fit_predict(X_scaled)

    df['cluster'] = labels

    # ── Evaluation ───────────────────────────────────────────────────────────
    if len(set(labels)) > 1:
        score = silhouette_score(X_scaled, labels)
        print(f"  Silhouette Score: {score:.4f}  (higher is better, >0.3 is good)")
    else:
        score = 0.0
        print("  ⚠️  Only 1 cluster found — try more data or adjust parameters")

    # ── Cluster profiles ─────────────────────────────────────────────────────
    profile = df.groupby('cluster')[available].mean().round(3)
    print("\n  Cluster Profiles:")
    print(profile.to_string())

    # ── Assign human-readable labels ─────────────────────────────────────────
    cluster_label_map = {}
    for cid, row in profile.iterrows():
        cluster_label_map[cid] = label_cluster(row)

    df['cluster_label'] = df['cluster'].map(cluster_label_map)

    print("\n  Cluster Label Distribution:")
    print(df['cluster_label'].value_counts().to_string())

    # ── Save outputs ─────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    joblib.dump(scaler, "data/scaler.pkl")
    joblib.dump(model, "data/cluster_model.pkl")
    profile.to_csv("data/cluster_profiles.csv")

    print(f"\n  ✅ Clustered sessions saved → {output_path}")
    print(f"  Silhouette Score: {score:.4f}")
    return df, score, profile


if __name__ == "__main__":
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    run_clustering(
        sessions_path=cfg['paths']['sessions'],
        output_path=cfg['paths']['sessions_clustered'],
        n_clusters=cfg['clustering']['n_clusters'],
        algorithm=cfg['clustering']['algorithm'],
        random_seed=cfg['clustering']['random_seed']
    )
