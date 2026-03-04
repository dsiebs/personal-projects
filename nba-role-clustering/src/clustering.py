from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def scale_features(
    feats: pd.DataFrame,
    feature_cols: Iterable[str],
) -> Tuple[np.ndarray, StandardScaler]:
    X = feats[list(feature_cols)].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def search_kmeans_k(
    feats: pd.DataFrame,
    feature_cols: Iterable[str],
    k_values: Iterable[int],
    random_state: int = 42,
) -> pd.DataFrame:
    X_scaled, _ = scale_features(feats, feature_cols)
    results: List[Dict[str, float]] = []
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        results.append({"k": k, "silhouette": sil, "inertia": km.inertia_})
    return pd.DataFrame(results).sort_values("k")


def run_kmeans(
    feats: pd.DataFrame,
    feature_cols: Iterable[str],
    k: int,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, KMeans, StandardScaler, float]:
    X_scaled, scaler = scale_features(feats, feature_cols)
    km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)

    out = feats.copy()
    out["CLUSTER"] = labels
    return out, km, scaler, sil


def nearest_examples_to_centroids(
    feats_clustered: pd.DataFrame,
    feature_cols: Iterable[str],
    kmeans: KMeans,
    scaler: StandardScaler,
    n_examples: int = 8,
) -> Dict[int, pd.DataFrame]:
    """
    For each cluster, return the n_examples players closest to the cluster centroid
    in standardized feature space.
    """
    feature_cols = list(feature_cols)
    X = feats_clustered[feature_cols].values
    X_scaled = scaler.transform(X)
    centroids = kmeans.cluster_centers_

    # Precompute distances to own centroid only
    labels = feats_clustered["CLUSTER"].to_numpy()
    examples: Dict[int, pd.DataFrame] = {}
    for cluster_id in sorted(np.unique(labels)):
        idx = np.where(labels == cluster_id)[0]
        Xc = X_scaled[idx]
        centroid = centroids[cluster_id]
        dists = np.linalg.norm(Xc - centroid, axis=1)
        order = np.argsort(dists)[:n_examples]
        subset = feats_clustered.iloc[idx[order]][
            ["PLAYER_NAME", "SEASON", "TEAM_ID", "POSITION_LISTED"] + list(feature_cols)
        ].copy()
        subset["DIST_TO_CENTROID"] = dists[order]
        examples[int(cluster_id)] = subset
    return examples

