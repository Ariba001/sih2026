"""Phase 3: Graph-Based Spatial Context - Detect wafer-level anomalies via spatial clustering.

Strategy: Build k-NN spatial graph on die coordinates. Identify isolated anomalies
(anomalous components surrounded by normal neighbors) and defect clusters (3+ adjacent
anomalies). Chain measurements across time intervals to catch progressive-only defects
(late bloomers).

Expected impact:
- Catches late bloomers (defects manifesting only in 168h)
- Reduces false alarms on normal lots (spatial clustering as confirmation)
- Targets ≥96% recall on late bloomers, ≤4.5% overkill overall
"""
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from typing import Dict, List, Optional, Tuple


class SpatialAnomalyDetector:
    """Detect anomalies via wafer-level spatial clustering and temporal progression."""

    def __init__(self, cfg=None, k_nn: int = 6, distance_threshold: float = 500.0):
        """
        Args:
            cfg: Config object (for consistency with other modules).
            k_nn: Number of nearest neighbors for spatial graph (default 6).
            distance_threshold: Max distance (μm) to consider neighbors (default 500μm on wafer).
        """
        self.cfg = cfg
        self.k_nn = k_nn
        self.distance_threshold = distance_threshold
        self.is_fitted = False
        self.normalization_percentile = 99  # 1.0 = 99th percentile on good parts

    def fit(self, df: pd.DataFrame, rows: pd.DataFrame = None):
        """
        Fit spatial detector on training data (calibrate normalization).

        Args:
            df: Training dataframe (component level).
            rows: Row-level scores (optional, for future temporal features).
        """
        # For now, normalization will be set during scoring on good components
        self.is_fitted = True
        return self

    def score(self, df: pd.DataFrame, rows: pd.DataFrame = None) -> pd.DataFrame:
        """
        Score components based on spatial anomalies and temporal progression.

        Args:
            df: Component-level dataframe with Lot_ID, Component_ID, die_x, die_y.
            rows: Row-level scores (component_id, param, A_score_base or individual detector scores).

        Returns:
            DataFrame with spatial_score column (normalized to ~1.0 = 99th percentile).
        """
        if rows is None or len(rows) == 0:
            # No rows data; return neutral scores
            return pd.DataFrame({"Component_ID": df["Component_ID"], "spatial_score": 0.0})

        # Ensure required columns
        has_coords = "die_x" in df.columns and "die_y" in df.columns
        if not has_coords:
            return pd.DataFrame({"Component_ID": df["Component_ID"], "spatial_score": 0.0})

        result = []

        # Process by lot (spatial anomalies are lot-specific)
        for lot_id, lot_group in df.groupby("Lot_ID"):
            lot_rows = rows[rows["Lot_ID"] == lot_id]
            if len(lot_group) == 0 or len(lot_rows) == 0:
                for _, comp in lot_group.iterrows():
                    result.append({"Component_ID": comp["Component_ID"], "spatial_score": 0.0})
                continue

            # Get spatial coordinates
            coords = lot_group[["die_x", "die_y"]].to_numpy(float)
            comp_ids = lot_group["Component_ID"].values

            # Get anomaly scores from rows (max across parameters per component)
            # Use max of individual detector scores (not A_score which doesn't exist yet)
            comp_to_anomaly = {}
            for comp_id in comp_ids:
                comp_rows = lot_rows[lot_rows["Component_ID"] == comp_id]
                if len(comp_rows) == 0:
                    comp_to_anomaly[comp_id] = 0.0
                else:
                    # Use max of DPAT, Mahalanobis, VAE detector scores
                    score = max(
                        float(comp_rows["dpat_score"].max()),
                        float(comp_rows["maha_score"].max()),
                        float(comp_rows.get("vae_score", 0).max())
                    )
                    comp_to_anomaly[comp_id] = score

            # Build k-NN spatial graph
            spatial_scores = self._score_by_spatial_clustering(
                coords, comp_ids, comp_to_anomaly
            )

            for comp_id, s_score in spatial_scores.items():
                result.append({"Component_ID": comp_id, "spatial_score": s_score})

        result_df = pd.DataFrame(result)

        # Normalize: 1.0 = 99th percentile on good components (A_score < 1.0)
        good_mask = result_df["spatial_score"] < 1.0
        if good_mask.sum() > 0:
            p99 = result_df[good_mask]["spatial_score"].quantile(0.99)
            if p99 > 0:
                result_df["spatial_score"] = result_df["spatial_score"] / p99
            else:
                # No variation in good parts; use min non-zero
                min_val = result_df[good_mask]["spatial_score"].max()
                if min_val > 0:
                    result_df["spatial_score"] = result_df["spatial_score"] / min_val

        return result_df

    def _score_by_spatial_clustering(
        self, coords: np.ndarray, comp_ids: np.ndarray, comp_to_anomaly: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Score components based on spatial isolation and clustering.

        Algorithm:
        1. Build k-NN graph on die coordinates.
        2. For each component:
           a. Isolation score: If anomalous, how many normal neighbors? (higher = more isolated)
           b. Cluster score: If normal, how many anomalous neighbors? (higher = in cluster)
           c. Temporal consistency: Does anomaly align with 168h measurement?
        3. Combine: isolation + cluster + temporal.

        Args:
            coords: (N, 2) array of die_x, die_y.
            comp_ids: (N,) array of component IDs.
            comp_to_anomaly: Dict mapping comp_id → anomaly score.

        Returns:
            Dict mapping comp_id → spatial_score.
        """
        n = len(comp_ids)
        if n <= 1:
            return {cid: 0.0 for cid in comp_ids}

        # Compute pairwise distances
        distances = cdist(coords, coords, metric="euclidean")

        # For each component, find k-NN (including self)
        spatial_scores = {}

        for i, comp_id in enumerate(comp_ids):
            anomaly_score = comp_to_anomaly.get(comp_id, 0.0)

            # Find k nearest neighbors (including self)
            k_effective = min(self.k_nn + 1, n)  # +1 for self
            neighbor_indices = np.argsort(distances[i])[:k_effective]
            neighbor_ids = comp_ids[neighbor_indices]
            neighbor_distances = distances[i][neighbor_indices]

            # Exclude self
            neighbor_indices = neighbor_indices[1:]
            neighbor_ids = neighbor_ids[1:]
            neighbor_distances = neighbor_distances[1:]

            if len(neighbor_ids) == 0:
                spatial_scores[comp_id] = anomaly_score
                continue

            # Get anomaly scores of neighbors
            neighbor_anomalies = np.array(
                [comp_to_anomaly.get(nid, 0.0) for nid in neighbor_ids]
            )

            # Isolation score (if this component is anomalous, how isolated is it?)
            if anomaly_score >= 1.0:
                # Anomalous component
                # Isolation = % of neighbors that are normal (< 1.0)
                normal_neighbors = (neighbor_anomalies < 1.0).sum()
                isolation_ratio = normal_neighbors / len(neighbor_ids)
                # Higher isolation (surrounded by good) = higher score
                isolation_score = isolation_ratio * anomaly_score
            else:
                isolation_score = 0.0

            # Cluster score (if this component is normal, is it in a defect cluster?)
            if anomaly_score < 1.0:
                # Normal component
                # Cluster = # of anomalous neighbors (3+ = cluster)
                anomalous_neighbors = (neighbor_anomalies >= 1.0).sum()
                if anomalous_neighbors >= 3:
                    # In a cluster of anomalies
                    cluster_score = 0.5 * (anomalous_neighbors / len(neighbor_ids))
                else:
                    cluster_score = 0.0
            else:
                cluster_score = 0.0

            # Temporal consistency score (placeholder for now)
            # In future: check if anomaly increases monotonically from 96h → 168h
            temporal_score = 0.0

            # Combine scores: max of (isolation, cluster, temporal)
            combined = max(isolation_score, cluster_score, temporal_score)

            spatial_scores[comp_id] = combined

        return spatial_scores

    def identify_defect_clusters(
        self, rows: pd.DataFrame, threshold: float = 0.8
    ) -> pd.DataFrame:
        """
        Identify spatial clusters of anomalies for visualization/explanation.

        Args:
            rows: Row-level scores from pipeline.score_rows().
            threshold: Anomaly threshold (A_score >= threshold).

        Returns:
            DataFrame with cluster assignments per lot.
        """
        if "die_x" not in rows.columns or "die_y" not in rows.columns:
            return pd.DataFrame()

        clusters = []
        lot_ids = rows["Lot_ID"].unique()

        for lot_id in lot_ids:
            lot_rows = rows[rows["Lot_ID"] == lot_id]

            # Identify anomalous components (max A_score >= threshold)
            comp_anomalies = lot_rows.groupby("Component_ID")["A_score"].max()
            anomalous_comps = comp_anomalies[comp_anomalies >= threshold].index.values

            if len(anomalous_comps) < 2:
                continue  # No cluster (need at least 2)

            # Get coordinates of anomalous components
            anomalous_rows = lot_rows[lot_rows["Component_ID"].isin(anomalous_comps)]
            unique_anomalies = anomalous_rows[
                ["Component_ID", "die_x", "die_y"]
            ].drop_duplicates()

            if len(unique_anomalies) < 2:
                continue

            coords = unique_anomalies[["die_x", "die_y"]].to_numpy(float)
            comp_ids = unique_anomalies["Component_ID"].values

            # Simple clustering: if 3+ anomalies within 500μm, mark as cluster
            distances = cdist(coords, coords, metric="euclidean")
            for i, comp_id in enumerate(comp_ids):
                nearby = (distances[i] <= self.distance_threshold) & (distances[i] > 0)
                if nearby.sum() >= 2:  # At least 2 neighbors within distance
                    clusters.append(
                        {
                            "lot_id": lot_id,
                            "Component_ID": comp_id,
                            "cluster_type": "spatial_cluster",
                            "cluster_size": nearby.sum() + 1,
                        }
                    )

        return pd.DataFrame(clusters) if clusters else pd.DataFrame()
