from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif
from statsmodels.stats.outliers_influence import variance_inflation_factor

logger = logging.getLogger(__name__)


class SelectKBestMutualInfo(BaseEstimator, TransformerMixin):
    def __init__(self, min_score: float = 0.01, k: int | None = None):
        self.min_score = min_score
        self.k = k
        self._selected_cols: list[str] = []
        self._scores: dict[str, float] = {}
        self.feature_names_in_: np.ndarray | None = None

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> "SelectKBestMutualInfo":
        if y is None:
            raise ValueError("y is required for Mutual Information")

        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.values
            X_numeric = X.select_dtypes(include=[np.number]).fillna(0)
        else:
            # If numpy, we hope we have feature names from a previous step if it's a pipeline
            # but VarianceThreshold might have dropped some.
            # Best to use set_output(transform="pandas") globally if possible,
            # or ensure we always pass DataFrames.
            X_numeric = pd.DataFrame(X).fillna(0)
            if hasattr(self, "feature_names_in_") and self.feature_names_in_ is not None:
                X_numeric.columns = self.feature_names_in_

        if X_numeric.shape[1] == 0:
            self._selected_cols = []
            return self

        y_array = np.asarray(y)
        mi_scores = mutual_info_classif(
            X_numeric, y_array, discrete_features=False, random_state=42
        )
        self._scores = {col: score for col, score in zip(X_numeric.columns, mi_scores)}

        logger.info(
            "Mutual Information scores: %s", {k: f"{v:.4f}" for k, v in self._scores.items()}
        )

        sorted_cols = sorted(self._scores.keys(), key=lambda c: self._scores[c], reverse=True)

        if self.k is not None:
            self._selected_cols = sorted_cols[: self.k]
        else:
            self._selected_cols = [
                col for col in sorted_cols if self._scores[col] >= self.min_score
            ]

        dropped = set(X_numeric.columns) - set(self._selected_cols)
        if dropped:
            logger.info("Dropped (low MI): %s", list(dropped))

        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names_in_)

        if not self._selected_cols:
            return X.iloc[:, :0]
        return X[self._selected_cols]

    def get_selected(self) -> list[str]:
        return self._selected_cols

    def get_scores(self) -> dict[str, float]:
        return self._scores


class RemoveHighVIFFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, threshold: float = 10.0):
        self.threshold = threshold
        self._dropped_cols: list[str] = []
        self.feature_names_in_: np.ndarray | None = None

    def _compute_vif(self, X: pd.DataFrame) -> pd.DataFrame:
        vif_data = pd.DataFrame()
        vif_data["feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        return vif_data.sort_values("VIF", ascending=False)

    def fit(self, X: pd.DataFrame | np.ndarray, y: Any = None) -> "RemoveHighVIFFeatures":
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.values
            X_numeric = X.select_dtypes(include=[np.number]).fillna(0)
        else:
            X_numeric = pd.DataFrame(X).fillna(0)
            if hasattr(self, "feature_names_in_") and self.feature_names_in_ is not None:
                X_numeric.columns = self.feature_names_in_

        if X_numeric.shape[1] < 2:
            self._dropped_cols = []
            return self

        self._dropped_cols = []
        remaining = list(X_numeric.columns)

        while True:
            if len(remaining) < 2:
                break

            X_remaining = X_numeric[remaining]
            vif_df = self._compute_vif(X_remaining)

            max_vif_row = vif_df.iloc[0]
            max_vif = max_vif_row["VIF"]
            max_feature = max_vif_row["feature"]

            if max_vif > self.threshold:
                self._dropped_cols.append(max_feature)
                logger.info("VIF: %s = %.2f > %.2f, dropping", max_feature, max_vif, self.threshold)
                remaining.remove(max_feature)
            else:
                break

        logger.info(
            "VIF scores (final): %s",
            {row["feature"]: f"{row['VIF']:.2f}" for _, row in vif_df.iterrows()},
        )

        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names_in_)

        if self._dropped_cols:
            logger.info("Dropping (high VIF): %s", self._dropped_cols)
            return X.drop(columns=self._dropped_cols)
        return X

    def get_dropped(self) -> list[str]:
        return self._dropped_cols


class CorrelationGroupPruner(BaseEstimator, TransformerMixin):
    """
    Prunes highly correlated features within rolling-window groups (e.g. _1d, _7d, _30d).
    Keeps the feature with the highest Mutual Information score within each group.
    """

    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold
        self._dropped_cols: list[str] = []
        self.feature_names_in_: np.ndarray | None = None

    def _get_feature_group(self, name: str) -> str:
        """Strip trailing _Nd suffix to group rolling-window features."""
        return re.sub(r"_\d+d$", "", name)

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray | None = None) -> "CorrelationGroupPruner":
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.values
            X_df = X
        else:
            X_df = pd.DataFrame(X)
            if hasattr(self, "feature_names_in_") and self.feature_names_in_ is not None:
                X_df.columns = self.feature_names_in_

        if y is None:
            logger.warning("CorrelationGroupPruner: y is None, MI scores cannot be computed. Using variance.")
            scores = X_df.var().to_dict()
        else:
            y_array = np.asarray(y)
            mi_scores = mutual_info_classif(X_df.fillna(0), y_array, discrete_features=False, random_state=42)
            scores = {col: score for col, score in zip(X_df.columns, mi_scores)}

        groups: dict[str, list[str]] = {}
        for feat in X_df.columns:
            grp = self._get_feature_group(feat)
            groups.setdefault(grp, []).append(feat)

        self._dropped_cols = []
        for _, members in groups.items():
            if len(members) < 2:
                continue

            corr_matrix = X_df[members].corr().abs()
            # Sort members by score descending
            sorted_members = sorted(members, key=lambda f: scores.get(f, 0), reverse=True)
            best = sorted_members[0]

            for feat in sorted_members[1:]:
                if corr_matrix.loc[feat, best] > self.threshold:
                    self._dropped_cols.append(feat)
                    logger.info(
                        "Corr: %s (score=%.4f) corr=%.3f with %s (score=%.4f) -> dropping",
                        feat,
                        scores.get(feat, 0),
                        corr_matrix.loc[feat, best],
                        best,
                        scores.get(best, 0),
                    )

        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names_in_)

        if self._dropped_cols:
            return X.drop(columns=self._dropped_cols)
        return X

    def get_dropped(self) -> list[str]:
        return self._dropped_cols
