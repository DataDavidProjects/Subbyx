from __future__ import annotations

import logging
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

    def fit(self, X: pd.DataFrame.Series | np, y: pd.ndarray) -> "SelectKBestMutualInfo":
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
        if y is None:
            raise ValueError("y is required for Mutual Information")

        X_numeric = X.select_dtypes(include=[np.number]).fillna(0)
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

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
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
        self._dropped_cols: list[tuple[str, float]] = []

    def _compute_vif(self, X: pd.DataFrame) -> pd.DataFrame:
        vif_data = pd.DataFrame()
        vif_data["feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        return vif_data.sort_values("VIF", ascending=False)

    def fit(self, X: pd.DataFrame, y: Any = None) -> "RemoveHighVIFFeatures":
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")

        X_numeric = X.select_dtypes(include=[np.number]).fillna(0)
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
                self._dropped_cols.append((max_feature, max_vif))
                logger.info("VIF: %s = %.2f > %.2f, dropping", max_feature, max_vif, self.threshold)
                remaining.remove(max_feature)
            else:
                break

        logger.info(
            "VIF scores (final): %s",
            {row["feature"]: f"{row['VIF']:.2f}" for _, row in vif_df.iterrows()},
        )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        dropped = [col for col, _ in self._dropped_cols]
        if dropped:
            logger.info("Dropping (high VIF): %s", dropped)
            return X.drop(columns=dropped)
        return X

    def get_dropped(self) -> list[str]:
        return [col for col, _ in self._dropped_cols]
