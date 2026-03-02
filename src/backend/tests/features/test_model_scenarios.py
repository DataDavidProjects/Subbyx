from __future__ import annotations

import pytest

from services.fraud.inference.model import production_model


class TestModelScoringLogic:
    """Test that model scoring logic is correct.

    These tests verify the model works with training-range values.
    They don't depend on Feast retrieval or current data state.
    """

    def test_model_discriminates_fraud_vs_clean(self) -> None:
        """Model should score fraud higher than clean when using training mean values."""
        cols = production_model.feature_columns

        fraud_features = {col: 0.0 for col in cols}
        fraud_features["payment_intent_stats_features__success_rate"] = 0.472
        fraud_features["payment_intent_stats_features__n_succeeded"] = 1.791
        fraud_features["charge_features__outcome_risk_score"] = 1.458
        fraud_features["payment_intent_features__subscription_value"] = 21.471
        fraud_features["charge_stats_features__failure_rate"] = 0.205
        fraud_features["payment_intent_features__amount"] = 0.4
        fraud_features["payment_intent_stats_features__n_payment_intents"] = 2.646
        fraud_features["checkout_features__subscription_value"] = 32.706

        clean_features = {col: 0.0 for col in cols}
        clean_features["payment_intent_stats_features__success_rate"] = 0.148
        clean_features["payment_intent_stats_features__n_succeeded"] = 0.382
        clean_features["charge_features__outcome_risk_score"] = 0.420
        clean_features["payment_intent_features__subscription_value"] = 7.348
        clean_features["charge_stats_features__failure_rate"] = 0.170
        clean_features["payment_intent_features__amount"] = 0.2
        clean_features["payment_intent_stats_features__n_payment_intents"] = 1.145
        clean_features["checkout_features__subscription_value"] = 18.470

        fraud_score = production_model.predict(fraud_features)
        clean_score = production_model.predict(clean_features)

        assert fraud_score > clean_score, (
            f"Fraud should score higher than clean. Fraud: {fraud_score:.3f}, Clean: {clean_score:.3f}"
        )

    def test_new_customer_no_history_scores_low(self) -> None:
        """Customers with no payment history should score low."""
        cols = production_model.feature_columns

        new_customer = {col: 0.0 for col in cols}
        score = production_model.predict(new_customer)

        assert score < 0.2, f"New customer should score low, got {score:.3f}"

    def test_model_output_is_valid_probability(self) -> None:
        """Model output should be valid probability [0, 1]."""
        cols = production_model.feature_columns

        test_features = {col: 0.5 for col in cols}
        score = production_model.predict(test_features)

        assert 0 <= score <= 1, f"Score {score} not in valid range [0, 1]"

    def test_null_features_handled_gracefully(self) -> None:
        """Model should handle null/missing features."""
        cols = production_model.feature_columns

        features = {col: None for col in cols}
        score = production_model.predict(features)

        assert 0 <= score <= 1, f"Score {score} not in valid range [0, 1]"
