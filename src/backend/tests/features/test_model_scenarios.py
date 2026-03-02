from __future__ import annotations

import pytest

from services.fraud.inference.model import production_model


class TestModelScoringLogic:
    """Test that model scoring logic is correct.

    These tests verify the model works with training-range values.
    They don't depend on Feast retrieval or current data state.
    """

    def test_model_discriminates_fraud_vs_clean(self) -> None:
        """Model should score fraud higher than clean when using realistic feature values.

        Sets feature values based on what fraud vs clean profiles look like:
        - Fraud: high geo fraud rates, low success rate, high failure rate,
          low identity-match scores, no payment history (missing indicator=1).
        - Clean: zero geo fraud, high success rate, high identity-match scores.
        """
        cols = production_model.feature_columns

        # --- Fraud profile: risky geo, poor payment history, weak identity ---
        fraud_features = {col: 0.0 for col in cols}
        # Geo-velocity: region with high recent fraud activity
        for col in cols:
            if "fraud_rate" in col:
                fraud_features[col] = 0.5
            elif "n_frauds" in col:
                fraud_features[col] = 5.0
            elif "n_requests" in col:
                fraud_features[col] = 10.0
        # Payment: low success, high failure
        fraud_features.update({k: v for k, v in {
            "payment_intent_stats_features__success_rate": 0.1,
            "payment_intent_stats_features__failure_rate": 0.6,
            "payment_intent_stats_features__n_succeeded": 0,
            "payment_intent_stats_features__n_payment_intents": 5,
            "charge_stats_features__failure_rate": 0.5,
            "payment_intent_features__amount": 200,
            "subscription_value": 200,
            "payment_intent_features__subscription_value": 200,
            # Identity: low match scores
            "customer_features__doc_name_email_match_score": 0.1,
            "customer_features__email_emails_match_score": 0.1,
            "pi_features__missing": 0,
        }.items() if k in cols})

        # --- Clean profile: safe geo, good payment history, strong identity ---
        clean_features = {col: 0.0 for col in cols}
        clean_features.update({k: v for k, v in {
            "payment_intent_stats_features__success_rate": 0.95,
            "payment_intent_stats_features__failure_rate": 0.0,
            "payment_intent_stats_features__n_succeeded": 10,
            "payment_intent_stats_features__n_payment_intents": 10,
            "charge_stats_features__failure_rate": 0.0,
            "payment_intent_features__amount": 30,
            "subscription_value": 30,
            "payment_intent_features__subscription_value": 30,
            # Identity: high match scores
            "customer_features__doc_name_email_match_score": 0.95,
            "customer_features__email_emails_match_score": 0.95,
            "pi_features__missing": 0,
        }.items() if k in cols})

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
