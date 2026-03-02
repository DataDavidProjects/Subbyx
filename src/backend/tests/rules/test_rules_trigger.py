from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import replace

import pytest
import pandas as pd

# Add src/backend to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from routes.fraud.schemas import CheckoutRequest
from services.fraud.context import CheckoutContext
from services.fraud.inference.model import ScoringResult


@pytest.fixture
def mock_checkout_context():
    return CheckoutContext(
        checkout_id="test_checkout",
        customer_id="test_customer",
        email="clean@example.com",
        store_id="test_store",
        card_fingerprint="fingerprint_123",
        fiscal_code="FISCAL123",
        timestamp="2024-12-01T12:00:00",
        gender="M",
        birth_date="1990-01-01",
        birth_province="MI",
        birth_country="IT",
        has_high_end_device=True,
        subscription_value=100.0,
        grade="A",
        category="test"
    )


@pytest.fixture
def mock_scoring_result():
    return ScoringResult(
        score=0.1,
        scored_by="production",
        production_score=0.1,
        shadow_score=0.15,
        features={"feat1": 1.0}
    )


@pytest.fixture
def empty_checkouts_df():
    return pd.DataFrame({
        "customer": [],
        "status": [],
        "mode": [],
        "created": []
    })


class TestRulesTrigger:
    """Tests that fraud rules correctly trigger BLOCK decisions."""

    @patch("services.fraud.context.resolve_checkout")
    @patch("services.fraud.inference.model.score_models")
    @patch("services.fraud.features.get_features")
    @patch("routes.fraud.checkout.load_checkouts")
    def test_blacklist_trigger(self, mock_load_checkouts, mock_get_features, mock_score_models, mock_resolve, mock_checkout_context, empty_checkouts_df):
        """Rule: Blacklist should trigger BLOCK."""
        from routes.fraud.checkout import fraud_checkout
        
        # Mock resolve_checkout to return a blacklisted email
        ctx = replace(mock_checkout_context, email="fraud@example.com")
        mock_resolve.return_value = ctx
        
        # Mock other dependencies
        mock_load_checkouts.return_value = empty_checkouts_df

        request = CheckoutRequest(checkout_id="test_checkout")
        response = fraud_checkout(request)

        assert response.decision == "BLOCK"
        assert response.rule_triggered == "blacklist"
        assert "blacklist" in response.reason.lower()
        # Ensure model was NOT called
        mock_score_models.assert_not_called()

    @patch("services.fraud.context.resolve_checkout")
    @patch("services.fraud.inference.model.score_models")
    @patch("services.fraud.features.get_features")
    @patch("routes.fraud.checkout.load_checkouts")
    def test_stripe_risk_trigger(self, mock_load_checkouts, mock_get_features, mock_score_models, mock_resolve, mock_checkout_context, empty_checkouts_df):
        """Rule: Stripe Risk 'highest' should trigger BLOCK."""
        from routes.fraud.checkout import fraud_checkout
        
        # Mock resolve_checkout to return an email with highest risk
        ctx = replace(mock_checkout_context, email="9c7886b2-b527-4545-a119-12d103583a84@hotmail.com")
        mock_resolve.return_value = ctx
        
        # Mock other dependencies
        mock_load_checkouts.return_value = empty_checkouts_df

        request = CheckoutRequest(checkout_id="test_checkout")
        response = fraud_checkout(request)

        assert response.decision == "BLOCK"
        assert response.rule_triggered == "stripe_risk"
        assert "Stripe" in response.reason
        # Ensure model was NOT called
        mock_score_models.assert_not_called()

    @patch("services.fraud.context.resolve_checkout")
    @patch("services.fraud.inference.model.score_models")
    @patch("services.fraud.features.get_features")
    @patch("routes.fraud.checkout.load_checkouts")
    def test_fiscal_code_duplicate_trigger(self, mock_load_checkouts, mock_get_features, mock_score_models, mock_resolve, mock_checkout_context, empty_checkouts_df):
        """Rule: Fiscal code used with multiple emails should trigger BLOCK."""
        from routes.fraud.checkout import fraud_checkout
        
        # FC: 002b2d76-ada1-4e88-99df-9af3fa7bc892
        ctx = replace(mock_checkout_context,
            fiscal_code="002b2d76-ada1-4e88-99df-9af3fa7bc892",
            email="new_email@test.com"
        )
        mock_resolve.return_value = ctx
        
        # Mock other dependencies
        mock_load_checkouts.return_value = empty_checkouts_df

        request = CheckoutRequest(checkout_id="test_checkout")
        response = fraud_checkout(request)

        assert response.decision == "BLOCK"
        assert response.rule_triggered == "fiscal_code_duplicate"
        assert "Fiscal code" in response.reason
        # Ensure model was NOT called
        mock_score_models.assert_not_called()

    @patch("services.fraud.context.resolve_checkout")
    @patch("services.fraud.inference.model.score_models")
    @patch("services.fraud.features.get_features")
    @patch("routes.fraud.checkout.load_checkouts")
    @patch("services.fraud.inference.model.Model.feature_columns", new_callable=PropertyMock)
    @patch("services.fraud.features.request_features.extract_request_features")
    def test_no_rules_triggered_model_called(self, mock_extract, mock_feat_cols, mock_load_checkouts, mock_get_features, mock_score_models, mock_resolve, mock_checkout_context, mock_scoring_result, empty_checkouts_df):
        """When no rules trigger, model should be called for decision."""
        from routes.fraud.checkout import fraud_checkout
        
        mock_resolve.return_value = mock_checkout_context
        mock_score_models.return_value = mock_scoring_result
        mock_get_features.return_value = {"feat1": 1.0}
        mock_extract.return_value = {}
        mock_load_checkouts.return_value = empty_checkouts_df
        mock_feat_cols.return_value = ["feat1"]

        request = CheckoutRequest(checkout_id="test_checkout")
        response = fraud_checkout(request)

        assert response.rule_triggered is None
        assert response.score == 0.1
        assert response.decision in ["APPROVE", "BLOCK"]
        mock_score_models.assert_called_once()
