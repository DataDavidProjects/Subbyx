from __future__ import annotations

from contextlib import ExitStack
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

# Add src/backend to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from routes.fraud.schemas import CheckoutRequest
from services.fraud.context import CheckoutContext
from services.fraud.inference.model import ScoringResult


@pytest.fixture
def mock_checkout_context() -> CheckoutContext:
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
        category="test",
        card_brand="visa",
        card_funding="credit",
        card_cvc_check="pass",
    )


@pytest.fixture
def mock_scoring_result() -> ScoringResult:
    return ScoringResult(
        score=0.1,
        scored_by="production",
        production_score=0.1,
        shadow_score=0.15,
        features={"feat1": 1.0},
    )


@pytest.fixture
def empty_checkouts_df() -> pd.DataFrame:
    return pd.DataFrame({"customer": [], "status": [], "mode": [], "created": []})


class TestRulesTrigger:
    """Tests that fraud rules correctly trigger BLOCK decisions."""

    @staticmethod
    def _run_checkout(
        *,
        ctx: CheckoutContext,
        checkouts_df: pd.DataFrame,
        blacklist: set[str] | None = None,
        high_risk_emails: set[str] | None = None,
        fiscal_code_mapping: dict[str, set[str]] | None = None,
        feast_features: dict | None = None,
        score_result: ScoringResult | None = None,
        request_features: dict | None = None,
    ):
        from routes.fraud.checkout import fraud_checkout

        with ExitStack() as stack:
            stack.enter_context(patch("routes.fraud.checkout.resolve_checkout", return_value=ctx))
            stack.enter_context(patch("routes.fraud.checkout.load_checkouts", return_value=checkouts_df))
            stack.enter_context(patch("routes.fraud.checkout.load_blacklist", return_value=blacklist or set()))
            stack.enter_context(
                patch(
                    "routes.fraud.checkout.load_charges_with_highest_risk",
                    return_value=high_risk_emails or set(),
                )
            )
            stack.enter_context(
                patch(
                    "routes.fraud.checkout.load_fiscal_code_to_emails",
                    return_value=fiscal_code_mapping or {},
                )
            )
            stack.enter_context(
                patch(
                    "routes.fraud.checkout.get_feast_features",
                    return_value=feast_features or {},
                )
            )
            stack.enter_context(
                patch(
                    "routes.fraud.checkout.extract_request_features",
                    return_value=request_features or {},
                )
            )
            stack.enter_context(patch("routes.fraud.checkout.get_feature_metadata", return_value={}))
            stack.enter_context(
                patch(
                    "routes.fraud.checkout.production_model",
                    new=SimpleNamespace(feature_columns=["feat1"]),
                )
            )
            mock_score_models = stack.enter_context(
                patch("routes.fraud.checkout.score_models", return_value=score_result)
            )
            response = fraud_checkout(CheckoutRequest(checkout_id="test_checkout"))

        return response, mock_score_models

    def test_blacklist_trigger(
        self,
        mock_checkout_context: CheckoutContext,
        empty_checkouts_df: pd.DataFrame,
    ) -> None:
        ctx = replace(mock_checkout_context, email="fraud@example.com")

        response, mock_score_models = self._run_checkout(
            ctx=ctx,
            checkouts_df=empty_checkouts_df,
            blacklist={"fraud@example.com"},
        )

        assert response.decision == "BLOCK"
        assert response.rule_triggered == "blacklist"
        assert "blacklist" in response.reason.lower()
        mock_score_models.assert_not_called()

    def test_stripe_risk_trigger(
        self,
        mock_checkout_context: CheckoutContext,
        empty_checkouts_df: pd.DataFrame,
    ) -> None:
        risky_email = "9c7886b2-b527-4545-a119-12d103583a84@hotmail.com"
        ctx = replace(mock_checkout_context, email=risky_email)

        response, mock_score_models = self._run_checkout(
            ctx=ctx,
            checkouts_df=empty_checkouts_df,
            high_risk_emails={risky_email},
        )

        assert response.decision == "BLOCK"
        assert response.rule_triggered == "stripe_risk"
        assert "highest risk" in response.reason.lower()
        mock_score_models.assert_not_called()

    def test_fiscal_code_duplicate_trigger(
        self,
        mock_checkout_context: CheckoutContext,
        empty_checkouts_df: pd.DataFrame,
    ) -> None:
        fiscal_code = "002b2d76-ada1-4e88-99df-9af3fa7bc892"
        ctx = replace(mock_checkout_context, fiscal_code=fiscal_code, email="new_email@test.com")

        response, mock_score_models = self._run_checkout(
            ctx=ctx,
            checkouts_df=empty_checkouts_df,
            fiscal_code_mapping={fiscal_code: {"old@test.com", "other@test.com"}},
        )

        assert response.decision == "BLOCK"
        assert response.rule_triggered == "fiscal_code_duplicate"
        assert "fiscal code" in response.reason.lower()
        mock_score_models.assert_not_called()

    def test_no_rules_triggered_model_called(
        self,
        mock_checkout_context: CheckoutContext,
        mock_scoring_result: ScoringResult,
        empty_checkouts_df: pd.DataFrame,
    ) -> None:
        response, mock_score_models = self._run_checkout(
            ctx=mock_checkout_context,
            checkouts_df=empty_checkouts_df,
            feast_features={"feat1": 1.0},
            score_result=mock_scoring_result,
            request_features={},
        )

        assert response.rule_triggered is None
        assert response.score == 0.1
        assert response.decision in ["APPROVE", "BLOCK"]
        mock_score_models.assert_called_once()

    def test_payment_failure_rule_trigger(
        self,
        mock_checkout_context: CheckoutContext,
    ) -> None:
        completed_checkouts_df = pd.DataFrame(
            {
                "customer": ["test_customer"],
                "status": ["complete"],
                "mode": ["subscription"],
                "created": ["2024-01-01T00:00:00"],
            }
        )

        high_failure_features = {
            "charge_stats_features__failure_rate": 0.85,
            "charge_stats_features__n_charges": 20.0,
            "payment_intent_stats_features__failure_rate": 0.0,
            "payment_intent_stats_features__n_payment_intents": 0.0,
        }

        response, mock_score_models = self._run_checkout(
            ctx=mock_checkout_context,
            checkouts_df=completed_checkouts_df,
            feast_features=high_failure_features,
        )

        assert response.decision == "BLOCK"
        assert response.rule_triggered == "payment_failure"
        assert "failure rate" in response.reason.lower()
        assert response.score is None
        mock_score_models.assert_not_called()
