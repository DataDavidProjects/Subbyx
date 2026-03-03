from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest


class TestSegmentDetermination:
    """Tests for customer_id based segment determination."""

    @pytest.fixture
    def determine_segment(self):
        from routes.fraud.checkout import determine_segment as _determine_segment

        return _determine_segment

    @pytest.fixture
    def sample_checkouts_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "id": ["c1", "c2", "c3", "c4", "c5"],
                "customer": ["cust_1", "cust_1", "cust_2", "cust_3", "cust_3"],
                "status": ["complete", "complete", "expired", "complete", "complete"],
                "mode": ["payment", "subscription", "payment", "setup", "payment"],
                "created": ["2024-01-01", "2024-06-01", "2024-03-01", "2024-02-01", "2024-07-01"],
            }
        )

    @staticmethod
    def _run_segment(
        determine_segment,
        checkouts_df: pd.DataFrame,
        customer_id: str,
        timestamp: str,
    ) -> tuple[str, str]:
        with patch("routes.fraud.checkout.load_checkouts", return_value=checkouts_df):
            return determine_segment(customer_id=customer_id, timestamp=timestamp)

    def test_new_customer_no_prior_checkouts(self, determine_segment) -> None:
        """Customer with no completed checkouts should be NEW_CUSTOMER."""
        empty_checkouts = pd.DataFrame({"customer": [], "status": [], "mode": [], "created": []})
        segment, reason = self._run_segment(
            determine_segment=determine_segment,
            checkouts_df=empty_checkouts,
            customer_id="new_cust",
            timestamp="2024-12-01T00:00:00",
        )

        assert segment == "NEW_CUSTOMER"
        assert "No prior completed checkouts" in reason

    def test_returning_customer_with_completed_checkout(
        self, determine_segment, sample_checkouts_df: pd.DataFrame
    ) -> None:
        """Customer with completed checkout should be RETURNING."""
        segment, reason = self._run_segment(
            determine_segment=determine_segment,
            checkouts_df=sample_checkouts_df,
            customer_id="cust_1",
            timestamp="2024-12-01T00:00:00",
        )
        assert segment == "RETURNING"
        assert "Prior completed checkout found" in reason

    def test_pit_filtering_old_checkout_not_counted(
        self, determine_segment
    ) -> None:
        """Checkout after cutoff should not count (PIT correctness)."""
        checkouts_df = pd.DataFrame(
            {"customer": ["cust_1"], "status": ["complete"], "mode": ["payment"], "created": ["2025-06-01"]}
        )
        segment, _ = self._run_segment(
            determine_segment=determine_segment,
            checkouts_df=checkouts_df,
            customer_id="cust_1",
            timestamp="2025-01-01T00:00:00",
        )
        assert segment == "NEW_CUSTOMER"

    def test_pit_filtering_old_checkout_counts(self, determine_segment) -> None:
        """Checkout before cutoff should count (PIT correctness)."""
        checkouts_df = pd.DataFrame(
            {"customer": ["cust_1"], "status": ["complete"], "mode": ["payment"], "created": ["2024-06-01"]}
        )
        segment, _ = self._run_segment(
            determine_segment=determine_segment,
            checkouts_df=checkouts_df,
            customer_id="cust_1",
            timestamp="2024-12-01T00:00:00",
        )
        assert segment == "RETURNING"

    def test_expired_checkout_does_not_count(self, determine_segment, sample_checkouts_df: pd.DataFrame) -> None:
        """Expired checkout should not make customer RETURNING."""
        segment, _ = self._run_segment(
            determine_segment=determine_segment,
            checkouts_df=sample_checkouts_df,
            customer_id="cust_2",
            timestamp="2024-12-01T00:00:00",
        )
        assert segment == "NEW_CUSTOMER"

    def test_setup_mode_does_not_count(self, determine_segment, sample_checkouts_df: pd.DataFrame) -> None:
        """mode=setup should not make customer RETURNING."""
        segment, _ = self._run_segment(
            determine_segment=determine_segment,
            checkouts_df=sample_checkouts_df,
            customer_id="cust_3",
            timestamp="2024-06-15T00:00:00",
        )
        assert segment == "NEW_CUSTOMER"

    def test_no_customer_id_returns_new(self, determine_segment) -> None:
        """Missing customer_id should return NEW_CUSTOMER."""
        checkouts_df = pd.DataFrame(
            {
                "customer": ["some_cust"],
                "status": ["complete"],
                "mode": ["payment"],
                "created": ["2024-01-01"],
            }
        )
        segment, reason = self._run_segment(
            determine_segment=determine_segment,
            checkouts_df=checkouts_df,
            customer_id="",
            timestamp="2024-12-01T00:00:00",
        )
        assert segment == "NEW_CUSTOMER"
        assert "No customer_id provided" in reason

    def test_payment_mode_counts(self, determine_segment) -> None:
        """mode=payment should count toward RETURNING."""
        checkouts_df = pd.DataFrame(
            {"customer": ["cust_1"], "status": ["complete"], "mode": ["payment"], "created": ["2024-01-01"]}
        )
        segment, _ = self._run_segment(
            determine_segment=determine_segment,
            checkouts_df=checkouts_df,
            customer_id="cust_1",
            timestamp="2024-12-01T00:00:00",
        )
        assert segment == "RETURNING"

    def test_subscription_mode_counts(self, determine_segment) -> None:
        """mode=subscription should count toward RETURNING."""
        checkouts_df = pd.DataFrame(
            {
                "customer": ["cust_1"],
                "status": ["complete"],
                "mode": ["subscription"],
                "created": ["2024-01-01"],
            }
        )
        segment, _ = self._run_segment(
            determine_segment=determine_segment,
            checkouts_df=checkouts_df,
            customer_id="cust_1",
            timestamp="2024-12-01T00:00:00",
        )
        assert segment == "RETURNING"

    def test_multiple_completed_checkouts(self, determine_segment, sample_checkouts_df: pd.DataFrame) -> None:
        """Multiple completed checkouts should still be RETURNING."""
        segment, _ = self._run_segment(
            determine_segment=determine_segment,
            checkouts_df=sample_checkouts_df,
            customer_id="cust_1",
            timestamp="2024-12-01T00:00:00",
        )
        assert segment == "RETURNING"
