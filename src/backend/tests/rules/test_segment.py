from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest


class TestSegmentDetermination:
    """Tests for customer_id based segment determination."""

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

    def test_new_customer_no_prior_checkouts(self, sample_checkouts_df: pd.DataFrame) -> None:
        """Customer with no completed checkouts should be NEW."""
        from routes.fraud.checkout import determine_segment

        with patch("routes.fraud.checkout.load_checkouts") as mock_load:
            mock_load.return_value = pd.DataFrame(
                {
                    "customer": [],
                    "status": [],
                    "mode": [],
                    "created": [],
                }
            )
            segment, reason = determine_segment(
                customer_id="new_cust", timestamp="2024-12-01T00:00:00"
            )
            assert segment == "NEW_CUSTOMER"
            assert "No prior completed checkouts" in reason

    def test_returning_customer_with_completed_checkout(
        self, sample_checkouts_df: pd.DataFrame
    ) -> None:
        """Customer with completed checkout should be RETURNING."""
        from routes.fraud.checkout import determine_segment

        with patch("routes.fraud.checkout.load_checkouts") as mock_load:
            mock_load.return_value = sample_checkouts_df
            segment, reason = determine_segment(
                customer_id="cust_1", timestamp="2024-12-01T00:00:00"
            )
            assert segment == "RETURNING"
            assert "Prior completed checkout found" in reason

    def test_pit_filtering_old_checkout_not_counted(
        self, sample_checkouts_df: pd.DataFrame
    ) -> None:
        """Checkout after cutoff should not count (PIT correctness)."""
        from routes.fraud.checkout import determine_segment

        with patch("routes.fraud.checkout.load_checkouts") as mock_load:
            # Checkout in June 2025, but cutoff is Jan 2025 - should not count
            mock_load.return_value = pd.DataFrame(
                {
                    "customer": ["cust_1"],
                    "status": ["complete"],
                    "mode": ["payment"],
                    "created": ["2025-06-01"],
                }
            )
            segment, reason = determine_segment(
                customer_id="cust_1", timestamp="2025-01-01T00:00:00"
            )
            assert segment == "NEW_CUSTOMER"

    def test_pit_filtering_old_checkout_counts(self, sample_checkouts_df: pd.DataFrame) -> None:
        """Checkout before cutoff should count (PIT correctness)."""
        from routes.fraud.checkout import determine_segment

        with patch("routes.fraud.checkout.load_checkouts") as mock_load:
            # Checkout in June 2024, cutoff is Dec 2024 - should count
            mock_load.return_value = pd.DataFrame(
                {
                    "customer": ["cust_1"],
                    "status": ["complete"],
                    "mode": ["payment"],
                    "created": ["2024-06-01"],
                }
            )
            segment, reason = determine_segment(
                customer_id="cust_1", timestamp="2024-12-01T00:00:00"
            )
            assert segment == "RETURNING"

    def test_expired_checkout_does_not_count(self, sample_checkouts_df: pd.DataFrame) -> None:
        """Expired checkout should not make customer RETURNING."""
        from routes.fraud.checkout import determine_segment

        with patch("routes.fraud.checkout.load_checkouts") as mock_load:
            mock_load.return_value = sample_checkouts_df
            segment, reason = determine_segment(
                customer_id="cust_2", timestamp="2024-12-01T00:00:00"
            )
            assert segment == "NEW_CUSTOMER"

    def test_setup_mode_does_not_count(self, sample_checkouts_df: pd.DataFrame) -> None:
        """mode=setup should not make customer RETURNING."""
        from routes.fraud.checkout import determine_segment

        with patch("routes.fraud.checkout.load_checkouts") as mock_load:
            mock_load.return_value = sample_checkouts_df
            # cust_3 has only 'setup' and one 'payment', both in July
            # But with early cutoff, should be NEW
            segment, reason = determine_segment(
                customer_id="cust_3", timestamp="2024-06-15T00:00:00"
            )
            assert segment == "NEW_CUSTOMER"

    def test_no_customer_id_returns_new(self) -> None:
        """Missing customer_id should return NEW_CUSTOMER."""
        from routes.fraud.checkout import determine_segment

        with patch("routes.fraud.checkout.load_checkouts") as mock_load:
            mock_load.return_value = pd.DataFrame(
                {
                    "customer": ["some_cust"],
                    "status": ["complete"],
                    "mode": ["payment"],
                    "created": ["2024-01-01"],
                }
            )
            segment, reason = determine_segment(customer_id="", timestamp="2024-12-01T00:00:00")
            assert segment == "NEW_CUSTOMER"
            assert "No customer_id provided" in reason

    def test_payment_mode_counts(self, sample_checkouts_df: pd.DataFrame) -> None:
        """mode=payment should count toward RETURNING."""
        from routes.fraud.checkout import determine_segment

        with patch("routes.fraud.checkout.load_checkouts") as mock_load:
            mock_load.return_value = pd.DataFrame(
                {
                    "customer": ["cust_1"],
                    "status": ["complete"],
                    "mode": ["payment"],
                    "created": ["2024-01-01"],
                }
            )
            segment, reason = determine_segment(
                customer_id="cust_1", timestamp="2024-12-01T00:00:00"
            )
            assert segment == "RETURNING"

    def test_subscription_mode_counts(self, sample_checkouts_df: pd.DataFrame) -> None:
        """mode=subscription should count toward RETURNING."""
        from routes.fraud.checkout import determine_segment

        with patch("routes.fraud.checkout.load_checkouts") as mock_load:
            mock_load.return_value = pd.DataFrame(
                {
                    "customer": ["cust_1"],
                    "status": ["complete"],
                    "mode": ["subscription"],
                    "created": ["2024-01-01"],
                }
            )
            segment, reason = determine_segment(
                customer_id="cust_1", timestamp="2024-12-01T00:00:00"
            )
            assert segment == "RETURNING"

    def test_multiple_completed_checkouts(self, sample_checkouts_df: pd.DataFrame) -> None:
        """Multiple completed checkouts should still be RETURNING."""
        from routes.fraud.checkout import determine_segment

        with patch("routes.fraud.checkout.load_checkouts") as mock_load:
            mock_load.return_value = sample_checkouts_df
            segment, reason = determine_segment(
                customer_id="cust_1", timestamp="2024-12-01T00:00:00"
            )
            assert segment == "RETURNING"
