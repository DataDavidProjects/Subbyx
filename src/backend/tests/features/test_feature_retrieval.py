from __future__ import annotations

from datetime import timezone
from pathlib import Path

import pandas as pd
import pytest


DATA_DIR = Path(__file__).parents[4] / "data"
FEATURE_SOURCES_DIR = (
    DATA_DIR.parent / "src" / "backend" / "feature_repo" / "data" / "sources"
)


def _get_store():
    from services.fraud.features.store import store

    if store is None:
        pytest.skip("Feast store not available")
    return store


def _single_entity_df(email: str, event_timestamp: object) -> pd.DataFrame:
    return pd.DataFrame({"email": [email], "event_timestamp": [event_timestamp]})


def _assert_equal_or_both_nan(actual: object, expected: object) -> None:
    if pd.isna(actual):
        assert pd.isna(expected), f"Expected {expected}, got {actual}"
        return
    assert actual == expected, f"Expected {expected}, got {actual}"


class TestHistoricalFeatureRetrieval:
    """Tests for historical feature retrieval - verify Feast returns correct values."""

    def test_charge_stats_aggregation_correct(self) -> None:
        """Verify charge stats are correctly aggregated from Feast."""
        store = _get_store()

        charges = pd.read_csv(DATA_DIR / "01-clean" / "charges.csv")
        email_with_history = charges["email"].value_counts().head(1).index[0]
        expected_count = len(charges[charges["email"] == email_with_history])

        entity_df = _single_entity_df(
            email=email_with_history,
            event_timestamp=pd.Timestamp("2025-01-01", tz=timezone.utc),
        )

        features = store.get_historical_features(
            entity_df=entity_df,
            features=["charge_stats_features:n_charges"],
            full_feature_names=True,
        )
        result_df = features.to_df()

        actual_count = result_df["charge_stats_features__n_charges"].iloc[0]
        assert actual_count == expected_count, (
            f"Expected {expected_count} charges, got {actual_count}"
        )

    def test_checkout_features_match_source_data(self) -> None:
        """Verify checkout features match the source parquet data."""
        store = _get_store()

        checkouts = pd.read_parquet(FEATURE_SOURCES_DIR / "checkouts.parquet")
        sample_row = checkouts.iloc[0]

        entity_df = _single_entity_df(
            email=sample_row["email"],
            event_timestamp=sample_row["created"],
        )

        features = store.get_historical_features(
            entity_df=entity_df,
            features=["checkout_features:subscription_value"],
            full_feature_names=True,
        )
        result_df = features.to_df()

        actual_value = result_df["checkout_features__subscription_value"].iloc[0]
        expected_value = sample_row["subscription_value"]
        _assert_equal_or_both_nan(actual_value, expected_value)

    def test_temporal_lookup_only_returns_prior_history(self) -> None:
        """Verify Feast only returns data PRIOR to event_timestamp, not future."""
        store = _get_store()

        charges = pd.read_csv(DATA_DIR / "01-clean" / "charges.csv")
        email = charges["email"].value_counts().head(1).index[0]
        charge_times = charges[charges["email"] == email]["created"]
        earliest_charge = pd.to_datetime(charge_times.min())
        later_charge = pd.to_datetime(charge_times.max())

        entity_df_early = _single_entity_df(
            email=email,
            event_timestamp=earliest_charge.to_pydatetime(),
        )

        entity_df_later = _single_entity_df(
            email=email,
            event_timestamp=later_charge.to_pydatetime(),
        )

        features_early = store.get_historical_features(
            entity_df=entity_df_early,
            features=["charge_stats_features:n_charges"],
            full_feature_names=True,
        )
        features_later = store.get_historical_features(
            entity_df=entity_df_later,
            features=["charge_stats_features:n_charges"],
            full_feature_names=True,
        )

        result_early = features_early.to_df()["charge_stats_features__n_charges"].iloc[0]
        result_later = features_later.to_df()["charge_stats_features__n_charges"].iloc[0]

        assert result_later >= result_early, (
            "Later timestamp should have equal or more charge history"
        )
        assert result_early >= 1, "Should have at least 1 charge at earliest timestamp"


class TestOnlineFeatureRetrieval:
    """Tests for online feature retrieval - verify correct values from Feast."""

    def test_online_lookup_returns_same_as_historical(self) -> None:
        """Online lookup should return same as historical when querying after all data."""
        from services.fraud.features import get_features
        store = _get_store()

        charges = pd.read_csv(DATA_DIR / "01-clean" / "charges.csv")
        email = charges["email"].value_counts().head(1).index[0]

        # Query historical at a date AFTER all charge data
        entity_df = _single_entity_df(
            email=email,
            event_timestamp=pd.Timestamp("2025-01-01", tz=timezone.utc),
        )

        # Use failure_rate — it's in the production feature service
        historical = store.get_historical_features(
            entity_df=entity_df,
            features=["charge_stats_features:failure_rate"],
            full_feature_names=True,
        ).to_df()

        online = get_features(email=email)

        historical_value = historical["charge_stats_features__failure_rate"].iloc[0]
        online_value = online.get("charge_stats_features__failure_rate")

        # Historical (at time after all data) should equal Online (latest)
        assert historical_value == online_value, (
            f"Historical={historical_value}, Online={online_value}"
        )

    def test_online_lookup_with_unknown_email_returns_nulls(self) -> None:
        """Unknown email should return features with null/zero values."""
        from services.fraud.features import get_features

        result = get_features(email="this-email-does-not-exist@example.com")

        assert isinstance(result, dict)
        for key, value in result.items():
            if key.startswith("charge_stats") or key.startswith("customer_features"):
                assert value is None or value == 0, f"Expected null/0 for {key}, got {value}"


class TestFeatureDataIntegrity:
    """Tests to verify data integrity in Feast sources."""

    def test_charge_source_has_no_duplicate_emails_per_timestamp(self) -> None:
        """Charge source should not have duplicate rows for same email-timestamp."""
        charges = pd.read_parquet(FEATURE_SOURCES_DIR / "charges.parquet")

        duplicates = charges.groupby(["email", "created"]).size()
        dup_count = (duplicates > 1).sum()

        assert dup_count == 0, f"Found {dup_count} duplicate email-timestamp pairs in charges"

    def test_customer_emails_are_valid_format(self) -> None:
        """Customer emails should be valid format (contain @)."""
        customers = pd.read_csv(DATA_DIR / "01-clean" / "customers.csv")

        invalid_emails = customers[
            customers["email"].notna() & ~customers["email"].str.contains("@", na=False)
        ]

        assert len(invalid_emails) == 0, f"Found {len(invalid_emails)} invalid emails"

    def test_checkout_timestamps_are_chronological(self) -> None:
        """Checkouts should have valid chronological timestamps."""
        checkouts = pd.read_parquet(FEATURE_SOURCES_DIR / "checkouts.parquet")

        checkouts["created"] = pd.to_datetime(checkouts["created"])
        min_date = checkouts["created"].min()
        max_date = checkouts["created"].max()

        assert min_date.year >= 2023, f"Min date {min_date} is too early"
        assert max_date.year <= 2025, f"Max date {max_date} is too late"
