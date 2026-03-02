from __future__ import annotations

from datetime import timezone
from pathlib import Path

import pandas as pd
import pytest


DATA_DIR = Path(__file__).parents[4] / "data"


def get_store():
    from services.fraud.features.store import store

    if store is None:
        pytest.skip("Feast store not available")
    return store


class TestCustomerIdentityFeatures:
    """Test customer identity features (require customer_id entity)."""

    def test_doc_name_email_match_retrieved(self) -> None:
        """Verify doc_name_email_match_score can be retrieved."""
        pytest.skip("customer_features requires customer_id entity - tested via integration")


class TestPaymentIntentFeatures:
    """Test payment intent features."""

    def test_payment_intent_amount_is_positive(self) -> None:
        """Verify payment_intent amount is >= 0 when present."""
        store = get_store()

        customers = pd.read_csv(DATA_DIR / "01-clean" / "customers.csv")
        payment_intents = pd.read_csv(DATA_DIR / "01-clean" / "payment_intents.csv")
        payment_intents = payment_intents.merge(
            customers[["id", "email"]], left_on="customer", right_on="id", how="left"
        )
        payment_intents = payment_intents[payment_intents["email"].notna()].head(1)
        if len(payment_intents) == 0:
            pytest.skip("No payment intents with email")

        row = payment_intents.iloc[0]

        entity_df = pd.DataFrame(
            {
                "email": [row["email"]],
                "event_timestamp": [pd.Timestamp(row["created"], tz=timezone.utc)],
            }
        )

        features = store.get_historical_features(
            entity_df=entity_df,
            features=["payment_intent_features:amount"],
            full_feature_names=True,
        )
        result_df = features.to_df()

        value = result_df["payment_intent_features__amount"].iloc[0]
        if value is not None:
            assert value >= 0, f"amount should be >= 0, got {value}"


class TestChargeStatsFeatures:
    """Test charge statistics features."""

    def test_n_charges_equals_cumulative_count(self) -> None:
        """Verify n_charges equals count of all charges for this email before timestamp."""
        store = get_store()

        charges = pd.read_csv(DATA_DIR / "01-clean" / "charges.csv")
        email = charges["email"].value_counts().head(1).index[0]
        expected_count = len(charges[charges["email"] == email])

        entity_df = pd.DataFrame(
            {
                "email": [email],
                "event_timestamp": [pd.Timestamp("2025-01-01", tz=timezone.utc)],
            }
        )

        features = store.get_historical_features(
            entity_df=entity_df,
            features=["charge_stats_features:n_charges"],
            full_feature_names=True,
        )
        result_df = features.to_df()

        actual = result_df["charge_stats_features__n_charges"].iloc[0]
        assert actual == expected_count, f"Expected {expected_count}, got {actual}"

    def test_failure_rate_is_between_0_and_1(self) -> None:
        """Verify failure_rate is always between 0 and 1."""
        store = get_store()

        charges = pd.read_csv(DATA_DIR / "01-clean" / "charges.csv")
        email = charges["email"].value_counts().head(1).index[0]

        entity_df = pd.DataFrame(
            {
                "email": [email],
                "event_timestamp": [pd.Timestamp("2025-01-01", tz=timezone.utc)],
            }
        )

        features = store.get_historical_features(
            entity_df=entity_df,
            features=["charge_stats_features:failure_rate"],
            full_feature_names=True,
        )
        result_df = features.to_df()

        value = result_df["charge_stats_features__failure_rate"].iloc[0]
        assert value is not None, "failure_rate should not be null for customer with history"
        assert 0 <= value <= 1, f"failure_rate {value} not in [0,1]"


class TestPaymentIntentStatsFeatures:
    """Test payment intent aggregated stats features."""

    def test_n_payment_intents_equals_cumulative_count(self) -> None:
        """Verify n_payment_intents equals count of all PIs for this email before timestamp."""
        store = get_store()

        customers = pd.read_csv(DATA_DIR / "01-clean" / "customers.csv")
        pis = pd.read_csv(DATA_DIR / "01-clean" / "payment_intents.csv")
        pis = pis.merge(customers[["id", "email"]], left_on="customer", right_on="id", how="left")
        pis = pis[pis["email"].notna()]

        email = pis["email"].value_counts().head(1).index[0]
        expected_count = len(pis[pis["email"] == email])

        entity_df = pd.DataFrame(
            {
                "email": [email],
                "event_timestamp": [pd.Timestamp("2025-01-01", tz=timezone.utc)],
            }
        )

        features = store.get_historical_features(
            entity_df=entity_df,
            features=["payment_intent_stats_features:n_payment_intents"],
            full_feature_names=True,
        )
        result_df = features.to_df()

        actual = result_df["payment_intent_stats_features__n_payment_intents"].iloc[0]
        assert actual == expected_count, f"Expected {expected_count}, got {actual}"

    def test_failure_rate_is_between_0_and_1(self) -> None:
        """Verify payment_intent failure_rate is always between 0 and 1."""
        store = get_store()

        customers = pd.read_csv(DATA_DIR / "01-clean" / "customers.csv")
        pis = pd.read_csv(DATA_DIR / "01-clean" / "payment_intents.csv")
        pis = pis.merge(customers[["id", "email"]], left_on="customer", right_on="id", how="left")
        pis = pis[pis["email"].notna()]

        email = pis["email"].value_counts().head(1).index[0]

        entity_df = pd.DataFrame(
            {
                "email": [email],
                "event_timestamp": [pd.Timestamp("2025-01-01", tz=timezone.utc)],
            }
        )

        features = store.get_historical_features(
            entity_df=entity_df,
            features=["payment_intent_stats_features:failure_rate"],
            full_feature_names=True,
        )
        result_df = features.to_df()

        value = result_df["payment_intent_stats_features__failure_rate"].iloc[0]
        assert value is not None, "failure_rate should not be null for customer with history"
        assert 0 <= value <= 1, f"failure_rate {value} not in [0,1]"

    def test_success_rate_is_between_0_and_1(self) -> None:
        """Verify payment_intent success_rate is always between 0 and 1."""
        store = get_store()

        customers = pd.read_csv(DATA_DIR / "01-clean" / "customers.csv")
        pis = pd.read_csv(DATA_DIR / "01-clean" / "payment_intents.csv")
        pis = pis.merge(customers[["id", "email"]], left_on="customer", right_on="id", how="left")
        pis = pis[pis["email"].notna()]

        email = pis["email"].value_counts().head(1).index[0]

        entity_df = pd.DataFrame(
            {
                "email": [email],
                "event_timestamp": [pd.Timestamp("2025-01-01", tz=timezone.utc)],
            }
        )

        features = store.get_historical_features(
            entity_df=entity_df,
            features=["payment_intent_stats_features:success_rate"],
            full_feature_names=True,
        )
        result_df = features.to_df()

        value = result_df["payment_intent_stats_features__success_rate"].iloc[0]
        assert value is not None, "success_rate should not be null for customer with history"
        assert 0 <= value <= 1, f"success_rate {value} not in [0,1]"


class TestChargeFeatures:
    """Test charge features."""

    def test_outcome_risk_score_is_between_0_and_100(self) -> None:
        """Verify Stripe risk score is in valid range [0, 100]."""
        store = get_store()

        charges = pd.read_csv(DATA_DIR / "01-clean" / "charges.csv")
        charges = charges[charges["outcome_risk_score"].notna()].head(1)
        if len(charges) == 0:
            pytest.skip("No charges with outcome_risk_score")

        row = charges.iloc[0]

        entity_df = pd.DataFrame(
            {
                "email": [row["email"]],
                "event_timestamp": [pd.Timestamp(row["created"], tz=timezone.utc)],
            }
        )

        features = store.get_historical_features(
            entity_df=entity_df,
            features=["charge_features:outcome_risk_score"],
            full_feature_names=True,
        )
        result_df = features.to_df()

        value = result_df["charge_features__outcome_risk_score"].iloc[0]
        if value is not None:
            assert 0 <= value <= 100, f"risk_score {value} not in [0,100]"


class TestCheckoutFeatures:
    """Test checkout features."""

    def test_checkout_subscription_value_matches_source(self) -> None:
        """Verify checkout subscription_value matches the checkout at that timestamp."""
        store = get_store()

        customers = pd.read_csv(DATA_DIR / "01-clean" / "customers.csv")
        checkouts = pd.read_csv(DATA_DIR / "01-clean" / "checkouts.csv")
        checkouts = checkouts.merge(
            customers[["id", "email"]], left_on="customer", right_on="id", how="left"
        )
        checkouts = checkouts[checkouts["subscription_value"].notna()]
        checkouts = checkouts[checkouts["email"].notna()].head(1)
        if len(checkouts) == 0:
            pytest.skip("No checkouts with subscription_value and email")

        row = checkouts.iloc[0]

        entity_df = pd.DataFrame(
            {
                "email": [row["email"]],
                "event_timestamp": [pd.Timestamp(row["created"], tz=timezone.utc)],
            }
        )

        features = store.get_historical_features(
            entity_df=entity_df,
            features=["checkout_features:subscription_value"],
            full_feature_names=True,
        )
        result_df = features.to_df()

        actual = result_df["checkout_features__subscription_value"].iloc[0]
        expected = row["subscription_value"]

        if pd.isna(actual):
            assert pd.isna(expected), f"Expected {expected}, got {actual}"
        else:
            assert actual == expected, f"Expected {expected}, got {actual}"


class TestStoreStatsFeatures:
    """Test store statistics features."""

    def test_store_avg_value_is_positive(self) -> None:
        """Verify store_avg_value is >= 0 when present."""
        store = get_store()

        customers = pd.read_csv(DATA_DIR / "01-clean" / "customers.csv")
        checkouts = pd.read_csv(DATA_DIR / "01-clean" / "checkouts.csv")
        checkouts = checkouts.merge(
            customers[["id", "email"]], left_on="customer", right_on="id", how="left"
        )
        checkouts = checkouts[checkouts["store_id"].notna()]
        checkouts = checkouts[checkouts["email"].notna()].head(1)
        if len(checkouts) == 0:
            pytest.skip("No checkouts with store_id and email")

        row = checkouts.iloc[0]

        entity_df = pd.DataFrame(
            {
                "email": [row["email"]],
                "store_id": [row["store_id"]],
                "event_timestamp": [pd.Timestamp(row["created"], tz=timezone.utc)],
            }
        )

        features = store.get_historical_features(
            entity_df=entity_df,
            features=["store_stats_features:store_avg_value"],
            full_feature_names=True,
        )
        result_df = features.to_df()

        value = result_df["store_stats_features__store_avg_value"].iloc[0]
        if value is not None:
            assert value >= 0, f"store_avg_value {value} should be >= 0"

    def test_store_success_rate_is_between_0_and_1(self) -> None:
        """Verify store_success_rate is in valid range."""
        store = get_store()

        customers = pd.read_csv(DATA_DIR / "01-clean" / "customers.csv")
        checkouts = pd.read_csv(DATA_DIR / "01-clean" / "checkouts.csv")
        checkouts = checkouts.merge(
            customers[["id", "email"]], left_on="customer", right_on="id", how="left"
        )
        checkouts = checkouts[checkouts["store_id"].notna()]
        checkouts = checkouts[checkouts["email"].notna()].head(1)
        if len(checkouts) == 0:
            pytest.skip("No checkouts with store_id and email")

        row = checkouts.iloc[0]

        entity_df = pd.DataFrame(
            {
                "email": [row["email"]],
                "store_id": [row["store_id"]],
                "event_timestamp": [pd.Timestamp(row["created"], tz=timezone.utc)],
            }
        )

        features = store.get_historical_features(
            entity_df=entity_df,
            features=["store_stats_features:store_success_rate"],
            full_feature_names=True,
        )
        result_df = features.to_df()

        value = result_df["store_stats_features__store_success_rate"].iloc[0]
        if value is not None:
            assert 0 <= value <= 1, f"store_success_rate {value} not in [0,1]"


class TestGeoTimeFeatures:
    """Test geo-temporal features."""

    def test_province_fraud_rate_is_valid_probability(self) -> None:
        """Verify province_fraud_rate is a valid probability [0, 1]."""
        store = get_store()

        charges = pd.read_csv(DATA_DIR / "01-clean" / "charges.csv")
        email = charges["email"].value_counts().head(1).index[0]

        entity_df = pd.DataFrame(
            {
                "email": [email],
                "event_timestamp": [pd.Timestamp("2025-01-01", tz=timezone.utc)],
            }
        )

        features = store.get_historical_features(
            entity_df=entity_df,
            features=["geo_time_features:province_fraud_rate_30d"],
            full_feature_names=True,
        )
        result_df = features.to_df()

        value = result_df["geo_time_features__province_fraud_rate_30d"].iloc[0]
        if value is not None:
            assert 0 <= value <= 1, f"province_fraud_rate {value} not in [0,1]"

    def test_province_n_requests_is_non_negative(self) -> None:
        """Verify province_n_requests is >= 0."""
        store = get_store()

        charges = pd.read_csv(DATA_DIR / "01-clean" / "charges.csv")
        email = charges["email"].value_counts().head(1).index[0]

        entity_df = pd.DataFrame(
            {
                "email": [email],
                "event_timestamp": [pd.Timestamp("2025-01-01", tz=timezone.utc)],
            }
        )

        features = store.get_historical_features(
            entity_df=entity_df,
            features=["geo_time_features:province_n_requests_60d"],
            full_feature_names=True,
        )
        result_df = features.to_df()

        value = result_df["geo_time_features__province_n_requests_60d"].iloc[0]
        if value is not None:
            assert value >= 0, f"province_n_requests {value} should be >= 0"

    def test_checkout_dow_is_valid_day(self) -> None:
        """Verify checkout_dow is a valid day of week [0-6]."""
        store = get_store()

        charges = pd.read_csv(DATA_DIR / "01-clean" / "charges.csv")
        email = charges["email"].value_counts().head(1).index[0]

        entity_df = pd.DataFrame(
            {
                "email": [email],
                "event_timestamp": [pd.Timestamp("2025-01-01", tz=timezone.utc)],
            }
        )

        features = store.get_historical_features(
            entity_df=entity_df,
            features=["geo_time_features:checkout_dow"],
            full_feature_names=True,
        )
        result_df = features.to_df()

        value = result_df["geo_time_features__checkout_dow"].iloc[0]
        assert value is not None, "checkout_dow should not be null"
        assert 0 <= value <= 6, f"checkout_dow {value} not in [0,6]"

    def test_checkout_hour_is_valid_hour(self) -> None:
        """Verify checkout_hour is a valid hour [0-23]."""
        store = get_store()

        charges = pd.read_csv(DATA_DIR / "01-clean" / "charges.csv")
        email = charges["email"].value_counts().head(1).index[0]

        entity_df = pd.DataFrame(
            {
                "email": [email],
                "event_timestamp": [pd.Timestamp("2025-01-01", tz=timezone.utc)],
            }
        )

        features = store.get_historical_features(
            entity_df=entity_df,
            features=["geo_time_features:checkout_hour"],
            full_feature_names=True,
        )
        result_df = features.to_df()

        value = result_df["geo_time_features__checkout_hour"].iloc[0]
        assert value is not None, "checkout_hour should not be null"
        assert 0 <= value <= 23, f"checkout_hour {value} not in [0,23]"

    def test_province_n_frauds_is_non_negative(self) -> None:
        """Verify province_n_frauds is >= 0."""
        store = get_store()

        charges = pd.read_csv(DATA_DIR / "01-clean" / "charges.csv")
        email = charges["email"].value_counts().head(1).index[0]

        entity_df = pd.DataFrame(
            {
                "email": [email],
                "event_timestamp": [pd.Timestamp("2025-01-01", tz=timezone.utc)],
            }
        )

        features = store.get_historical_features(
            entity_df=entity_df,
            features=["geo_time_features:province_n_frauds_10d"],
            full_feature_names=True,
        )
        result_df = features.to_df()

        value = result_df["geo_time_features__province_n_frauds_10d"].iloc[0]
        if value is not None:
            assert value >= 0, f"province_n_frauds {value} should be >= 0"

    def test_postal_n_requests_is_non_negative(self) -> None:
        """Verify postal_n_requests is >= 0."""
        store = get_store()

        charges = pd.read_csv(DATA_DIR / "01-clean" / "charges.csv")
        email = charges["email"].value_counts().head(1).index[0]

        entity_df = pd.DataFrame(
            {
                "email": [email],
                "event_timestamp": [pd.Timestamp("2025-01-01", tz=timezone.utc)],
            }
        )

        features = store.get_historical_features(
            entity_df=entity_df,
            features=["geo_time_features:postal_n_requests_30d"],
            full_feature_names=True,
        )
        result_df = features.to_df()

        value = result_df["geo_time_features__postal_n_requests_30d"].iloc[0]
        if value is not None:
            assert value >= 0, f"postal_n_requests {value} should be >= 0"

    def test_is_weekend_is_boolean(self) -> None:
        """Verify is_weekend is 0 or 1."""
        store = get_store()

        charges = pd.read_csv(DATA_DIR / "01-clean" / "charges.csv")
        email = charges["email"].value_counts().head(1).index[0]

        entity_df = pd.DataFrame(
            {
                "email": [email],
                "event_timestamp": [pd.Timestamp("2025-01-01", tz=timezone.utc)],
            }
        )

        features = store.get_historical_features(
            entity_df=entity_df,
            features=["geo_time_features:is_weekend"],
            full_feature_names=True,
        )
        result_df = features.to_df()

        value = result_df["geo_time_features__is_weekend"].iloc[0]
        assert value is not None, "is_weekend should not be null"
        assert value in [0, 1], f"is_weekend {value} should be 0 or 1"

    def test_postal_n_frauds_is_non_negative(self) -> None:
        """Verify postal_n_frauds is >= 0."""
        store = get_store()

        charges = pd.read_csv(DATA_DIR / "01-clean" / "charges.csv")
        email = charges["email"].value_counts().head(1).index[0]

        entity_df = pd.DataFrame(
            {
                "email": [email],
                "event_timestamp": [pd.Timestamp("2025-01-01", tz=timezone.utc)],
            }
        )

        features = store.get_historical_features(
            entity_df=entity_df,
            features=["geo_time_features:postal_n_frauds_30d"],
            full_feature_names=True,
        )
        result_df = features.to_df()

        value = result_df["geo_time_features__postal_n_frauds_30d"].iloc[0]
        if value is not None:
            assert value >= 0, f"postal_n_frauds {value} should be >= 0"
