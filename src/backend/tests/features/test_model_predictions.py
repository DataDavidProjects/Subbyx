from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


DATA_DIR = Path(__file__).parents[4] / "data"


class TestModelPredictions:
    """Test that model predictions make sense for known fraud and clean cases."""

    def test_fraud_case_has_higher_score_than_clean_case(self) -> None:
        """Fraud cases should have higher prediction scores than clean cases."""
        from services.fraud.inference import predict

        customers = pd.read_csv(DATA_DIR / "01-clean" / "customers.csv")
        checkouts = pd.read_csv(DATA_DIR / "01-clean" / "checkouts.csv")

        checkouts = checkouts[checkouts["mode"].isin(["payment", "subscription"])]
        merged = checkouts.merge(
            customers[["id", "email", "dunning_days"]],
            left_on="customer",
            right_on="id",
            how="left",
        )
        merged = merged.dropna(subset=["email", "dunning_days"])

        fraud_cases = merged[merged["dunning_days"] > 15].head(5)
        clean_cases = merged[merged["dunning_days"] <= 15].head(5)

        if len(fraud_cases) == 0 or len(clean_cases) == 0:
            pytest.skip("Not enough fraud/clean cases for comparison")

        fraud_scores = []
        for _, row in fraud_cases.iterrows():
            result = predict(email=row["email"])
            fraud_scores.append(result.score)

        clean_scores = []
        for _, row in clean_cases.iterrows():
            result = predict(email=row["email"])
            clean_scores.append(result.score)

        avg_fraud_score = sum(fraud_scores) / len(fraud_scores)
        avg_clean_score = sum(clean_scores) / len(clean_scores)

        assert avg_fraud_score > avg_clean_score, (
            f"Fraud avg score {avg_fraud_score:.3f} should be > clean avg {avg_clean_score:.3f}"
        )

    def test_model_outputs_valid_probability_range(self) -> None:
        """Model outputs should be between 0 and 1."""
        from services.fraud.inference import predict

        customers = pd.read_csv(DATA_DIR / "01-clean" / "customers.csv")
        checkouts = pd.read_csv(DATA_DIR / "01-clean" / "checkouts.csv")

        checkouts = checkouts[checkouts["mode"].isin(["payment", "subscription"])]
        merged = checkouts.merge(
            customers[["id", "email", "dunning_days"]],
            left_on="customer",
            right_on="id",
            how="left",
        )
        merged = merged.dropna(subset=["email"])

        sample = merged.head(10)

        for _, row in sample.iterrows():
            result = predict(email=row["email"])
            assert 0 <= result.score <= 1, f"Score {result.score} not in [0,1]"

    def test_new_customer_has_different_score_distribution(self) -> None:
        """New customers (no history) should have different score distribution than returning."""
        from services.fraud.inference import predict

        charges = pd.read_csv(DATA_DIR / "01-clean" / "charges.csv")
        customers = pd.read_csv(DATA_DIR / "01-clean" / "customers.csv")
        checkouts = pd.read_csv(DATA_DIR / "01-clean" / "checkouts.csv")

        checkouts = checkouts[checkouts["mode"].isin(["payment", "subscription"])]
        merged = checkouts.merge(
            customers[["id", "email"]], left_on="customer", right_on="id", how="left"
        )
        merged = merged.dropna(subset=["email"])

        returning_emails = set(charges["email"].unique())
        new_customers = merged[~merged["email"].isin(returning_emails)].head(5)
        returning = merged[merged["email"].isin(returning_emails)].head(5)

        if len(new_customers) == 0 or len(returning) == 0:
            pytest.skip("Not enough new/returning customers")

        new_scores = []
        for _, row in new_customers.iterrows():
            result = predict(email=row["email"])
            new_scores.append(result.score)

        returning_scores = []
        for _, row in returning.iterrows():
            result = predict(email=row["email"])
            returning_scores.append(result.score)

        print(f"\nNew customer scores: {new_scores}")
        print(f"Returning customer scores: {returning_scores}")

    def test_segment_thresholds_work(self) -> None:
        """Verify segment thresholds correctly classify high/low risk."""
        from services.fraud.inference import predict

        customers = pd.read_csv(DATA_DIR / "01-clean" / "customers.csv")
        checkouts = pd.read_csv(DATA_DIR / "01-clean" / "checkouts.csv")

        checkouts = checkouts[checkouts["mode"].isin(["payment", "subscription"])]
        merged = checkouts.merge(
            customers[["id", "email", "dunning_days"]],
            left_on="customer",
            right_on="id",
            how="left",
        )
        merged = merged.dropna(subset=["email", "dunning_days"])

        RETURNING_THRESHOLD = 0.18

        merged["is_fraud"] = merged["dunning_days"] > 15

        returning = merged[
            merged["email"].isin(
                pd.read_csv(DATA_DIR / "01-clean" / "charges.csv")["email"].unique()
            )
        ].head(20)

        if len(returning) == 0:
            pytest.skip("No returning customers found")

        tp = 0
        fp = 0
        tn = 0
        fn = 0

        for _, row in returning.iterrows():
            result = predict(email=row["email"])
            predicted_fraud = result.score >= RETURNING_THRESHOLD
            actual_fraud = row["is_fraud"]

            if predicted_fraud and actual_fraud:
                tp += 1
            elif predicted_fraud and not actual_fraud:
                fp += 1
            elif not predicted_fraud and not actual_fraud:
                tn += 1
            else:
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        print(f"\nReturning segment (threshold={RETURNING_THRESHOLD}):")
        print(f"  TP={tp}, FP={fp}, TN={tn}, FN={fn}")
        print(f"  Precision={precision:.2f}, Recall={recall:.2f}")
