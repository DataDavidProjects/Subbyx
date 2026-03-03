from __future__ import annotations

import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score
import yaml

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
SELECTED_FEATURES_PATH = REPO_ROOT / "src" / "backend" / "feature_repo" / "selected_features.yaml"

def load_test_data():
    with open(SELECTED_FEATURES_PATH) as f:
        cfg = yaml.safe_load(f)
    training_dir = Path(cfg["training_dir"])
    test_path = training_dir / "test.parquet"
    logger.info(f"Loading test data from {test_path}")
    return pd.read_parquet(test_path)

def evaluate_rule(y_true, y_pred_triggered, rule_name):
    # Rules are binary: triggered (1) or not (0)
    # For ROC AUC and PR AUC, we use the binary trigger as the "score"
    try:
        roc_auc = roc_auc_score(y_true, y_pred_triggered)
    except Exception:
        roc_auc = 0.5 # Default if only one class present or other error
        
    try:
        pr_auc = average_precision_score(y_true, y_pred_triggered)
    except Exception:
        pr_auc = 0.0
        
    precision = precision_score(y_true, y_pred_triggered, zero_division=0)
    recall = recall_score(y_true, y_pred_triggered, zero_division=0)
    n_triggered = sum(y_pred_triggered)
    n_total = len(y_true)
    
    return {
        "Rule": rule_name,
        "Triggered": f"{n_triggered} ({n_triggered/n_total:.1%})",
        "Precision": f"{precision:.3f}",
        "Recall": f"{recall:.3f}",
        "ROC AUC": f"{roc_auc:.3f}",
        "PR AUC": f"{pr_auc:.3f}"
    }

# Rule Implementations (Simulated for batch evaluation)

def get_blacklist_predictions(df):
    blacklist_path = DATA_DIR / "blacklist.json"
    if blacklist_path.exists():
        with open(blacklist_path) as f:
            emails = set(json.load(f).get("emails", []))
    else:
        emails = set()
    return df["email"].apply(lambda x: 1 if x in emails else 0)

def get_stripe_risk_predictions(df):
    # Stripe risk level 'highest' is a strong signal.
    # In the test set, we have 'charge_features__outcome_risk_score'.
    # Stripe 'highest' risk typically corresponds to score >= 85 (or similar).
    # However, our rule 'load_charges_with_highest_risk' checks historical data.
    # A better proxy in the test set is if the PIT-correct score was already high.
    # But let's look for the actual rule logic: 'outcome_risk_level' == 'highest'.
    # We don't have 'outcome_risk_level' in test.parquet, only 'charge_features__outcome_risk_score'.
    # Let's assume score > 90 is 'highest'.
    return df["charge_features__outcome_risk_score"].apply(lambda x: 1 if (not pd.isna(x) and x >= 90) else 0)

def get_fiscal_code_duplicate_predictions(df):
    # This rule checks if the SAME fiscal code is used with DIFFERENT emails.
    # In a batch setting, we can't easily simulate PIT without the full history.
    # However, we can use the test set as a proxy or just mark it as 0 if we can't compute it accurately.
    # Let's try a simplified version: if we see multiple emails for one fiscal code in the test set.
    # BUT PIT IS CRITICAL. 
    # Actually, let's use the 'customer_profile_features__n_emails_per_fiscal_code' feature which is already in the parquet!
    return df["customer_profile_features__n_emails_per_fiscal_code"].apply(lambda x: 1 if (not pd.isna(x) and x > 1) else 0)

def get_payment_failure_predictions(df):
    # FAILURE_RATE_THRESHOLD = 0.80
    # MIN_ATTEMPTS = 15
    def check_pf(row):
        # Check both payment intent and charge stats
        pi_rate = row.get("payment_intent_stats_features__failure_rate")
        pi_count = row.get("payment_intent_stats_features__n_payment_intents")
        
        ch_rate = row.get("charge_stats_features__failure_rate")
        ch_count = row.get("charge_stats_features__n_charges")
        
        if not pd.isna(pi_rate) and not pd.isna(pi_count):
            if pi_count >= 15 and pi_rate >= 0.80:
                return 1
        
        if not pd.isna(ch_rate) and not pd.isna(ch_count):
            if ch_count >= 15 and ch_rate >= 0.80:
                return 1
        return 0
        
    return df.apply(check_pf, axis=1)

def main():
    df = load_test_data()
    y_true = df["label"].values
    
    results = []
    
    # 1. Blacklist
    y_pred_bl = get_blacklist_predictions(df)
    results.append(evaluate_rule(y_true, y_pred_bl, "Blacklist"))
    
    # 2. Stripe Risk (Proxy)
    y_pred_sr = get_stripe_risk_predictions(df)
    results.append(evaluate_rule(y_true, y_pred_sr, "Stripe Risk (Score >= 90)"))
    
    # 3. Fiscal Code Duplicate
    y_pred_fc = get_fiscal_code_duplicate_predictions(df)
    results.append(evaluate_rule(y_true, y_pred_fc, "Fiscal Code Duplicate"))
    
    # 4. Payment Failure
    y_pred_pf = get_payment_failure_predictions(df)
    results.append(evaluate_rule(y_true, y_pred_pf, "Payment Failure (Updated)"))
    
    # 5. Combined Rules (Total Rules Engine performance)
    y_combined = (y_pred_bl | y_pred_sr | y_pred_fc | y_pred_pf).astype(int)
    results.append(evaluate_rule(y_true, y_combined, "Rules Engine (ALL)"))
    
    # Format and print results
    report_df = pd.DataFrame(results)
    print("\nRule Performance Evaluation (on Test Set)")
    print("=" * 80)
    print(report_df.to_string(index=False))
    print("=" * 80)
    print("Note: Stripe Risk is evaluated using a score threshold proxy (>= 90).")
    print("Note: Fiscal Code Duplicate uses the n_emails_per_fiscal_code feature.")

if __name__ == "__main__":
    main()
