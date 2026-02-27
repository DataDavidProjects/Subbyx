from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

# Add the backend directory to Python path so we can import modules
_BACKEND_DIR = Path(__file__).parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))


@pytest.fixture(scope="session")
def temp_data_dir():
    """Create a temporary directory for test data files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(scope="session")
def sample_customers_csv(temp_data_dir):
    """Create a sample customers.csv for testing segment determination."""
    customers_data = pd.DataFrame(
        {
            "id": ["cust_001", "cust_002", "cust_003", "cust_004"],
            "email": [
                "newuser@example.com",
                "returning@example.com",
                "returning@example.com",  # Same email as cust_002
                "setup_only@example.com",
            ],
            "gender": ["M", "F", "M", "F"],
            "dunning_days": [0, 5, 0, 20],
            "created": ["2024-01-01", "2023-06-01", "2023-08-01", "2024-02-01"],
        }
    )
    filepath = temp_data_dir / "customers.csv"
    customers_data.to_csv(filepath, index=False)
    return filepath


@pytest.fixture(scope="session")
def sample_checkouts_csv(temp_data_dir, sample_customers_csv):
    """Create a sample checkouts.csv for testing segment determination."""
    customers_df = pd.read_csv(sample_customers_csv)

    checkouts_data = pd.DataFrame(
        {
            "id": ["chk_001", "chk_002", "chk_003", "chk_004", "chk_005"],
            "customer": [
                customers_df.loc[customers_df["email"] == "returning@example.com", "id"].iloc[0],
                customers_df.loc[customers_df["email"] == "returning@example.com", "id"].iloc[1],
                customers_df.loc[customers_df["email"] == "setup_only@example.com", "id"].iloc[0],
                "cust_001",  # newuser's checkout
                "cust_999",  # unknown customer
            ],
            "email": [
                "returning@example.com",
                "returning@example.com",
                "setup_only@example.com",
                "newuser@example.com",
                "unknown@example.com",
            ],
            "mode": ["payment", "payment", "setup", "payment", "payment"],
            "subscription_value": [50.0, 75.0, None, 100.0, 60.0],
            "created": [
                "2023-07-01",
                "2023-09-01",
                "2024-02-15",
                "2024-01-15",
                "2024-03-01",
            ],
        }
    )
    filepath = temp_data_dir / "checkouts.csv"
    checkouts_data.to_csv(filepath, index=False)
    return filepath


@pytest.fixture(scope="session")
def historical_data_dir(temp_data_dir, sample_customers_csv, sample_checkouts_csv):
    """Create the historical data directory structure."""
    historical_dir = temp_data_dir / "historical"
    historical_dir.mkdir()

    # Copy files to historical directory
    pd.read_csv(sample_customers_csv).to_csv(historical_dir / "customers.csv", index=False)
    pd.read_csv(sample_checkouts_csv).to_csv(historical_dir / "checkouts.csv", index=False)

    return historical_dir


@pytest.fixture(scope="session")
def sample_blacklist(temp_data_dir):
    """Create a sample blacklist.json file."""
    blacklist_data = {
        "emails": [
            "blocked@example.com",
            "fraudster@evil.com",
            "test+blocked@example.com",
        ]
    }
    filepath = temp_data_dir / "blacklist.json"
    with open(filepath, "w") as f:
        json.dump(blacklist_data, f)
    return filepath


@pytest.fixture
def mock_env_vars(historical_data_dir, sample_blacklist, monkeypatch):
    """Patch config to use test data files."""
    monkeypatch.setenv("DATA_PATH", str(historical_data_dir.parent))

    # Patch the shared config for paths
    import routes.config as config_module

    config_module.shared_config["paths"]["historical_data"] = str(historical_data_dir)
    config_module.shared_config["paths"]["blacklist"] = str(sample_blacklist)

    yield {
        "historical_data_dir": historical_data_dir,
        "blacklist_path": sample_blacklist,
    }


@pytest.fixture
def test_client(mock_env_vars):
    """Create a FastAPI test client."""
    from main import app

    return TestClient(app)


@pytest.fixture
def sample_checkout_request():
    """Sample checkout request payload."""
    return {
        "customer_id": "cust_001",
        "email": "newuser@example.com",
        "checkout_data": {
            "subscription_value": 50.0,
            "grade": "new",
            "category": "smartphone",
            "condition": "new",
            "has_linked_products": False,
            "has_vetrino": True,
            "has_protezione_totale": False,
            "has_protezione_furto": False,
            "store_id": "store_001",
            "sku": "IPHONE15",
        },
    }


@pytest.fixture
def returning_customer_email():
    """Email for a returning customer."""
    return "returning@example.com"


@pytest.fixture
def new_customer_email():
    """Email for a new customer."""
    return "brandnew@example.com"


@pytest.fixture
def blocked_email():
    """Email that is in the blacklist."""
    return "blocked@example.com"
