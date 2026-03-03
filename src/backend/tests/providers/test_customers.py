from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.fraud.context.providers.customers import CustomerRecord, get_by_id


@pytest.fixture(autouse=True)
def clear_cache():
    get_by_id.cache_clear()
    yield
    get_by_id.cache_clear()


class TestGetById:
    def test_returns_customer_record_from_feast(self):
        mock_response = MagicMock()
        mock_response.to_dict.return_value = {
            "customer_features__email": ["test@example.com"],
            "customer_features__fiscal_code": ["ABCDEF12G"],
            "customer_features__gender": ["M"],
            "customer_features__birth_date": ["1990-01-01"],
            "customer_features__birth_province": ["MI"],
            "customer_features__birth_country": ["IT"],
            "customer_features__high_end_count": [1.0],
        }

        mock_store = MagicMock()
        mock_store.get_online_features.return_value = mock_response

        with patch(
            "services.fraud.context.providers.customers._get_customer_store",
            return_value=mock_store,
        ):
            result = get_by_id("cust_123")

        assert isinstance(result, CustomerRecord)
        assert result.id == "cust_123"
        assert result.email == "test@example.com"
        assert result.fiscal_code == "ABCDEF12G"
        assert result.has_high_end_device is True

    def test_falls_back_to_csv_when_feast_fails(self):
        csv_data = (
            "id,email,fiscal_code,gender,birth_date,birth_province,birth_country,high_end_count\n"
            "cust_456,csv@example.com,XYZ789,F,1985-05-15,RM,IT,0\n"
        )

        with patch(
            "services.fraud.context.providers.customers._get_customer_store", return_value=None
        ):
            with patch("pathlib.Path.exists", return_value=True):
                with patch("builtins.open", mock_open(read_data=csv_data)):
                    result = get_by_id("cust_456")

        assert result.email == "csv@example.com"

    def test_raises_when_not_found_in_feast_or_csv(self):
        mock_response = MagicMock()
        mock_response.to_dict.return_value = {
            "customer_features__email": [None],
        }

        mock_store = MagicMock()
        mock_store.get_online_features.return_value = mock_response

        with patch(
            "services.fraud.context.providers.customers._get_customer_store",
            return_value=mock_store,
        ):
            with patch("pathlib.Path.exists", return_value=True):
                with patch("builtins.open", mock_open(read_data="id,email\nother,test@x.com\n")):
                    with pytest.raises(ValueError, match="not found"):
                        get_by_id("missing_id")
