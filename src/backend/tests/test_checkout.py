"""
Tests for the fraud checkout endpoint.

These tests validate the skeleton implementation of the fraud detection system:
1. Blacklist rule - blocks emails in the blacklist
2. Segment determination - determines NEW_CUSTOMER vs RETURNING
3. Score generation - returns a random score (placeholder)
4. Decision engine - applies threshold based on segment

Run with: cd src/backend && uv run pytest tests/ -v
"""

from __future__ import annotations

from fastapi.testclient import TestClient


class TestBlacklistRule:
    """Tests for the blacklist rule functionality."""

    def test_blacklist_returns_blocked_decision(
        self, test_client: TestClient, sample_checkout_request: dict, blocked_email: str
    ):
        """When email is in blacklist, should return BLOCK with blacklist rule triggered."""
        sample_checkout_request["email"] = blocked_email

        response = test_client.post("/fraud/v1/checkout", json=sample_checkout_request)

        assert response.status_code == 200
        data = response.json()
        assert data["decision"] == "BLOCK"
        assert data["rule_triggered"] == "blacklist"
        assert "blacklist" in data["reason"].lower()
        assert data["score"] is None
        # Segment is determined before the blacklist check
        assert data["segment"] is not None

    def test_blacklist_check_function(self, mock_env_vars):
        """Test the load_blacklist function directly."""
        from routes.fraud.checkout import load_blacklist

        blacklist = load_blacklist()

        assert isinstance(blacklist, set)
        assert "blocked@example.com" in blacklist
        assert "fraudster@evil.com" in blacklist
        assert "legit@example.com" not in blacklist


class TestSegmentDetermination:
    """Tests for segment determination logic."""

    def test_new_customer_segment(
        self, test_client: TestClient, sample_checkout_request: dict, new_customer_email: str
    ):
        """When email is not found, should return NEW_CUSTOMER segment."""
        sample_checkout_request["email"] = new_customer_email

        response = test_client.post("/fraud/v1/checkout", json=sample_checkout_request)

        assert response.status_code == 200
        data = response.json()
        assert data["segment"] == "NEW_CUSTOMER"
        assert data["segment_reason"] is not None

    def test_returning_customer_segment(
        self, test_client: TestClient, sample_checkout_request: dict, returning_customer_email: str
    ):
        """When email has prior real checkouts, should return RETURNING segment."""
        sample_checkout_request["email"] = returning_customer_email

        response = test_client.post("/fraud/v1/checkout", json=sample_checkout_request)

        assert response.status_code == 200
        data = response.json()
        assert data["segment"] == "RETURNING"
        assert "prior subscription" in data["segment_reason"].lower()

    def test_determine_segment_with_no_historical_data(self, temp_data_dir, monkeypatch):
        """When historical data is missing, should default to NEW_CUSTOMER."""
        from routes.fraud.checkout import determine_segment

        # Create empty historical dir
        empty_dir = temp_data_dir / "empty_historical"
        empty_dir.mkdir()

        # Patch the config path - use shared_config for paths
        import routes.config as config_module

        config_module.shared_config["paths"]["historical_data"] = str(empty_dir)

        segment, reason = determine_segment("anyone@example.com")

        assert segment == "NEW_CUSTOMER"
        assert "no historical data" in reason.lower()

    def test_determine_segment_setup_only_customer(
        self, test_client: TestClient, sample_checkout_request: dict
    ):
        """Customer with only setup-mode checkouts should be treated as NEW_CUSTOMER."""
        sample_checkout_request["email"] = "setup_only@example.com"

        response = test_client.post("/fraud/v1/checkout", json=sample_checkout_request)

        assert response.status_code == 200
        data = response.json()
        # setup_only@example.com has only mode='setup' checkouts, so should be NEW_CUSTOMER
        assert data["segment"] == "NEW_CUSTOMER"


class TestScoreGeneration:
    """Tests for score generation (random placeholder)."""

    def test_score_is_float(self, test_client: TestClient, sample_checkout_request: dict):
        """Score should be a float value."""
        response = test_client.post("/fraud/v1/checkout", json=sample_checkout_request)

        assert response.status_code == 200
        data = response.json()
        assert data["score"] is not None
        assert isinstance(data["score"], float)

    def test_score_in_valid_range(self, test_client: TestClient, sample_checkout_request: dict):
        """Score should be between 0 and 1."""
        # Run multiple times to check range (since it's random)
        for _ in range(10):
            response = test_client.post("/fraud/v1/checkout", json=sample_checkout_request)
            data = response.json()
            assert 0.0 <= data["score"] <= 1.0

    def test_score_returns_none_when_blacklisted(
        self, test_client: TestClient, sample_checkout_request: dict, blocked_email: str
    ):
        """Score should be None when email is blacklisted."""
        sample_checkout_request["email"] = blocked_email

        response = test_client.post("/fraud/v1/checkout", json=sample_checkout_request)

        assert response.status_code == 200
        data = response.json()
        assert data["score"] is None


class TestDecisionEngine:
    """Tests for the decision engine logic."""

    def test_decision_approve_below_threshold(self, mock_env_vars):
        """Score below threshold should return APPROVE."""
        from routes.fraud.checkout import get_decision

        decision, reason = get_decision(score=0.3, segment="NEW_CUSTOMER")

        assert decision == "APPROVE"
        assert "within threshold" in reason.lower()

    def test_decision_block_above_threshold(self, mock_env_vars):
        """Score above threshold should return BLOCK."""
        from routes.fraud.checkout import get_decision

        decision, reason = get_decision(score=0.6, segment="NEW_CUSTOMER")

        assert decision == "BLOCK"
        assert "exceeds threshold" in reason.lower()

    def test_decision_returning_segment_higher_threshold(self, mock_env_vars):
        """RETURNING segment should have higher threshold (0.8 vs 0.5)."""
        from routes.fraud.checkout import get_decision

        # Score 0.6 should BLOCK for NEW_CUSTOMER (threshold 0.5)
        decision_new, _ = get_decision(score=0.6, segment="NEW_CUSTOMER")

        # Score 0.6 should APPROVE for RETURNING (threshold 0.8)
        decision_returning, _ = get_decision(score=0.6, segment="RETURNING")

        assert decision_new == "BLOCK"
        assert decision_returning == "APPROVE"

    def test_decision_boundary_exact_threshold(self, mock_env_vars):
        """Score exactly at threshold should approve (not strictly greater)."""
        from routes.fraud.checkout import get_decision

        # At exactly 0.5, should approve (score > threshold is False)
        decision, reason = get_decision(score=0.5, segment="NEW_CUSTOMER")

        assert decision == "APPROVE"


class TestCheckoutEndpointIntegration:
    """Integration tests for the full /fraud/v1/checkout endpoint."""

    def test_endpoint_returns_200(self, test_client: TestClient, sample_checkout_request: dict):
        """Endpoint should return 200 for valid requests."""
        response = test_client.post("/fraud/v1/checkout", json=sample_checkout_request)

        assert response.status_code == 200

    def test_response_schema_validation(
        self, test_client: TestClient, sample_checkout_request: dict
    ):
        """Response should match the CheckoutResponse schema."""
        response = test_client.post("/fraud/v1/checkout", json=sample_checkout_request)

        assert response.status_code == 200
        data = response.json()

        # Required fields
        assert "decision" in data
        assert "reason" in data
        assert "rule_triggered" in data
        assert "score" in data
        assert "segment" in data
        assert "segment_reason" in data

        # Types
        assert isinstance(data["decision"], str)
        assert isinstance(data["reason"], str)
        assert data["decision"] in ["APPROVE", "BLOCK"]

    def test_full_flow_new_customer_approve(
        self, test_client: TestClient, sample_checkout_request: dict, new_customer_email: str
    ):
        """Full flow: new customer with low score should APPROVE."""
        sample_checkout_request["email"] = new_customer_email

        # We can't control the random score, so we just verify the flow works
        response = test_client.post("/fraud/v1/checkout", json=sample_checkout_request)

        assert response.status_code == 200
        data = response.json()
        assert data["segment"] == "NEW_CUSTOMER"
        assert data["score"] is not None
        assert data["decision"] in ["APPROVE", "BLOCK"]

    def test_full_flow_returning_customer(
        self, test_client: TestClient, sample_checkout_request: dict, returning_customer_email: str
    ):
        """Full flow: returning customer should have RETURNING segment."""
        sample_checkout_request["email"] = returning_customer_email

        response = test_client.post("/fraud/v1/checkout", json=sample_checkout_request)

        assert response.status_code == 200
        data = response.json()
        assert data["segment"] == "RETURNING"
        assert "prior subscription" in data["segment_reason"].lower()

    def test_full_flow_blacklist_blocks_early(
        self, test_client: TestClient, sample_checkout_request: dict, blocked_email: str
    ):
        """Blacklisted email should be blocked before scoring."""
        sample_checkout_request["email"] = blocked_email

        response = test_client.post("/fraud/v1/checkout", json=sample_checkout_request)

        assert response.status_code == 200
        data = response.json()

        # Should be blocked by blacklist
        assert data["decision"] == "BLOCK"
        assert data["rule_triggered"] == "blacklist"

        # Segment is determined before blacklist check (needed for segment-aware rules)
        assert data["segment"] is not None
        # Score should be None (model scoring is skipped)
        assert data["score"] is None

    def test_invalid_request_missing_fields(self, test_client: TestClient):
        """Missing required fields should return validation error."""
        invalid_request = {"email": "test@example.com"}

        response = test_client.post("/fraud/v1/checkout", json=invalid_request)

        # FastAPI returns 422 for validation errors
        assert response.status_code == 422

    def test_invalid_request_wrong_type(self, test_client: TestClient):
        """Wrong field types should return validation error."""
        invalid_request = {
            "customer_id": 123,  # Should be string
            "email": "test@example.com",
            "checkout_data": {},
        }

        response = test_client.post("/fraud/v1/checkout", json=invalid_request)

        assert response.status_code == 422


class TestHealthEndpoint:
    """Basic health check tests."""

    def test_health_endpoint(self, test_client: TestClient):
        """Health endpoint should return 200."""
        response = test_client.get("/health")

        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_root_endpoint(self, test_client: TestClient):
        """Root endpoint should return API info."""
        response = test_client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
