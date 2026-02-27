from __future__ import annotations

import pytest

from services.fraud.features.base import Entity, Feature, Source
from services.fraud.features.registry import (
    FeatureRegistry,
    registry,
    register_static,
    register_computed,
)


@pytest.fixture(autouse=True)
def clear_registry():
    """Clear registry before each test."""
    registry.clear()
    yield
    registry.clear()


class TestFeatureRegistry:
    """Tests for FeatureRegistry singleton."""

    def test_singleton(self):
        """Test that registry is a singleton."""
        r1 = FeatureRegistry()
        r2 = FeatureRegistry()
        assert r1 is r2

    def test_register_feature(self):
        """Test registering a single feature."""
        feat = Feature(
            name="test_feature",
            entity=Entity.CUSTOMER_ID,
            source=Source.FEAST,
            feature_view="test_view",
        )
        registry.register(feat)
        assert registry.get("test_feature") is not None

    def test_register_with_group(self):
        """Test registering a feature with group."""
        feat = Feature(
            name="test_feature",
            entity=Entity.CUSTOMER_ID,
            source=Source.FEAST,
            feature_view="test_view",
        )
        registry.register(feat, group="test_group")
        assert registry.get("test_feature") is not None
        assert "test_group" in registry.get_groups()

    def test_get_by_entity(self):
        """Test filtering by entity."""
        f1 = Feature(name="f1", entity=Entity.CUSTOMER_ID, source=Source.FEAST)
        f2 = Feature(name="f2", entity=Entity.EMAIL, source=Source.FEAST)
        registry.register(f1)
        registry.register(f2)

        customer_features = registry.get_by_entity(Entity.CUSTOMER_ID)
        assert len(customer_features) == 1
        assert customer_features[0].name == "f1"

    def test_get_by_group(self):
        """Test filtering by group."""
        f1 = Feature(name="f1", entity=Entity.CUSTOMER_ID, source=Source.FEAST)
        f2 = Feature(name="f2", entity=Entity.CUSTOMER_ID, source=Source.FEAST)
        registry.register(f1, group="customer")
        registry.register(f2, group="risk")

        customer_features = registry.get_by_group("customer")
        assert len(customer_features) == 1

    def test_get_by_source(self):
        """Test filtering by source."""
        f1 = Feature(name="f1", entity=Entity.CUSTOMER_ID, source=Source.FEAST)
        f2 = Feature(name="f2", entity=Entity.CUSTOMER_ID, source=Source.COMPUTED)
        registry.register(f1)
        registry.register(f2)

        feast_features = registry.get_by_source(Source.FEAST)
        assert len(feast_features) == 1


class TestRegisterStatic:
    """Tests for register_static helper."""

    def test_register_static(self):
        """Test registering static feature."""
        register_static(
            name="test_feature",
            entity=Entity.CUSTOMER_ID,
            feature_view="customer_features",
            group="customer",
        )
        feat = registry.get("test_feature")
        assert feat is not None
        assert feat.source == Source.FEAST
        assert feat.entity == Entity.CUSTOMER_ID


class TestRegisterComputed:
    """Tests for register_computed helper."""

    def test_register_computed(self):
        """Test registering computed feature."""

        def compute_fn(features: dict) -> float:
            return 1.0

        register_computed(
            name="computed_feature",
            entity=Entity.EMAIL,
            compute_fn=compute_fn,
            dependencies=["dep1", "dep2"],
            group="computed",
        )
        feat = registry.get("computed_feature")
        assert feat is not None
        assert feat.source == Source.COMPUTED
        assert feat.compute_fn is not None

    def test_computed_evaluation(self):
        """Test computed feature evaluation."""

        def compute_fn(features: dict) -> float:
            return features.get("a", 0) + features.get("b", 0)

        register_computed(
            name="sum_feature",
            entity=Entity.EMAIL,
            compute_fn=compute_fn,
            dependencies=["a", "b"],
        )

        feat = registry.get("sum_feature")
        assert feat is not None
        result = feat.compute_fn({"a": 1, "b": 2})
        assert result == 3


class TestComputedFeatures:
    """Tests for computed features in the registry."""

    def test_computed_features_evaluated(self):
        """Test that computed features work with get_features logic."""
        # Register a base feature
        register_static(
            name="base_feature",
            entity=Entity.EMAIL,
            feature_view="test_view",
        )

        # Register a computed feature that depends on base_feature
        def compute_fn(features: dict) -> float:
            base = features.get("base_feature", 0)
            return base * 2

        register_computed(
            name="computed",
            entity=Entity.EMAIL,
            compute_fn=compute_fn,
            dependencies=["base_feature"],
        )

        # Simulate what get_features does
        features = {"base_feature": 5.0}
        computed = registry.get_by_source(Source.COMPUTED)
        for feat in computed:
            if feat.compute_fn:
                features[feat.name] = feat.compute_fn(features)

        assert features["computed"] == 10.0
