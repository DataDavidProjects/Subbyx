from __future__ import annotations

from collections.abc import Callable

from services.fraud.features.base import Entity, Feature, Source


def register_static(
    name: str,
    entity: Entity,
    feature_view: str,
    group: str | None = None,
    column: str | None = None,
) -> None:
    """Helper to register a static feature from Feast."""
    feat = Feature(
        name=name,
        entity=entity,
        source=Source.FEAST,
        feature_view=feature_view,
        column=column or name,
    )
    registry.register(feat, group)


def register_computed(
    name: str,
    entity: Entity,
    compute_fn: Callable[[dict], float],
    dependencies: list[str],
    group: str | None = None,
) -> None:
    """Helper to register a computed/derived feature."""
    feat = Feature(
        name=name,
        entity=entity,
        source=Source.COMPUTED,
        compute_fn=compute_fn,
        dependencies=dependencies,
    )
    registry.register(feat, group)


class FeatureRegistry:
    _instance: FeatureRegistry | None = None
    _features: dict[str, Feature]
    _groups: dict[str, set[str]]

    def __new__(cls) -> FeatureRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._features = {}
            cls._instance._groups = {}
        return cls._instance

    def register(self, feature: Feature, group: str | None = None) -> None:
        self._features[feature.name] = feature
        if group is not None:
            if group not in self._groups:
                self._groups[group] = set()
            self._groups[group].add(feature.name)

    def get(self, name: str) -> Feature | None:
        return self._features.get(name)

    def get_by_entity(self, entity: Entity) -> list[Feature]:
        return [f for f in self._features.values() if f.entity == entity]

    def get_by_group(self, group: str) -> list[Feature]:
        feature_names = self._groups.get(group, set())
        return [self._features[name] for name in feature_names if name in self._features]

    def get_by_source(self, source: Source) -> list[Feature]:
        return [f for f in self._features.values() if f.source == source]

    def get_all(self) -> list[Feature]:
        return list(self._features.values())

    def get_groups(self) -> list[str]:
        return list(self._groups.keys())

    def clear(self) -> None:
        self._features.clear()
        self._groups.clear()

registry = FeatureRegistry()
