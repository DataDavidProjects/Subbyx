from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


class Entity(str, Enum):
    CUSTOMER_ID = "customer_id"
    EMAIL = "email"


class Source(str, Enum):
    FEAST = "feast"
    COMPUTED = "computed"


@dataclass
class Feature:
    name: str
    entity: Entity
    source: Source
    feature_view: str | None = None
    column: str | None = None
    compute_fn: Callable[[dict], float] | None = None
    dependencies: list[str] = field(default_factory=list)
