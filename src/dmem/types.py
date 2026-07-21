"""Core data types shared across every tier and surface.

These are plain dataclasses (no third-party dependency) so the core stays
importable in any ecosystem. Stores serialize/deserialize them; providers and
the engine pass them around.

Design notes tied to the brief:
- ``Provenance`` is mandatory on every ``Fact`` (source, timestamp, origin).
- ``Fact`` carries a bi-temporal validity window (``valid_from`` / ``valid_to``)
  so a contradicted fact is *closed*, never deleted. ``valid_to is None`` means
  "still believed true".
- ``FactType`` keeps extraction typed/categorized, not freeform blobs.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional


def _now() -> float:
    """Wall-clock seconds. Centralized so tests can monkeypatch if needed."""
    return time.time()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


class FactType(str, Enum):
    """Predefined categories for typed fact extraction.

    Keeping facts typed (rather than freeform) makes retrieval filterable and
    reduces noise, per the hardened-engineering requirements.
    """

    PREFERENCE = "preference"      # "I prefer dark mode", "use tabs not spaces"
    DECISION = "decision"          # "we decided to ship on Friday"
    PROJECT_FACT = "project_fact"  # "the repo is named dmem", "prod is on AWS"
    IDENTITY = "identity"          # "my name is X", "I work at Y"
    CREDENTIAL = "credential"      # secret-bearing — treated specially (see note)
    RELATIONSHIP = "relationship"  # "Alice manages Bob"
    EVENT = "event"                # "the demo is on the 30th"
    OTHER = "other"

    @classmethod
    def coerce(cls, value: "str | FactType | None") -> "FactType":
        if isinstance(value, FactType):
            return value
        if not value:
            return cls.OTHER
        try:
            return cls(str(value).strip().lower())
        except ValueError:
            return cls.OTHER


class Cardinality(str, Enum):
    """Whether a (subject, predicate) can hold one value or many at once.

    This drives contradiction handling. A SINGLE-valued predicate (e.g.
    ``works_at``) supersedes on change — the old value's validity window is
    closed. A MULTI-valued predicate (e.g. ``uses``) accumulates — "I use
    Postgres" and "I use Redis" are both true, not a contradiction.
    """

    SINGLE = "single"
    MULTI = "multi"

    @classmethod
    def coerce(cls, value: "str | Cardinality | None") -> "Optional[Cardinality]":
        if value is None:
            return None
        if isinstance(value, Cardinality):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError:
            return None


# Default cardinality per predicate. Unknown predicates default to SINGLE
# (preserving supersede-on-change), so genuinely additive predicates must be
# listed here or overridden via MULTI_VALUED_PREDICATES. Predicates emitted by
# the heuristic extractor are all covered.
DEFAULT_PREDICATE_CARDINALITY: dict[str, Cardinality] = {
    # single-valued: latest value wins, prior value is closed
    "has_name": Cardinality.SINGLE,
    "is": Cardinality.SINGLE,
    "works_at": Cardinality.SINGLE,
    "is_named": Cardinality.SINGLE,
    "scheduled_for": Cardinality.SINGLE,
    "lives_in": Cardinality.SINGLE,
    "email_is": Cardinality.SINGLE,
    "reports_to": Cardinality.SINGLE,
    "born_on": Cardinality.SINGLE,
    # multi-valued: values accumulate; all remain current
    "uses": Cardinality.MULTI,
    "prefers": Cardinality.MULTI,
    "likes": Cardinality.MULTI,
    "dislikes": Cardinality.MULTI,
    "decided": Cardinality.MULTI,
    "knows": Cardinality.MULTI,
    "owns": Cardinality.MULTI,
    "speaks": Cardinality.MULTI,
    "has_skill": Cardinality.MULTI,
    "manages": Cardinality.MULTI,
}


def predicate_cardinality(
    predicate: str,
    overrides: "Optional[dict[str, Cardinality]]" = None,
    default: Cardinality = Cardinality.SINGLE,
) -> Cardinality:
    """Resolve a predicate's cardinality: overrides > defaults > fallback."""
    p = predicate.strip().lower()
    if overrides and p in overrides:
        return overrides[p]
    return DEFAULT_PREDICATE_CARDINALITY.get(p, default)


# Default priority ordering for token-budgeted handoff. Higher = kept longer.
# When the handoff is over budget, the lowest-priority facts drop first.
FACT_TYPE_PRIORITY: dict[FactType, int] = {
    FactType.IDENTITY: 100,
    FactType.DECISION: 90,
    FactType.PROJECT_FACT: 80,
    FactType.PREFERENCE: 70,
    FactType.RELATIONSHIP: 60,
    FactType.EVENT: 50,
    FactType.CREDENTIAL: 10,   # low: usually should not travel in a handoff blob
    FactType.OTHER: 30,
}


@dataclass
class Provenance:
    """Where a stored item came from. Required on every fact."""

    source: str                       # "conversation" | "document" | "import" | ...
    timestamp: float                  # when it was recorded (unix seconds)
    origin_id: Optional[str] = None   # message id, document id, episode id, ...
    origin_ref: Optional[str] = None  # human-readable pointer (file path, url, span)
    actor: Optional[str] = None       # who asserted it (user id, agent name)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Provenance":
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__})  # type: ignore[attr-defined]


@dataclass
class Fact:
    """A single typed, provenance-tagged, bi-temporally-versioned fact."""

    subject: str
    predicate: str
    object: str
    fact_type: FactType = FactType.OTHER
    provenance: Optional[Provenance] = None

    id: str = field(default_factory=lambda: _new_id("fact"))
    namespace: str = "default"        # tenant/user isolation key
    confidence: float = 1.0
    embedding: Optional[list[float]] = None

    # Bi-temporal window. valid_to=None => currently believed true.
    valid_from: float = field(default_factory=_now)
    valid_to: Optional[float] = None
    recorded_at: float = field(default_factory=_now)

    # If this fact superseded another (contradiction handling), link it.
    supersedes: Optional[str] = None

    @property
    def is_current(self) -> bool:
        return self.valid_to is None

    @property
    def statement(self) -> str:
        return f"{self.subject} {self.predicate} {self.object}".strip()

    def dedup_key(self) -> str:
        """Identity for dedup: same subject+predicate+object in same namespace."""
        return "|".join(
            [self.namespace, self.subject.strip().lower(),
             self.predicate.strip().lower(), self.object.strip().lower()]
        )

    def priority(self) -> int:
        base = FACT_TYPE_PRIORITY.get(self.fact_type, 30)
        # Nudge by confidence so ties break sensibly.
        return int(base + round(self.confidence * 5))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["fact_type"] = self.fact_type.value
        if self.provenance is not None:
            d["provenance"] = self.provenance.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Fact":
        prov = d.get("provenance")
        return cls(
            subject=d["subject"],
            predicate=d["predicate"],
            object=d["object"],
            fact_type=FactType.coerce(d.get("fact_type")),
            provenance=Provenance.from_dict(prov) if prov else None,
            id=d.get("id") or _new_id("fact"),
            namespace=d.get("namespace", "default"),
            confidence=float(d.get("confidence", 1.0)),
            embedding=d.get("embedding"),
            valid_from=float(d.get("valid_from", _now())),
            valid_to=d.get("valid_to"),
            recorded_at=float(d.get("recorded_at", _now())),
            supersedes=d.get("supersedes"),
        )


@dataclass
class Chunk:
    """A retrievable slice of a document, respecting concept boundaries."""

    text: str
    document_id: str
    concept: Optional[str] = None      # OKF concept-file this chunk belongs to
    section: Optional[str] = None      # heading path, e.g. "Intro > Goals"
    ordinal: int = 0                   # position within the document
    id: str = field(default_factory=lambda: _new_id("chunk"))
    namespace: str = "default"
    embedding: Optional[list[float]] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    provenance: Optional[Provenance] = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.provenance is not None:
            d["provenance"] = self.provenance.to_dict()
        return d


@dataclass
class Episode:
    """A raw conversational turn (or batch) fed into ingestion."""

    text: str
    namespace: str = "default"
    role: str = "user"                 # user | assistant | system | tool
    id: str = field(default_factory=lambda: _new_id("ep"))
    timestamp: float = field(default_factory=_now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """A scored item returned from retrieval — either a fact or a chunk."""

    kind: str                          # "fact" | "chunk"
    text: str
    score: float
    source_id: str
    payload: dict[str, Any] = field(default_factory=dict)
    # Populated when this item conflicts with / supersedes another.
    conflict: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Handoff:
    """The token-budgeted, priority-ordered context handoff for a model switch."""

    text: str
    token_estimate: int
    included: list[RetrievalResult] = field(default_factory=list)
    dropped: list[RetrievalResult] = field(default_factory=list)
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    low_confidence: bool = False       # triggers the hard-fallback path
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "token_estimate": self.token_estimate,
            "included": [r.to_dict() for r in self.included],
            "dropped": [r.to_dict() for r in self.dropped],
            "conflicts": self.conflicts,
            "low_confidence": self.low_confidence,
            "notes": self.notes,
        }
