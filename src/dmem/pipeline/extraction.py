"""Typed fact extraction.

Facts are extracted into predefined categories (see `FactType`), never freeform
blobs, so retrieval stays filterable and low-noise. Every extracted fact gets
provenance attached by the caller (the engine).

Two extractors:
- `HeuristicFactExtractor` — deterministic, dependency-free, offline. Pattern-
  based (first-person preferences, decisions, identity, simple relations). Lower
  recall than an LLM, but zero-setup and predictable. This is the default.
- `LLMFactExtractor` — uses a BYO LLM to emit typed subject/predicate/object
  triples as JSON. Higher quality; used when `LLM_HOST_URL` is configured.

Both return a list of (subject, predicate, object, fact_type, confidence).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

from ..providers.llm import LLMProvider
from ..types import FactType


@dataclass
class ExtractedFact:
    subject: str
    predicate: str
    object: str
    fact_type: FactType
    confidence: float = 0.7


@runtime_checkable
class FactExtractor(Protocol):
    def extract(self, text: str, *, speaker: str = "user") -> list[ExtractedFact]:
        ...


# --------------------------------------------------------------------------- #
# Heuristic extractor
# --------------------------------------------------------------------------- #

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")

# (regex, predicate, fact_type, subject_is_speaker)
_PATTERNS: list[tuple[re.Pattern[str], str, FactType]] = [
    (re.compile(r"\bmy name is\s+([A-Z][\w'-]+(?:\s+[A-Z][\w'-]+)?)", re.I),
     "has_name", FactType.IDENTITY),
    (re.compile(r"\bi am\s+([A-Z][\w'-]+(?:\s+[A-Z][\w'-]+)?)(?:\b|$)"),
     "is", FactType.IDENTITY),
    (re.compile(r"\bi work (?:at|for)\s+([\w .&'-]{2,40})", re.I),
     "works_at", FactType.IDENTITY),
    (re.compile(r"\bi (?:prefer|like|love|favor)\s+(.{2,60})", re.I),
     "prefers", FactType.PREFERENCE),
    (re.compile(r"\bi (?:dislike|hate|avoid|don't like|do not like)\s+(.{2,60})", re.I),
     "dislikes", FactType.PREFERENCE),
    (re.compile(r"\bi (?:use|am using)\s+(.{2,60})", re.I),
     "uses", FactType.PREFERENCE),
    (re.compile(r"\b(?:we|i|they) (?:decided|chose|will use|are going with|"
                r"picked|settled on)\s+(.{2,80})", re.I),
     "decided", FactType.DECISION),
    (re.compile(r"\b(?:the )?(?:project|repo|repository|product|app) "
                r"(?:is (?:called|named)|name is)\s+([\w.-]{2,40})", re.I),
     "is_named", FactType.PROJECT_FACT),
    (re.compile(r"\b(?:my|the) (?:deadline|due date|demo) is\s+(.{2,60})", re.I),
     "scheduled_for", FactType.EVENT),
]

_CREDENTIAL_RE = re.compile(
    r"\b(api[_ ]?key|token|password|secret|passwd)\b", re.I)


class HeuristicFactExtractor:
    OFFLINE = True

    def __init__(self, speaker_entity: str = "user"):
        self.speaker_entity = speaker_entity

    def extract(self, text: str, *, speaker: str = "user") -> list[ExtractedFact]:
        subject = speaker or self.speaker_entity
        found: list[ExtractedFact] = []
        seen: set[tuple[str, str, str]] = set()
        for sentence in _SENT_SPLIT_RE.split(text):
            s = sentence.strip()
            if not s:
                continue
            for pat, predicate, ftype in _PATTERNS:
                m = pat.search(s)
                if not m:
                    continue
                obj = _clean_object(m.group(1))
                if not obj:
                    continue
                # Credential-bearing statements are tagged as such (low priority,
                # handled with care downstream) rather than dropped silently.
                ft = FactType.CREDENTIAL if _CREDENTIAL_RE.search(s) else ftype
                key = (subject.lower(), predicate, obj.lower())
                if key in seen:
                    continue
                seen.add(key)
                found.append(ExtractedFact(
                    subject=subject, predicate=predicate, object=obj,
                    fact_type=ft, confidence=0.6))
        return found


def _clean_object(raw: str) -> str:
    obj = raw.strip().rstrip(".!?,;:").strip()
    # trim trailing conjunction clauses that pull in unrelated text
    obj = re.split(r"\b(?:because|since|so that|but|and then)\b", obj)[0].strip()
    return obj[:120]


# --------------------------------------------------------------------------- #
# LLM extractor
# --------------------------------------------------------------------------- #

_LLM_SYSTEM = (
    "You extract durable facts from a message as typed triples. "
    "Return ONLY a JSON array. Each item: "
    '{"subject","predicate","object","fact_type","confidence"}. '
    "fact_type is one of: preference, decision, project_fact, identity, "
    "credential, relationship, event, other. "
    "Extract only stable facts worth remembering, not chit-chat. "
    "Use the literal speaker label as subject for first-person statements. "
    "If nothing is worth storing, return []."
)


class LLMFactExtractor:
    OFFLINE = False

    def __init__(self, llm: LLMProvider):
        self._llm = llm

    def extract(self, text: str, *, speaker: str = "user") -> list[ExtractedFact]:
        prompt = f"Speaker label: {speaker}\nMessage:\n{text}"
        raw = self._llm.complete(prompt, system=_LLM_SYSTEM, max_tokens=600)
        data = _parse_json_array(raw)
        out: list[ExtractedFact] = []
        for item in data:
            try:
                out.append(ExtractedFact(
                    subject=str(item["subject"]).strip(),
                    predicate=str(item["predicate"]).strip(),
                    object=str(item["object"]).strip(),
                    fact_type=FactType.coerce(item.get("fact_type")),
                    confidence=float(item.get("confidence", 0.8)),
                ))
            except (KeyError, TypeError, ValueError):
                continue
        return out


def _parse_json_array(raw: str) -> list[dict]:
    raw = raw.strip()
    # tolerate models that wrap JSON in prose or code fences
    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1 or end < start:
        return []
    try:
        parsed = json.loads(raw[start:end + 1])
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        return []
