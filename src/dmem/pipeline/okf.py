"""OKF-style structuring of markdown into discrete concept sections.

"OKF" (Open Knowledge Format) here is a *convention applied at ingestion time*,
not a running service: an OCR'd/markdown document is split into discrete concept
units following its heading hierarchy, each carrying metadata (heading path,
ordinal). Downstream chunking respects these boundaries so a chunk is never cut
mid-section or mid-table.

Input: markdown text (from OCR, or a document that was already markdown/text).
Output: an ordered list of `OKFConcept`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")


@dataclass
class OKFConcept:
    concept: str                    # the concept/section title
    section_path: str               # e.g. "Introduction > Goals"
    text: str                       # the section body (may include tables)
    ordinal: int
    level: int = 0
    metadata: dict = field(default_factory=dict)


def structure_markdown(markdown: str, *, default_title: str = "document") -> list[OKFConcept]:
    """Split markdown into concept sections by heading hierarchy.

    Content before the first heading becomes a leading "preamble" concept so no
    text is lost. Heading path is tracked so nested sections keep their context.
    """
    lines = markdown.splitlines()
    concepts: list[OKFConcept] = []

    stack: list[tuple[int, str]] = []   # (level, title) for building the path
    cur_title = default_title
    cur_level = 0
    buf: list[str] = []
    ordinal = 0

    def flush():
        nonlocal ordinal, buf
        body = "\n".join(buf).strip()
        if body:
            path = " > ".join(t for _, t in stack) or cur_title
            concepts.append(OKFConcept(
                concept=cur_title, section_path=path, text=body,
                ordinal=ordinal, level=cur_level,
                metadata={"has_table": any(_TABLE_ROW_RE.match(l) for l in buf)},
            ))
            ordinal += 1
        buf = []

    for line in lines:
        m = _HEADING_RE.match(line)
        if m:
            flush()
            level = len(m.group(1))
            title = m.group(2).strip() or default_title
            # maintain heading stack for section path
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            cur_title = title
            cur_level = level
        else:
            buf.append(line)
    flush()

    if not concepts:
        # No headings at all — treat the whole thing as one concept.
        body = markdown.strip()
        if body:
            concepts.append(OKFConcept(concept=default_title,
                                       section_path=default_title, text=body,
                                       ordinal=0, level=0))
    return concepts
