"""OCR provider (optional).

Documents that are already text (``.md``, ``.txt``, ``.csv``) skip OCR. Binary
documents (PDF, images) are sent to an adopter-configured OCR endpoint
(``OCR_HOST_URL``) — e.g. a self-hosted DeepSeek-OCR / Unlimited-OCR server that
returns markdown. When no OCR endpoint is configured, DMem raises a clear error
if a binary document is submitted, rather than silently indexing garbage.

The endpoint contract is intentionally minimal: POST the raw bytes (plus an
optional filename), receive ``{"markdown": "..."}`` or plain text back.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from ..config import OCRConfig
from ..errors import ProviderError


@runtime_checkable
class OCRProvider(Protocol):
    def to_markdown(self, data: bytes, filename: Optional[str] = None) -> str:
        ...


class HTTPOCRProvider:
    """POSTs bytes to an OCR endpoint that returns markdown. Needs 'http' extra."""

    def __init__(self, cfg: OCRConfig):
        if not cfg.host_url:
            raise ProviderError("HTTPOCRProvider requires OCR_HOST_URL.")
        try:
            import httpx  # noqa: F401
        except ImportError as e:  # pragma: no cover - depends on extra
            raise ProviderError(
                "OCR needs the 'http' extra: pip install dmem[http]"
            ) from e
        self._httpx = __import__("httpx")
        self.cfg = cfg
        self._url = cfg.host_url.rstrip("/")

    def to_markdown(self, data: bytes, filename: Optional[str] = None) -> str:
        headers = {}
        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"
        try:
            resp = self._httpx.post(
                self._url,
                headers=headers,
                files={"file": (filename or "document", data)},
                timeout=300.0,
            )
            resp.raise_for_status()
        except Exception as e:  # pragma: no cover - network dependent
            raise ProviderError(f"OCR request failed: {e}") from e
        ctype = resp.headers.get("content-type", "")
        if "application/json" in ctype:
            body = resp.json()
            return body.get("markdown") or body.get("text") or ""
        return resp.text


def build_ocr_provider(cfg: OCRConfig) -> Optional[OCRProvider]:
    """Return an OCR provider or None when OCR is not configured."""
    if cfg.enabled:
        return HTTPOCRProvider(cfg)
    return None
