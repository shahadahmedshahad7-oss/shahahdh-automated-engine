from __future__ import annotations

import base64
import binascii
import json
import logging
import os
import re
import socket
import threading
import xml.etree.ElementTree as ET
from datetime import datetime
from email.utils import parsedate_to_datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent
PAGE_PATH = ROOT / "jarvis.html"
MAX_REQUEST_BYTES = 16_384
MAX_UPSTREAM_BYTES = 16 * 1024 * 1024
MAX_IMAGE_BYTES = 12 * 1024 * 1024
IMAGE_MODEL = "gemini-nano-banana-2.1"
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/interactions"
ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp"}
LOG = logging.getLogger("sage")


class RequestError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


def _read_url(request: Request, timeout: int, limit: int = MAX_UPSTREAM_BYTES) -> bytes:
    with urlopen(request, timeout=timeout) as response:
        payload = response.read(limit + 1)
    if len(payload) > limit:
        raise RequestError(502, "The upstream response exceeded the size limit.")
    return payload


def _safe_https_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return ""
    if parsed.scheme != "https" or not parsed.netloc:
        return ""
    return value


def _format_date(value: str) -> str:
    if not value:
        return ""
    try:
        return parsedate_to_datetime(value).astimezone().strftime("%b %-d, %Y %-I:%M %p")
    except (TypeError, ValueError, OverflowError):
        try:
            return parsedate_to_datetime(value).strftime("%b %d, %Y %I:%M %p").lstrip("0")
        except (TypeError, ValueError, OverflowError):
            return value


def search_web(query: str) -> dict[str, object]:
    cleaned = query.strip()
    if not cleaned:
        raise RequestError(400, "Enter a search query.")
    if len(cleaned) > 240:
        raise RequestError(400, "Search queries must be 240 characters or fewer.")

    results: list[dict[str, str]] = []
    failures: list[str] = []
    news_url = "https://news.google.com/rss/search?" + urlencode(
        {"q": cleaned, "hl": "en-US", "gl": "US", "ceid": "US:en"}
    )
    try:
        request = Request(news_url, headers={"User-Agent": "Mozilla/5.0 SagePersonalAssistant/1.0"})
        root = ET.fromstring(_read_url(request, timeout=15))
        channel = root.find("channel")
        if channel is not None:
            for item in channel.findall("item")[:4]:
                title = (item.findtext("title") or "").strip()
                link = _safe_https_url((item.findtext("link") or "").strip())
                if not title or not link:
                    continue
                source_node = item.find("source")
                results.append(
                    {
                        "title": title,
                        "snippet": "",
                        "source": source_node.text.strip() if source_node is not None and source_node.text else "",
                        "published": _format_date(item.findtext("pubDate") or ""),
                        "url": link,
                        "sourceUrl": _safe_https_url(source_node.get("url", "")) if source_node is not None else "",
                        "kind": "news",
                    }
                )
    except (HTTPError, URLError, TimeoutError, socket.timeout, ET.ParseError, OSError, RequestError):
        failures.append("Live news search was unavailable.")

    wiki_answer = ""
    wiki_params = urlencode(
        {
            "action": "query",
            "generator": "search",
            "gsrsearch": cleaned,
            "gsrlimit": "4",
            "prop": "extracts",
            "exintro": "1",
            "explaintext": "1",
            "exchars": "700",
            "format": "json",
        }
    )
    try:
        request = Request(
            "https://en.wikipedia.org/w/api.php?" + wiki_params,
            headers={"User-Agent": "Mozilla/5.0 SagePersonalAssistant/1.0"},
        )
        data = json.loads(_read_url(request, timeout=15))
        pages = data.get("query", {}).get("pages", {})
        for page in list(pages.values())[:4]:
            title = str(page.get("title", "")).strip()
            if not title:
                continue
            extract = str(page.get("extract", "")).strip()
            if not wiki_answer and extract:
                wiki_answer = extract
            page_id = page.get("pageid")
            if not isinstance(page_id, int):
                continue
            results.append(
                {
                    "title": title,
                    "snippet": extract,
                    "source": "Wikipedia",
                    "published": "",
                    "url": f"https://en.wikipedia.org/?curid={page_id}",
                    "sourceUrl": "https://en.wikipedia.org/",
                    "kind": "encyclopedia",
                }
            )
    except (HTTPError, URLError, TimeoutError, socket.timeout, json.JSONDecodeError, OSError, RequestError):
        failures.append("Encyclopedia lookup was unavailable.")

    news_results = [item for item in results if item["kind"] == "news"]
    other_results = [item for item in results if item["kind"] != "news"]
    current_query = re.search(
        r"\b(latest|today|current|news|weather|stock|score|breaking|recent|forecast)\b",
        cleaned,
        re.IGNORECASE,
    )
    if current_query and news_results:
        headlines = [
            f"{item['title']} ({item['source']})" if item["source"] else item["title"]
            for item in news_results[:3]
        ]
        answer = "Current web headlines: " + "; ".join(headlines) + "."
    elif wiki_answer:
        answer = "Wikipedia summary: " + wiki_answer
    elif news_results:
        headlines = [
            f"{item['title']} ({item['source']})" if item["source"] else item["title"]
            for item in news_results[:2]
        ]
        answer = "The live news results include: " + "; ".join(headlines) + "."
    else:
        answer = ""

    combined = news_results + other_results[:4]
    return {"answer": answer, "results": combined[:7], "errors": failures}


def search_images(query: str) -> dict[str, object]:
    cleaned = query.strip()
    if not cleaned:
        raise RequestError(400, "Enter a search query.")
    if len(cleaned) > 240:
        raise RequestError(400, "Search queries must be 240 characters or fewer.")

    params = urlencode(
        {
            "action": "query",
            "generator": "search",
            "gsrsearch": "filetype:bitmap " + cleaned,
            "gsrnamespace": "6",
            "gsrlimit": "12",
            "prop": "imageinfo",
            "iiprop": "url",
            "iiurlwidth": "480",
            "format": "json",
        }
    )
    request = Request(
        "https://commons.wikimedia.org/w/api.php?" + params,
        headers={"User-Agent": "Mozilla/5.0 SagePersonalAssistant/1.0"},
    )
    try:
        data = json.loads(_read_url(request, timeout=20))
    except HTTPError as exc:
        LOG.warning("Commons image search rejected with HTTP status %s", exc.code)
        raise RequestError(502, "Image search is temporarily unavailable. Please try again.") from None
    except (URLError, TimeoutError, socket.timeout, json.JSONDecodeError, OSError, RequestError) as exc:
        LOG.warning("Commons image search failed: %s", type(exc).__name__)
        raise RequestError(502, "Could not complete image search. Please try again.") from None

    pages = data.get("query", {}).get("pages", {})
    results: list[dict[str, str]] = []
    for page in pages.values():
        title = str(page.get("title", "")).strip()
        imageinfo = page.get("imageinfo", [])
        if not title or not imageinfo or not isinstance(imageinfo[0], dict):
            continue
        image = imageinfo[0]
        thumbnail = _safe_https_url(str(image.get("thumburl", "")))
        page_url = _safe_https_url(str(image.get("descriptionurl", "")))
        if not thumbnail.startswith("https://thumb.wikimedia.org/") or not page_url.startswith(
            "https://commons.wikimedia.org/"
        ):
            continue
        results.append({"title": title.removeprefix("File:"), "thumbnail": thumbnail, "url": page_url})
    return {"results": results}


def generate_image(api_key: str, prompt: str) -> dict[str, str]:
    if not api_key or len(api_key) > 256:
        raise RequestError(400, "Enter a valid Google Gemini API key.")
    cleaned = prompt.strip()
    if not cleaned:
        raise RequestError(400, "Enter a description for the image.")
    if len(cleaned) > 1500:
        raise RequestError(400, "Image descriptions must be 1500 characters or fewer.")

    body = json.dumps(
        {
            "model": IMAGE_MODEL,
            "input": [{"type": "text", "text": cleaned}],
        }
    ).encode("utf-8")
    request = Request(
        GEMINI_ENDPOINT,
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
            "Api-Revision": "2026-05-20",
        },
        method="POST",
    )
    try:
        data = json.loads(_read_url(request, timeout=180))
    except HTTPError as exc:
        LOG.warning("Gemini request rejected with HTTP status %s", exc.code)
        raise RequestError(502, "Gemini rejected the image request. Check the API key, model access, and quota.") from None
    except (URLError, TimeoutError, socket.timeout, json.JSONDecodeError, OSError, RequestError) as exc:
        LOG.warning("Gemini image request failed: %s", type(exc).__name__)
        raise RequestError(502, "Could not complete the Gemini image request. Please try again.") from None

    output_image = data.get("output_image")
    if not isinstance(output_image, dict):
        raise RequestError(502, "Gemini did not return an image.")
    encoded = output_image.get("data")
    mime_type = output_image.get("mime_type")
    if not isinstance(encoded, str) or mime_type not in ALLOWED_IMAGE_TYPES:
        raise RequestError(502, "Gemini returned an unsupported image format.")
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise RequestError(502, "Gemini returned invalid image data.") from None
    if not image_bytes or len(image_bytes) > MAX_IMAGE_BYTES:
        raise RequestError(502, "The generated image was empty or exceeded the size limit.")
    return {"image": base64.b64encode(image_bytes).decode("ascii"), "mimeType": mime_type}


class JarvisHandler(BaseHTTPRequestHandler):
    server_version = "SAGEWeb/1.0"
    sys_version = ""

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "strict-origin-when-cross-origin")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self' 'unsafe-inline' https://accounts.google.com; "
            "style-src 'self' 'unsafe-inline'; img-src 'self' data: https://thumb.wikimedia.org https://*.googleusercontent.com; "
            "connect-src 'self' https://accounts.google.com https://oauth2.googleapis.com "
            "https://www.googleapis.com https://gmail.googleapis.com https://photospicker.googleapis.com; "
            "base-uri 'none'; frame-ancestors 'none'; form-action 'self' https://aistudio.google.com",
        )
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, payload: object) -> None:
        self._send(status, "application/json; charset=utf-8", json.dumps(payload, ensure_ascii=False).encode("utf-8"))

    def _read_json(self) -> dict[str, object]:
        if not self.headers.get("Content-Type", "").lower().startswith("application/json"):
            raise RequestError(415, "Expected a JSON request.")
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            raise RequestError(400, "Invalid request length.") from None
        if length < 1 or length > MAX_REQUEST_BYTES:
            raise RequestError(413, "Request body must be between 1 and 16384 bytes.")
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise RequestError(400, "Invalid JSON request.") from None
        if not isinstance(payload, dict):
            raise RequestError(400, "Expected a JSON object.")
        return payload

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path in {"/", "/jarvis.html"}:
            try:
                body = PAGE_PATH.read_bytes()
            except OSError:
                self._send_json(500, {"error": "S.A.G.E. page is unavailable."})
                return
            self._send(200, "text/html; charset=utf-8", body)
            return
        if path == "/healthz":
            self._send_json(200, {"ok": True})
            return
        if path == "/api/status":
            self._send_json(
                200,
                {
                    "ready": False,
                    "hosted": True,
                    "apps": [],
                    "capabilities": {"search": True, "imageSearch": True, "imageGeneration": True, "chat": False, "desktop": False},
                },
            )
            return
        if path == "/api/google-config":
            self._send_json(200, {"clientId": os.environ.get("GOOGLE_CLIENT_ID", "").strip()})
            return
        if path == "/api/brain/status":
            self._send_json(200, {"available": False, "model": "", "error": "Local AI is unavailable in hosted mode."})
            return
        self._send_json(404, {"error": "Not found."})

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        if path not in {"/api/search", "/api/images", "/api/image", "/api/chat", "/api/action"}:
            self._send_json(404, {"error": "Not found."})
            return
        try:
            payload = self._read_json()
            if path == "/api/search":
                query = payload.get("query")
                if not isinstance(query, str):
                    raise RequestError(400, "Enter a search query.")
                result = search_web(query)
                if not result["results"] and result["errors"]:
                    raise RequestError(502, "Live search sources are unavailable. Please try again.")
                self._send_json(200, result)
                return
            if path == "/api/images":
                query = payload.get("query")
                if not isinstance(query, str):
                    raise RequestError(400, "Enter an image search query.")
                self._send_json(200, search_images(query))
                return
            if path == "/api/image":
                api_key = payload.get("apiKey")
                prompt = payload.get("prompt")
                if not isinstance(api_key, str) or not isinstance(prompt, str):
                    raise RequestError(400, "Enter a Gemini API key and image description.")
                self._send_json(200, generate_image(api_key, prompt))
                return
            if path == "/api/chat":
                raise RequestError(503, "AI chat is not configured in the hosted web version. Web search is available.")
            raise RequestError(403, "Launching desktop apps is disabled in the hosted web version.")
        except RequestError as exc:
            self._send_json(exc.status, {"error": str(exc)})

    def log_message(self, format: str, *args: object) -> None:
        LOG.info("%s - %s", self.address_string(), format % args)


class JarvisServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main() -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    port = int(os.environ.get("PORT", "8765"))
    server = JarvisServer(("0.0.0.0", port), JarvisHandler)
    LOG.info("S.A.G.E. web server listening on port %s", port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
