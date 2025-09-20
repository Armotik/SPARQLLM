# -*- coding: utf-8 -*-
"""
Browser (Playwright) MCP Provider.
Ajouts compliance:
- robots.txt (désactivable: ignore_robots)
- throttling par host (min_delay_ms)
- redaction basique PII (redact_pi)
- snippet_only (ne conserve qu'un extrait court)
- détection paywall/captcha
"""
from __future__ import annotations
import hashlib
import re
import time
import threading
import urllib.parse
import urllib.request
from typing import Dict, Any, Callable
from datetime import datetime, timezone

try:
    from playwright.sync_api import sync_playwright
except ImportError as e:
    raise ImportError("Installez 'playwright' puis 'playwright install chromium'") from e


# --- Contrôles & helpers compliance ---------------------------------
_RATE_LOCK = threading.Lock()
_LAST_CALL_PER_HOST: Dict[str, float] = {}
_DEFAULT_MIN_DELAY_SEC = 2.0  # délai par défaut entre deux hits sur le même host

_ROBOTS_CACHE: Dict[str, str] = {}
_ROBOTS_TTL_SEC = 3600
_ROBOTS_FETCHED: Dict[str, float] = {}

PAYWALL_HINTS = (
    "subscribe to", "abonnez-vous", "log in to continue", "create an account",
    "paywall", "subscriber-only", "purchase access"
)
CAPTCHA_HINTS = ("captcha", "are you a robot", "verify you are human")

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"\b(?:\+?\d[\d .-]{6,}\d)\b")

def _throttle(host: str, min_delay_sec: float):
    with _RATE_LOCK:
        last = _LAST_CALL_PER_HOST.get(host, 0.0)
        now = time.time()
        delta = now - last
        if delta < min_delay_sec:
            time.sleep(min_delay_sec - delta)
        _LAST_CALL_PER_HOST[host] = time.time()

def _fetch_robots(base: str) -> str:
    now = time.time()
    if base in _ROBOTS_CACHE and now - _ROBOTS_FETCHED.get(base, 0) < _ROBOTS_TTL_SEC:
        return _ROBOTS_CACHE[base]
    try:
        with urllib.request.urlopen(base + "/robots.txt", timeout=5) as r:
            if getattr(r, "status", 200) == 200:
                txt = r.read().decode("utf-8", "ignore")
                _ROBOTS_CACHE[base] = txt
                _ROBOTS_FETCHED[base] = now
                return txt
    except Exception:
        pass
    _ROBOTS_CACHE[base] = ""
    _ROBOTS_FETCHED[base] = now
    return ""

def _robots_disallow(url: str, ua: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    robots = _fetch_robots(base)
    if not robots:
        return False
    path = parsed.path or "/"
    ua_block = False
    for line in robots.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        low = line.lower()
        if low.startswith("user-agent:"):
            agent = line.split(":", 1)[1].strip()
            ua_block = (agent == "*" or agent.lower() in ua.lower())
        elif ua_block and low.startswith("disallow:"):
            rule = line.split(":", 1)[1].strip() or "/"
            if rule == "/":
                return True
            if path.startswith(rule):
                return True
    return False

def _redact(text: str) -> str:
    # Redaction simple e-mails & numéros
    text = EMAIL_RE.sub("[EMAIL_REDACTED]", text)
    text = PHONE_RE.sub("[PHONE_REDACTED]", text)
    return text

def _detect_flags(body: str) -> Dict[str, bool]:
    lower = body.lower()
    paywall = any(h in lower for h in PAYWALL_HINTS)
    captcha = any(h in lower for h in CAPTCHA_HINTS)
    denied = ("access denied" in lower) or ("error code" in lower)
    return {
        "access_denied": denied,
        "paywall": paywall,
        "captcha": captcha
    }


class BrowserProvider:
    TOOL_PREFIX = "browser."
    _SAFE_TIMEOUT = 30000  # ms

    def __init__(self):
        self._tools: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
            "browser.snapshot": self._tool_snapshot
        }

    def tools(self):
        return list(self._tools.keys())

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    def call(self, tool_name: str, args: Dict[str, Any]):
        if tool_name not in self._tools:
            raise ValueError(f"Tool Browser inconnu: {tool_name}")
        return self._tools[tool_name](args or {})

    def get(self, tool_name: str, **kwargs):
        return self.call(tool_name, kwargs)

    def snapshot(self, url: str, **kwargs):
        p = dict(kwargs); p["url"] = url
        return self.call("browser.snapshot", p)

    # Impl tool ------------------------------------------------
    def _tool_snapshot(self, args: Dict[str, Any]) -> Dict[str, Any]:
        url = args.get("url")
        if not url:
            raise ValueError("Paramètre 'url' requis (browser.snapshot)")

        wait_ms = int(args.get("wait_ms", 0))
        selector = args.get("selector")
        include_html = bool(args.get("include_html", False))
        text_max = int(args.get("text_max", 50000))
        snippet_only = bool(args.get("snippet_only", False))
        redact_pi = bool(args.get("redact_pi", False))
        user_agent = args.get("user_agent") or (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36"
        )
        locale = args.get("locale", "en-US")
        disable_js = bool(args.get("disable_js", False))
        ignore_robots = bool(args.get("ignore_robots", False))
        min_delay_ms = int(args.get("min_delay_ms", _DEFAULT_MIN_DELAY_SEC * 1000))
        ignore_https_errors = bool(args.get("ignore_https_errors", False))
        wait_until = args.get("wait_until") or "load"  # load|domcontentloaded|networkidle|commit
        timeout_ms = int(args.get("timeout_ms", self._SAFE_TIMEOUT))
        raw_headers = args.get("headers") or {}
        headers = {}
        if isinstance(raw_headers, dict):
            headers = {str(k): str(v) for k, v in raw_headers.items()}

        # Throttle
        host = urllib.parse.urlparse(url).netloc
        _throttle(host, max(0.2, min_delay_ms / 1000.0))

        # robots.txt
        if not ignore_robots and _robots_disallow(url, user_agent):
            now = datetime.now(timezone.utc).isoformat()
            return {
                "media_type": "application/ld+json",
                "jsonld": {
                    "@context": {"schema": "https://schema.org/"},
                    "@type": "schema:WebPage",
                    "schema:url": url,
                    "schema:name": "Blocked by robots.txt",
                    "schema:dateCreated": now,
                    "schema:text": "",
                    "schema:breadcrumb": "robots-blocked"
                }
            }

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            ctx = browser.new_context(
                user_agent=user_agent,
                locale=locale,
                java_script_enabled=not disable_js,
                extra_http_headers=headers or None,
                ignore_https_errors=ignore_https_errors
            )
            page = ctx.new_page()
            try:
                response = page.goto(url, timeout=timeout_ms, wait_until=wait_until)
            except Exception as nav_err:
                now = datetime.now(timezone.utc).isoformat()
                hsrc = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
                page_id = f"urn:snapshot:{hsrc}"
                jsonld = {
                    "@context": {"schema": "https://schema.org/"},
                    "@id": page_id,
                    "@type": "schema:WebPage",
                    "schema:url": url,
                    "schema:name": "Connection Error",
                    "schema:breadcrumb": "connection-error",
                    "schema:dateCreated": now,
                    "schema:text": "",
                    "schema:identifier": page_id,
                    "schema:comment": f"Navigation failed: {type(nav_err).__name__}"
                }
                browser.close()
                return {"media_type": "application/ld+json", "jsonld": jsonld}
            if wait_ms > 0:
                page.wait_for_timeout(wait_ms)

            status = None
            try:
                if response:
                    status = response.status
            except Exception:
                pass

            try:
                title = page.title() or ""
            except Exception:
                title = ""

            meta_desc = ""
            try:
                el = page.locator("meta[name='description']")
                if el.count() > 0:
                    meta_desc = el.first.get_attribute("content") or ""
            except Exception:
                pass

            if selector:
                try:
                    body_text = page.locator(selector).inner_text()
                except Exception:
                    body_text = ""
            else:
                try:
                    body_text = page.evaluate("() => document.body.innerText") or ""
                except Exception:
                    body_text = ""

            html_full = ""
            if include_html and not snippet_only:
                try:
                    html_full = page.content()
                except Exception:
                    html_full = ""

            browser.close()

        body_text = body_text.strip()
        flags = _detect_flags(body_text)

        # --- Retour anticipé si accès restreint (paywall / captcha / denied) ---
        if flags.get("access_denied") or flags.get("paywall") or flags.get("captcha"):
            now = datetime.now(timezone.utc).isoformat()
            import hashlib as _hashlib  # import local pour éviter dépendance si renommage en tête
            hsrc = _hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
            page_id = f"urn:snapshot:{hsrc}"
            if flags["access_denied"]:
                name = "Access Denied"; breadcrumb = "access-denied"
            elif flags["paywall"]:
                name = "Paywall"; breadcrumb = "paywall"
            else:
                name = "CAPTCHA"; breadcrumb = "captcha"
            jsonld = {
                "@context": {"schema": "https://schema.org/"},
                "@id": page_id,
                "@type": "schema:WebPage",
                "schema:url": url,
                "schema:name": name,
                "schema:dateCreated": now,
                "schema:text": "",
                "schema:breadcrumb": breadcrumb,
                "schema:inLanguage": locale,
                "schema:identifier": page_id,
                "schema:comment": "Content suppressed due to access restriction flag."
            }
            if status is not None:
                jsonld["schema:interactionStatistic"] = {
                    "@type": "schema:InteractionCounter",
                    "schema:interactionType": f"HTTP {status}"
                }
            return {
                "media_type": "application/ld+json",
                "jsonld": jsonld
            }

        # Redaction PII
        if redact_pi:
            body_text = _redact(body_text)

        # Tronquer
        if snippet_only:
            # Extrait court
            if len(body_text) > min(500, text_max):
                body_text = body_text[:min(500, text_max)] + "…"
            include_html = False  # on force off
            html_full = ""
        else:
            if len(body_text) > text_max:
                body_text = body_text[:text_max] + "…"

        now = datetime.now(timezone.utc).isoformat()
        hsrc = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
        page_id = f"urn:snapshot:{hsrc}"

        jsonld = {
            "@context": { "schema": "https://schema.org/" },
            "@id": page_id,
            "@type": "schema:WebPage",
            "schema:url": url,
            "schema:name": title or (
                "Access Denied" if flags["access_denied"] else (
                    "Paywall" if flags["paywall"] else (
                        "CAPTCHA" if flags["captcha"] else url
                    )
                )
            ),
            "schema:dateCreated": now,
            "schema:description": meta_desc,
            "schema:text": body_text,
            "schema:inLanguage": locale,
            "schema:identifier": page_id
        }

        if status is not None:
            jsonld["schema:interactionStatistic"] = {
                "@type": "schema:InteractionCounter",
                "schema:interactionType": f"HTTP {status}"
            }

        if flags["access_denied"]:
            jsonld["schema:breadcrumb"] = "access-denied"
        elif flags["paywall"]:
            jsonld["schema:breadcrumb"] = "paywall"
        elif flags["captcha"]:
            jsonld["schema:breadcrumb"] = "captcha"

        jsonld["schema:potentialAction"] = {
            "@type": "schema:SearchAction",
            "schema:target": url
        }

        if include_html and html_full:
            jsonld["schema:encoding"] = {
                "@type": "schema:MediaObject",
                "schema:encodingFormat": "text/html",
                "schema:text": html_full
            }

        # Conformité (métadonnées internes)
        jsonld["schema:isAccessibleForFree"] = True
        jsonld["schema:comment"] = "Fetched with compliance safeguards (robots, throttle, truncation)."

        return {
            "media_type": "application/ld+json",
            "jsonld": jsonld
        }


_browser_provider_singleton: BrowserProvider | None = None

def get_browser_provider() -> BrowserProvider:
    global _browser_provider_singleton
    if _browser_provider_singleton is None:
        _browser_provider_singleton = BrowserProvider()
    return _browser_provider_singleton