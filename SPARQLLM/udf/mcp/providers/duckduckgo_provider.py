# -*- coding: utf-8 -*-
"""
DuckDuckGo MCP Provider (ddgs).
"""
from __future__ import annotations
import hashlib
import urllib.parse
from typing import Dict, Any, Callable
from datetime import datetime, timezone

# Compat import: d'abord ddgs (nouveau nom), fallback ancien paquet si présent
try:
    from ddgs import DDGS
except ImportError:  # fallback si ddgs pas encore installé
    try:
        from duckduckgo_search import DDGS  # type: ignore
    except ImportError as e:
        raise ImportError("Installez le paquet 'ddgs' (pip install ddgs).") from e


class DuckDuckGoProvider:
    TOOL_PREFIX = "duckduckgo."
    DEFAULT_REGION = "wt-wt"
    SAFESEARCH_VALUES = {"off", "moderate", "strict"}

    def __init__(self):
        self._tools: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
            "duckduckgo.search": self._tool_search
        }

    def tools(self):
        return list(self._tools.keys())

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    def call(self, tool_name: str, args: Dict[str, Any]):
        if tool_name not in self._tools:
            raise ValueError(f"Tool DuckDuckGo inconnu: {tool_name}")
        return self._tools[tool_name](args or {})

    def get(self, tool_name: str, **kwargs):
        return self.call(tool_name, kwargs)

    def search(self, query: str, **kwargs):
        params = dict(kwargs)
        params["query"] = query
        return self.call("duckduckgo.search", params)

    def _tool_search(self, args: Dict[str, Any]) -> Dict[str, Any]:
        query = args.get("query") or args.get("q")
        if not query:
            raise ValueError("Paramètre 'query' requis (duckduckgo.search)")
        try:
            limit = int(args.get("limit", 10))
        except Exception:
            limit = 10
        limit = max(1, min(limit, 50))
        region = args.get("region", self.DEFAULT_REGION)
        safesearch = (args.get("safesearch") or "moderate").lower()
        if safesearch not in self.SAFESEARCH_VALUES:
            safesearch = "moderate"

        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, region=region, safesearch=safesearch, max_results=limit):
                if r:
                    results.append(r)

        now = datetime.now(timezone.utc).isoformat()
        feed_id = f"urn:duckduckgo:search:{urllib.parse.quote_plus(query)}"

        elements = []
        for i, r in enumerate(results):
            raw_id = f"{r.get('href','')}-{r.get('title','')}"
            rid = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:16]
            item_id = f"urn:duckduckgo:result:{rid}"
            href = r.get("href") or (item_id + "#page")
            title = r.get("title")
            desc = r.get("body")
            elements.append({
                "@id": item_id,
                "@type": "schema:DataFeedItem",
                "schema:position": i + 1,
                "schema:item": {
                    "@id": href,
                    "@type": "schema:WebPage",
                    "schema:name": title,
                    "schema:description": desc
                }
            })

        jsonld = {
            "@context": {"schema": "https://schema.org/"},
            "@id": feed_id,
            "@type": "schema:DataFeed",
            "schema:name": f"DuckDuckGo search results for {query}",
            "schema:query": query,
            "schema:dateCreated": now,
            "schema:dataFeedElement": elements
        }
        return {
            "media_type": "application/ld+json",
            "jsonld": jsonld
        }


_duck_provider_singleton: DuckDuckGoProvider | None = None

def get_duckduckgo_provider() -> DuckDuckGoProvider:
    global _duck_provider_singleton
    if _duck_provider_singleton is None:
        _duck_provider_singleton = DuckDuckGoProvider()
    return _duck_provider_singleton