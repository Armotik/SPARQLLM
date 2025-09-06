# -*- coding: utf-8 -*-
"""GitHub MCP provider abstraction (client-side wrapper within SPARQLLM).

Objectif:
- Centraliser les appels REST GitHub nécessaires aux tools MCP exposés à SPARQLLM.
- Offrir une surface 'tools' cohérente: chaque méthode = un tool (nom fully-qualified).
- Retour JSON brut -> conversion JSON-LD gérée plus haut (slm_mcp_tool) ou via mappers dédiés.

Design:
- GitHubProvider gère auth, base URL, helpers (pagination simple optionnelle).
- METHODS dictionnaire pour introspection future (tools/list éventuel).

Tu peux étendre en ajoutant d'autres tools: issues, commits, releases, workflows, etc.
"""
from __future__ import annotations
import os, requests, typing as t

class GitHubProvider:
    BASE = "https://api.github.com"

    def __init__(self, token: str|None=None, user_agent: str="SPARQLLM-GitHubMCP/0.1"):
        self.token = token or os.getenv("GITHUB_TOKEN") or os.getenv("GITHUB_PAT")
        self.ua = user_agent
        if not self.token:
            # on n'interrompt pas, on peut quand même appeler les endpoints publics
            pass

    # --- low-level ---
    def _headers(self) -> dict:
        h = {
            "Accept": "application/vnd.github+json",
            "User-Agent": self.ua,
        }
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def _get(self, path: str, params: dict|None=None, timeout: int=30) -> tuple[int, t.Any]:
        url = self.BASE + path
        r = requests.get(url, headers=self._headers(), params=params or {}, timeout=timeout)
        try:
            data = r.json()
        except Exception:
            data = {"raw": r.text[:1000]}
        return r.status_code, data

    # --- tools implementations ---
    def tool_pull_requests_list(self, owner: str, repo: str, state: str="open", limit: int=20) -> dict:
        """Retourne un fragment JSON-LD ancré sur le repository avec les PR comme hasPart.

        Contract JSON-LD minimal:
        {
          "media_type": "application/ld+json",
          "jsonld": { ... graph fragment ... }
        }
        """
        params = {"state": state, "per_page": min(limit,100)}
        code, data = self._get(f"/repos/{owner}/{repo}/pulls", params)
        if code != 200:
            return {"error": code, "payload": data}

        repo_url = f"https://github.com/{owner}/{repo}"
        context = {
            "schema": "https://schema.org/",
            "prov": "http://www.w3.org/ns/prov#"
        }
        has_part = []
        for pr in data:
            number = pr.get("number")
            pr_url = pr.get("html_url") or f"{repo_url}/pull/{number}" if number else None
            user_login = pr.get("user",{}).get("login")
            author = None
            if user_login:
                author = {
                    "@id": f"https://github.com/{user_login}",
                    "@type": "schema:Person",
                    "schema:identifier": user_login
                }
            pr_obj = {
                "@id": pr_url,
                "@type": "schema:PullRequest",
                "schema:identifier": number,
                "schema:name": pr.get("title"),
                "schema:dateCreated": pr.get("created_at"),
                "schema:author": author,
                "schema:url": pr_url,
                "schema:status": pr.get("state")
            }
            # Nettoyage des None
            pr_obj = {k:v for k,v in pr_obj.items() if v is not None}
            has_part.append(pr_obj)

        jsonld = {
            "@context": context,
            "@id": repo_url,
            "@type": "schema:Repository",
            "schema:hasPart": has_part,
            "prov:generatedAtTime": data[0].get("created_at") if data else None,
            "schema:name": repo
        }
        # purge None
        jsonld = {k:v for k,v in jsonld.items() if v is not None}

        return {
            "media_type": "application/ld+json",
            "jsonld": jsonld,
            "graph_anchor": repo_url
        }

    # Placeholder pour autres tools
    def tool_issues_list(self, owner: str, repo: str, state: str="open", limit: int=20) -> dict:
        """Retourne un fragment JSON-LD ancré sur le repository avec les Issues comme hasPart.

        Même contrat que pour les PR: media_type + jsonld.
        """
        params = {"state": state, "per_page": min(limit,100)}
        code, data = self._get(f"/repos/{owner}/{repo}/issues", params)
        if code != 200:
            return {"error": code, "payload": data}
        repo_url = f"https://github.com/{owner}/{repo}"
        context = {
            "schema": "https://schema.org/",
            "prov": "http://www.w3.org/ns/prov#"
        }
        parts = []
        for it in data:
            if "pull_request" in it:  # exclure PR déguisées
                continue
            number = it.get("number")
            issue_url = it.get("html_url") or f"{repo_url}/issues/{number}" if number else None
            user_login = it.get("user",{}).get("login")
            author = None
            if user_login:
                author = {
                    "@id": f"https://github.com/{user_login}",
                    "@type": "schema:Person",
                    "schema:identifier": user_login
                }
            issue_obj = {
                "@id": issue_url,
                "@type": "schema:Issue",
                "schema:identifier": number,
                "schema:name": it.get("title"),
                "schema:dateCreated": it.get("created_at"),
                "schema:author": author,
                "schema:url": issue_url,
                "schema:status": it.get("state")
            }
            issue_obj = {k:v for k,v in issue_obj.items() if v is not None}
            parts.append(issue_obj)

        jsonld = {
            "@context": context,
            "@id": repo_url,
            "@type": "schema:Repository",
            "schema:hasPart": parts,
            "schema:name": repo
        }
        jsonld = {k:v for k,v in jsonld.items() if v is not None}

        return {
            "media_type": "application/ld+json",
            "jsonld": jsonld,
            "graph_anchor": repo_url
        }

    # --- registry introspection ---
    def list_tools(self) -> list[dict]:
        return [
            {"name": "github.pullRequests.list", "description": "List pull requests", "inputs": ["owner","repo","state","limit"]},
            {"name": "github.issues.list", "description": "List issues", "inputs": ["owner","repo","state","limit"]},
        ]

    def call(self, tool_name: str, args: dict) -> dict:
        if tool_name == "github.pullRequests.list":
            return self.tool_pull_requests_list(
                owner=args.get("owner"), repo=args.get("repo"), state=args.get("state","open"), limit=int(args.get("limit",20))
            )
        if tool_name == "github.issues.list":
            return self.tool_issues_list(
                owner=args.get("owner"), repo=args.get("repo"), state=args.get("state","open"), limit=int(args.get("limit",20))
            )
        return {"error": "unknown_tool", "tool": tool_name}

# Factory globale (option simple)
_provider_singleton: GitHubProvider|None = None

def get_provider() -> GitHubProvider:
    global _provider_singleton
    if _provider_singleton is None:
        _provider_singleton = GitHubProvider()
    return _provider_singleton
