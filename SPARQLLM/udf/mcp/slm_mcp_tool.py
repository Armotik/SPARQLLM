# SPARQLLM/udf/mcp/slm_mcp_tool.py
# -*- coding: utf-8 -*-

import json, hashlib, traceback, logging
from typing import Union, Dict, Any, List

from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDF, XSD
from rdflib import ConjunctiveGraph, Graph, URIRef,BNode

from SPARQLLM.udf.SPARQLLM import store
from SPARQLLM.utils.utils import named_graph_exists

from SPARQLLM.udf.mcp.client import MCPClient
from SPARQLLM.udf.mcp.providers.github_provider import get_provider
from SPARQLLM.udf.mcp.providers.postgres_provider import get_pg_provider

logger = logging.getLogger(__name__)

# --- MCP client (adapter) ---
# Tu peux brancher ici ton client HTTP/STDIO ; il suffit d'implémenter tools_call(handle, tool, args)
# Exemple : from SPARQLLM.udf.mcp.client import MCPClient ; _MCP = MCPClient()
#_MCP = None  # à injecter au démarrage (voir en bas)
_MCP = MCPClient()

import os
gh_token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GITHUB_PAT")

# Option 1 : handle HTTP générique (si un jour un vrai serveur MCP GitHub est dispo)
# if gh_token:
#     _MCP.connect_http("github-http", "https://api.github.com", f"Bearer {gh_token}")

# Option 2 : handle statique + tool spécialisé GitHub PRs
_MCP.connect_static("github")

_provider = get_provider()
_pg_provider = get_pg_provider()

def _tool_dispatch(tool_args: dict, tool_name: str):
    return _provider.call(tool_name, tool_args)

_MCP.register_static_tool("github", "github.pullRequests.list", lambda a: _tool_dispatch(a, "github.pullRequests.list"))
_MCP.register_static_tool("github", "github.issues.list", lambda a: _tool_dispatch(a, "github.issues.list"))
_MCP.connect_static("postgres")
_MCP.register_static_tool("postgres", "postgres.tables.list", lambda a: _pg_provider.call("postgres.tables.list", a))
_MCP.register_static_tool("postgres", "postgres.table.preview", lambda a: _pg_provider.call("postgres.table.preview", a))
_MCP.register_static_tool("postgres", "postgres.sql.query", lambda a: _pg_provider.call("postgres.sql.query", a))
_MCP.connect_stdio("echo", ["python","SPARQLLM/servers/echo_mcp_server.py"])



SCHEMA = Namespace("https://schema.org/")

# --- mappers optionnels par tool (JSON -> JSON-LD) ---
PROFILE_MAPPERS = {}

def register_mapper(tool_name: str, fn):
    """fn: (dict|list) -> dict(JSON-LD)"""
    PROFILE_MAPPERS[tool_name] = fn

def _heuristic_json_to_jsonld(data: Union[Dict[str,Any], List[Dict[str,Any]]]) -> Dict[str,Any]:
    """Plan B: JSON -> JSON-LD minimal ; essaie de trouver une liste d'objets et des IDs."""
    def to_item(o: Dict[str,Any]):
        if not isinstance(o, dict):
            return None
        iid = o.get("html_url") or o.get("url") or o.get("id")
        it = {"@type": "Thing"}
        for k, v in o.items():
            if isinstance(v, (str, int, float, bool)) and v is not None:
                it[k] = v
        if iid:
            it["@id"] = str(iid)
        return it

    items: List[Dict[str,Any]] = []
    if isinstance(data, list):
        items = [x for x in map(to_item, data) if x]
    elif isinstance(data, dict):
        arr = next((v for v in data.values() if isinstance(v, list) and v and isinstance(v[0], dict)), None)
        items = [x for x in map(to_item, arr or [data]) if x]

    return {"@context": "https://schema.org/", "@graph": items}

def _jsonld_to_named_graph(jsonld_obj: Union[Dict[str,Any], str],
                           graph_name_hint: str) -> URIRef:
    """Parse JSON-LD -> Graph tmp -> copie dans named graph store -> retourne IRI."""
    if isinstance(jsonld_obj, dict):
        s = json.dumps(jsonld_obj)
    else:
        s = jsonld_obj  # string JSON-LD
    # 1) parse dans un Graph rdflib
    gtmp = Graph()
    gtmp.parse(data=s, format="json-ld")

    # 2) IRI du named graph (stable si même contenu)
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]
    graph_uri = URIRef(graph_name_hint or f"urn:mcp:graph:{h}")

    # 3) créer/récupérer le named graph dans le store
    if named_graph_exists(store, graph_uri):
        logger.debug(f"[MCP] Graph {graph_uri} already exists")
        return graph_uri
    named_graph = store.get_context(graph_uri)

    # 4) copier les triples
    for t in gtmp:
        named_graph.add(t)

    return graph_uri

def json_to_jsonld(data):
    """
    Convert JSON data to JSON-LD format.
    """
    context = "https://schema.org/"
    jsonld_data = {
        "@context": context,
        "@graph": data
    }
    return jsonld_data


def slm_mcp_tool(handle: str,
                 tool_name: str,
                 args_json: str,
                 graph_name_hint: str = None) -> Union[URIRef, None]:
    """
    Appelle un serveur MCP, transforme le résultat en RDF, crée un named graph dans `store`,
    et *retourne l'IRI du graph* (comme slm_csv).

    - handle: identifiant de connexion à ton MCP (que tu gères côté client)
    - tool_name: nom du tool MCP ("github.pullRequests.list", "openlibrary.search", ...)
    - args_json: string JSON des arguments (ou dict si tu adaptes)
    - graph_name_hint: URI/URN à utiliser si tu veux forcer le nom du graphe

    Retour: URIRef (graph IRI) ou None en cas d'erreur.
    """
    logger.debug(f"[MCP] call tool={tool_name} args={args_json}")
    try:
        assert _MCP is not None, "MCP client not configured"
        args = json.loads(args_json) if isinstance(args_json, str) else args_json

        # (ré)enregistrement paresseux si handle github mais tools absents (cas import ordre)
        if handle == "github":
            try:
                handles = getattr(_MCP, "_handles", {})
                if "github" not in handles:
                    logger.debug("[MCP] handle 'github' absent -> reconnect_static")
                    _MCP.connect_static("github")
                if not _MCP.has_tool("github", "github.pullRequests.list"):
                    logger.debug("[MCP] (re)register github.pullRequests.list")
                    _MCP.register_static_tool("github", "github.pullRequests.list", lambda a: _tool_dispatch(a, "github.pullRequests.list"))
                if not _MCP.has_tool("github", "github.issues.list"):
                    logger.debug("[MCP] (re)register github.issues.list")
                    _MCP.register_static_tool("github", "github.issues.list", lambda a: _tool_dispatch(a, "github.issues.list"))
                logger.debug(f"[MCP] Handles actifs: {list(handles.keys())}")
            except Exception as re_err:
                logger.warning(f"[MCP] Echec re-registration github tools: {re_err}")

        # 1) appel MCP
        result = _MCP.tools_call(handle, tool_name, args)
        print("MCP result:", result)

        # 2) cas JSON-LD natif
        if isinstance(result, dict) and result.get("media_type") == "application/ld+json":
            jsonld_data = result.get("jsonld")
            print("MCP JSON-LD:", json.dumps(jsonld_data))
            graph_uri = BNode()
            named_graph = store.get_context(graph_uri)
            named_graph.parse(data=json.dumps(jsonld_data), format="json-ld")
            print("Named graph has", len(named_graph), "triples")
            for t in named_graph:
                print(t)
            return graph_uri

    # 3) mapper dédié si disponible
        mapper = PROFILE_MAPPERS.get(tool_name)
        if mapper and isinstance(result, (dict, list)):
            jsonld = mapper(result)
            return _jsonld_to_named_graph(jsonld, graph_name_hint)

        # 4) heuristique générique JSON -> JSON-LD
        if isinstance(result, (dict, list)):
            print("MCP JSON:", json.dumps(result))
            jsonld_data = json_to_jsonld(result)
#            print("MCP JSON-LD:", json.dumps(jsonld_data))

            graph_uri = BNode()
            named_graph = store.get_context(graph_uri)
            named_graph.parse(data=json.dumps(jsonld_data), format="json-ld")
            # print("Named graph has", len(named_graph), "triples")
            # for t in named_graph:
            #     print(t)
            return graph_uri

        # 5) fallback: pas de RDF possible
        logger.warning(f"[MCP] Non-RDF output for tool {tool_name}: {type(result)}")
        return None

    except Exception as e:
        logger.error(f"[MCP] Error calling tool: {e}")
        traceback.print_exc()
        return None
