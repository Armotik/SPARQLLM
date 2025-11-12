# -*- coding: utf-8 -*-
import os, json
from .client import MCPClient
from .postprocess import postprocess_to_rdf

# cache d'instances client par handle
_MCP = MCPClient()
_HANDLES = {}

def _get_conf(key, default=None):
    # à adapter à ta manière de lire [Requests]
    import os
    return os.environ.get(key, default)

def MCP_CONNECT(transport: str=None, target: str=None, token: str=""):
    """
    transport: 'http' ou 'stdio' (défaut: SLM-MCP-TRANSPORT)
    target: base_url (http) ou commande (stdio) (défaut: SLM-MCP-TARGET)
    token: 'Bearer XYZ' sinon lu dans env SLM-MCP-TOKENENV
    """
    transport = transport or _get_conf("SLM-MCP-TRANSPORT","http")
    target    = target    or _get_conf("SLM-MCP-TARGET","")
    if not token:
        envname = _get_conf("SLM-MCP-TOKENENV","")
        if envname and os.environ.get(envname):
            token = f"Bearer {os.environ[envname]}"
    handle = f"mcp://{abs(hash((transport,target,token)))%10**10}"
    if handle in _HANDLES:
        return handle
    if transport == "stdio":
        _MCP.connect_stdio(handle, target.split(" "))
    else:
        _MCP.connect_http(handle, target, token or None)
    _HANDLES[handle] = True
    return handle

def MCP_TOOLS_LIST(handle: str) -> str:
    return json.dumps(_MCP.tools_list(handle))

def MCP_TOOL(
        engine, handle: str,
        tool: str,
        args_json: str,
        mapping_url: str = None
) -> str:
    args = json.loads(args_json) if isinstance(args_json,str) else args_json
    raw  = _MCP.tools_call(handle, tool, args)

    if not mapping_url:
        res  = postprocess_to_rdf(engine, tool, raw)
        # mode RDF-only optionnel
        rdf_only = (_get_conf("SLM-MCP-RDF-ONLY","true").lower() == "true")
        if rdf_only and "graph" not in res:
            return json.dumps({"ok": False, "error": "Non-RDF output from MCP tool", "tool": tool})
        return json.dumps(res)
    else :

        # pour l'instant on retourne un graph vide
        # TODO: implémenter le mapping
        return json.dumps({"ok": True, "graph": []})
