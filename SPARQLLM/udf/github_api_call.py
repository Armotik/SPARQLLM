import requests
from rdflib import Graph, Namespace, URIRef, Literal, BNode
from rdflib.namespace import RDF, RDFS
import os
import logging
import json

from SPARQLLM.udf.SPARQLLM import store 
from SPARQLLM.udf.mcp.slm_mcp_tool import slm_mcp_tool


logger = logging.getLogger(__name__)


def github_api_call(owner, repo, endpoint="pulls", token=None, limit=10):
    """
    Wrapper for calling the equivalent of slm:SLM-MCP-TOOL for GitHub API.
    Args:
        owner (str): Repository owner
        repo (str): Repository name
        endpoint (str): GitHub endpoint (e.g., 'issues', 'pulls')
        token (str): GitHub token (optional, not used here)
        limit (int): Maximum number of results
    Returns:
        URIRef: IRI of the RDF graph structured with the results
    """

    logger.debug(f"Calling github_api_call with owner={owner}, repo={repo}, endpoint={endpoint}, limit={limit}")

    # Prepare parameters for MCP tool
    params = {
        "owner": owner,
        "repo": repo,
        "state": "open",
        "limit": limit
    }
    if str(endpoint) == "pulls":
        tool_name = "github.pullRequests.list"
    else:
        tool_name = "github.issues.list"

    # Convert parameters to JSON
    args_json = json.dumps(params)

    # Call slm_mcp_tool
    try:
        graph_iri = slm_mcp_tool("github", tool_name, args_json)
        logger.info(f"slm_mcp_tool call successful for {tool_name} with params {params}")
    except Exception as e:
        logger.error(f"Error calling slm_mcp_tool: {e}")
        raise

    return graph_iri
