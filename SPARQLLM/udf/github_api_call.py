
import requests
from rdflib import Graph, Namespace, URIRef, Literal, BNode
from rdflib.namespace import RDF, RDFS
import os
import logging

from SPARQLLM.udf.SPARQLLM import store


logger = logging.getLogger(__name__)


def github_api_call(owner, repo, endpoint="issues", token=None, limit=10):
    """
    UDF: Appelle l'API GitHub pour un repo donné et retourne un graphe RDF structuré (issues ou pull requests).
    Args:
        owner (str): Propriétaire du repo
        repo (str): Nom du repo
        endpoint (str): Endpoint GitHub (ex: 'issues', 'pulls')
        token (str): Token GitHub (optionnel)
        limit (int): Nombre max de résultats
    Returns:
        rdflib.Graph: Graphe RDF structuré (Repository, hasPart, PullRequest/Issue)
    """

    logger.debug(f"Calling github_api_call with owner={owner}, repo={repo}, endpoint={endpoint}, limit={limit}")    

    headers = {"Accept": "application/vnd.github+json"}
    if token is None:
        token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    url = f"https://api.github.com/repos/{owner}/{repo}/{endpoint}?per_page={limit}"
    logger.info(f"GitHub API call: {url} (endpoint={endpoint})")
    try:
        resp = requests.get(url, headers=headers)
        logger.debug(f"Response status: {resp.status_code}")
        resp.raise_for_status()
        data = resp.json()
        logger.debug(f"Response JSON: {str(data)[:500]}")
    except Exception as e:
        logger.error(f"GitHub API error: {e}")
        raise

    g =  store.get_context(BNode())
    SCHEMA = Namespace("https://schema.org/")
    PROV = Namespace("http://www.w3.org/ns/prov#")
    REPO_URI = URIRef(f"https://github.com/{owner}/{repo}")
    g.add((REPO_URI, RDF.type, SCHEMA.Repository))
    g.add((REPO_URI, SCHEMA.name, Literal(repo)))
    # Pour prov:generatedAtTime, on prend la date du premier élément si dispo
    if data and isinstance(data, list) and data[0].get("created_at"):
        g.add((REPO_URI, PROV.generatedAtTime, Literal(data[0]["created_at"])))
    for item in data:
        # Pull requests
        if endpoint == "pulls":
            pr_uri = URIRef(item.get("html_url"))
            g.add((pr_uri, RDF.type, SCHEMA.PullRequest))
            g.add((REPO_URI, SCHEMA.hasPart, pr_uri))
            if item.get("title"):
                g.add((pr_uri, SCHEMA.name, Literal(item["title"])))
            if item.get("number"):
                g.add((pr_uri, SCHEMA.identifier, Literal(item["number"])))
            if item.get("created_at"):
                g.add((pr_uri, SCHEMA.dateCreated, Literal(item["created_at"])))
            if item.get("state"):
                g.add((pr_uri, SCHEMA.status, Literal(item["state"])))
            if item.get("html_url"):
                g.add((pr_uri, SCHEMA.url, Literal(item["html_url"])))
            # Auteur
            user = item.get("user")
            if user and user.get("login"):
                author_uri = URIRef(f"https://github.com/{user['login']}")
                g.add((pr_uri, SCHEMA.author, author_uri))
                g.add((author_uri, RDF.type, SCHEMA.Person))
                g.add((author_uri, SCHEMA.identifier, Literal(user["login"])))
        # Issues (par défaut)
        else:
            # Exclure les PR déguisées
            if "pull_request" in item:
                continue
            issue_uri = URIRef(item.get("html_url"))
            g.add((issue_uri, RDF.type, SCHEMA.Issue))
            g.add((REPO_URI, SCHEMA.hasPart, issue_uri))
            if item.get("title"):
                g.add((issue_uri, RDFS.label, Literal(item["title"])))
            if item.get("state"):
                g.add((issue_uri, SCHEMA.status, Literal(item["state"])))
            if item.get("number"):
                g.add((issue_uri, SCHEMA.identifier, Literal(item["number"])))
            if item.get("html_url"):
                g.add((issue_uri, SCHEMA.url, Literal(item["html_url"])))
            if item.get("created_at"):
                g.add((issue_uri, SCHEMA.dateCreated, Literal(item["created_at"])))
            # Auteur
            user = item.get("user")
            if user and user.get("login"):
                author_uri = URIRef(f"https://github.com/{user['login']}")
                g.add((issue_uri, SCHEMA.author, author_uri))
                g.add((author_uri, RDF.type, SCHEMA.Person))
                g.add((author_uri, SCHEMA.identifier, Literal(user["login"])))
    for s, p, o in g:
        print(s, p, o)
    return g.identifier
