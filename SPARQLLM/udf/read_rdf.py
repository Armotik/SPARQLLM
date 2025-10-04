import hashlib
import rdflib
from rdflib import Graph, Literal, URIRef, BNode
from rdflib.namespace import XSD
from rdflib.plugins.sparql import prepareQuery
from rdflib.plugins.sparql.operators import register_custom_function

from string import Template
from urllib.parse import urlencode,quote
from urllib.request import Request, urlopen

import os
import json

import requests
import html
import html2text
import unidecode
from urllib.parse import urlparse


from SPARQLLM.udf.SPARQLLM import store
from SPARQLLM.config import ConfigSingleton
from SPARQLLM.utils.utils import named_graph_exists, print_result_as_table

import logging
logger = logging.getLogger(__name__)

config = ConfigSingleton()

def read_rdf(path_uri,format="turtle"):
    logger.debug(f"uri: {path_uri}")    
    graph_uri = BNode()    
    named_graph = store.get_context(graph_uri)

    try:
        path= urlparse(path_uri).path
        logger.debug(f"Reading {path} with format {format}")
        named_graph.parse(path, format=str(format))
        logger.debug(f"Graph {graph_uri} has {len(named_graph)} triples")

    except requests.exceptions.RequestException as e:
        logger.error("Error reading {uri} : {e}")
        return graph_uri
    return graph_uri


def load_rdf_file(file_path: str, format: str | None = None):
    """Load an RDF file from a local filesystem path into the global store and
    return the named graph URI. Similar to read_rdf() but strictly for local files
    (no URL parsing) and with light format auto-detection.

    Parameters
    ----------
    file_path : str
        Path to a local RDF file (e.g. data/events.ttl)
    format : str | None
        Explicit rdflib parse format (e.g. "turtle", "nt", "xml", "json-ld"). If None,
        the function will guess from the file extension (.ttl, .nt, .nq, .rdf, .xml, .jsonld, .trig, .n3).

    Returns
    -------
    rdflib.term.URIRef
        The graph URI in the shared store.
    """
    file_path = os.path.abspath(file_path)
    if not os.path.exists(file_path):
        logger.warning(f"LOAD: file does not exist: {file_path}")
        # still return a deterministic URI so caller can reference (empty) graph
        return URIRef(f"http://loadrdf.org/absent/{hashlib.sha256(file_path.encode()).hexdigest()}")

    # Guess format if not provided
    if format is None:
        ext = os.path.splitext(file_path)[1].lower()
        format_map = {
            ".ttl": "turtle",
            ".nt": "nt",
            ".nq": "nquads",
            ".trig": "trig",
            ".jsonld": "json-ld",
            ".rdf": "xml",
            ".xml": "xml",
            ".n3": "n3",
        }
        format = format_map.get(ext, "turtle")

    # Use a fresh blank node each time (caller can still keep the identifier variable if needed)
    graph_uri = BNode()
    named_graph = store.get_context(graph_uri)
    try:
        logger.debug(f"LOAD: parsing {file_path} as {format}")
        named_graph.parse(file_path, format=str(format))
        logger.info(f"LOAD: graph {graph_uri} loaded with {len(named_graph)} triples from {file_path}")
    except Exception as e:
        logger.error(f"LOAD: error parsing {file_path}: {e}")
    return graph_uri

