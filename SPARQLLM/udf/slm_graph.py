from rdflib import ConjunctiveGraph, Graph, URIRef,BNode
from rdflib.plugins.sparql.operators import register_custom_function
from rdflib.term import Literal
from rdflib.namespace import XSD
from SPARQLLM.udf.SPARQLLM import store
from SPARQLLM.utils.utils import print_result_as_table, named_graph_exists

import logging
logger = logging.getLogger(__name__)

def slm_graph(triples_str, graph_uri=None):
    if graph_uri is None:
        graph_uri = BNode()
    logger.debug(f"str: {triples_str}, graph_uri: {graph_uri}")    

    if  named_graph_exists(store, graph_uri):
        logger.debug(f"Graph {graph_uri} already exists : appending")
    named_graph = store.get_context(graph_uri)

    try:
        named_graph.parse(data=triples_str, format="turtle")  # assumes N-Triples-style input
#        for triple in named_graph:
#            logger.debug(f"Graph({graph_uri}): {triple}")
        return graph_uri
    except Exception as e:
        # Debug: return error as string literal
        logger.debug(f"error:{e}")

