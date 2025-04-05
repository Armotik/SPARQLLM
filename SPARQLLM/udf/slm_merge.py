from rdflib import ConjunctiveGraph, Graph, URIRef,BNode
from rdflib.plugins.sparql.operators import register_custom_function
from rdflib.term import Literal
from rdflib.namespace import XSD
from SPARQLLM.udf.SPARQLLM import store
from SPARQLLM.utils.utils import print_result_as_table, named_graph_exists

import logging
logger = logging.getLogger(__name__)

def slm_merge(g1, g2, gout=None):
    if gout is None:
        gout = BNode()
    logger.debug(f"g1: {g1}, g2: {g2}, len(g2):{len(g2)}, gout:{gout}")    
    if  named_graph_exists(store, gout):
        logger.debug(f"Graph {gout} already exists : append")
    named_graph = store.get_context(gout)

    # check if g2 is empty -> return empty
    # (for recursion)
    if len(g2)  == 0:
        logger.debug(f"Graph {g2} is empty")
        return g2
    try:
        # copy gin in new named graph...
        for triple in store.graph(g1):
            named_graph.add(triple)
        for triple in store.graph(g2):
            named_graph.add(triple)

        for triple in named_graph:
            logger.debug(f"triple: {triple}")
        return gout
    except Exception as e:
        # Debug: return error as string literal
        logger.debug(f"error:{e}")

