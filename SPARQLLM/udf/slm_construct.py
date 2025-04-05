from rdflib import ConjunctiveGraph, Graph, URIRef
from rdflib.plugins.sparql.operators import register_custom_function
from rdflib.term import Literal
from rdflib.namespace import XSD
from SPARQLLM.udf.SPARQLLM import store
from SPARQLLM.utils.utils import print_result_as_table, named_graph_exists

import logging
logger = logging.getLogger(__name__)

def slm_construct(triples_str, graph_uri, gin):
    """
    Not really a construct, 
    overwrite gin with the triples_str in a new named graph graph_uri.    
    """
    logger.debug(f"str: {triples_str}, graph_uri: {graph_uri}, graph_in:{gin}")    

    if  named_graph_exists(store, graph_uri):
        logger.debug(f"Graph {graph_uri} already exists (good)")
        return None
    else:
        named_graph = store.get_context(graph_uri)


    try:
        # copy gin in new named graph...
        for triple in store.graph(gin):
            named_graph.add(triple)
        # and overwrite it with the new triples
        tmp_graph=Graph()
        tmp_graph.parse(data=triples_str, format="nt")  # assumes N-Triples-style input
        for s, p, o in tmp_graph:
            logger.debug(f"examining {s} {p} {o}")
            for _, _, o2 in named_graph.triples((s, p, None)):
                if o2 != o:
                    logger.debug(f"replace {s} {p} {o2} by {o}")
                    # Supprimer l'ancien
                    named_graph.remove((s, p, o2))
                    # Ajouter le nouveau
                    named_graph.add((s, p, o))
        for triple in named_graph:
            logger.debug(f"triple: {triple}")
        return URIRef(graph_uri)
    except Exception as e:
        # Debug: return error as string literal
        logger.debug(f"error:{e}")

