from rdflib import BNode, Graph, Literal, URIRef
from rdflib.namespace import XSD
from rdflib.plugins.sparql import prepareQuery
from rdflib.plugins.sparql.operators import register_custom_function

from string import Template
from urllib.parse import urlencode,quote
from urllib.request import Request, urlopen

from SPARQLLM.udf.SPARQLLM import store
from SPARQLLM.utils.utils import named_graph_exists
from SPARQLLM.udf.readdir import gettype, RDIR

import os
import json

import logging
import traceback

from pathlib import Path

logger = logging.getLogger(__name__)

def recurse_update(query_str,ginit,max_depth_lit=Literal(10)):
    logger.debug(f"query:{query_str},  ginit:{ginit.n3()}, max_depth:{max_depth_lit}")
    max_depth = max_depth_lit.value

    def func_recurse_on(gin_rec,depth=0):
        logger.debug(f"Recurse on:{gin_rec}, size:{len(store.graph(gin_rec))}, depth:{depth}")
        if depth <= max_depth:
            return func_recurse_on(store.graph(gin_rec),depth+1)
        else:
            logger.debug(f"max depth :  {max_depth} reached")
            return gin_rec

    try:
        g_output = func_recurse_on(ginit,0)
    except Exception as e:
        traceback.print_exc()
        raise ValueError("Recurse Error : "+str(e))

    logger.debug(f"RECURSE end: graph {g_output.identifier} has {len(store.graph(g_output))} triples")
#    for triple in g_output:
#        logger.debug(f"recurse end triple: {triple}")
    return g_output.identifier

