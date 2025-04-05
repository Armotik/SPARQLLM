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

def recurse(query_str,ginit,max_depth_lit=Literal(10)):
    logger.debug(f"query:{query_str},  ginit:{ginit.n3()}, max_depth:{max_depth_lit}")
    max_depth = max_depth_lit.value

    def func_recurse_on(gin_rec,depth=0):
        logger.debug(f"Recurse on:{gin_rec}, size:{len(store.graph(gin_rec))}, depth:{depth}")
#        for triple in store.graph(gin_rec):
#            logger.debug(f"gin triple: {triple}")

        # query the graph 
        # If we want to use graphs in the construct query -> need a DS. (not a graph)      
#        result = store.graph(gin_rec).query(query_str)
#        result = store.query(query_str) # dangerous -> all graphs
        result = store.query(query_str,initBindings={'?gin':gin_rec})
        if (result.type!="CONSTRUCT"): 
            logger.debug(f"recursive query is not a construct: {result}")
            raise ValueError("result is not a graph : not a construct query")
        if len(result) == 0:
            logger.debug(f"result is empty")
            return gin_rec
        else:
            gout=store.get_context(BNode())
#            logger.debug(f"result: {result.graph} : {type(result.graph)}")
            for s,p,o in result.graph:
                gout.add((s,p,o))
#            logger.debug(f"len(gin):{len(gin_rec)}, {len(store.graph(gin_rec))} len(gout):{len(gout)} ")
            if len(gout) == len(store.graph(gin_rec)):
                logger.debug(f"stop condition : {len(gout)} == {len(store.graph(gin_rec))}")
                return gout.identifier
            else:
#                logger.debug(f"depth {depth}, {type(depth)}, max_depth {max_depth} {type(max_depth)}")
                if depth <= max_depth:
                    return func_recurse_on(gout.identifier,depth+1)
                else:
                    logger.debug(f"max depth :  {max_depth} reached")
                    return gout.identifier

    try:
        g_output = func_recurse_on(ginit,0)
    except Exception as e:
        traceback.print_exc()
        raise ValueError("Recurse Error : "+str(e))

    logger.debug(f"RECURSE end: graph {g_output} has {len(store.graph(g_output))} triples")
#    for triple in g_output:
#        logger.debug(f"recurse end triple: {triple}")
    return g_output

