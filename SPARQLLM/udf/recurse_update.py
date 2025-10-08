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


        gin=store.graph(gin_rec)
        # Clone of initial initial
        graph_before = Graph()
        for triple in gin:
            graph_before.add(triple)

        #store.update(query_str,initBindings={'?gin':gin_rec})
        store.graph(gin_rec).update(query_str)

        inserted = set(gin) - set(graph_before)
        deleted = set(graph_before) - set(gin)

        # nothing inserted/deleted -> end of recurse
        if len(inserted)==len(deleted)==0:
            logger.debug(f"no update")
            return gin_rec

        logger.debug(f"inserted:{len(inserted)}, deleted:{len(deleted)}")
        for triple in inserted:
            logger.debug(f"  + {triple[0].n3()} {triple[1].n3()} {triple[2].n3()} .")
        for triple in deleted:
            logger.debug(f"  - {triple[0].n3()} {triple[1].n3()} {triple[2].n3()} .")

         # recurse if depth not exceeded

        for triple in gin.triples((None,None,None)):
            logger.debug(f" state before rec {triple[0].n3()} {triple[1].n3()} {triple[2].n3()} .")

        if depth <= max_depth:
            return func_recurse_on(gin_rec,depth+1)
        else:
            logger.debug(f"max depth :  {max_depth} reached")
            return gin_rec

    try:
        g_output = func_recurse_on(ginit,0)
    except Exception as e:
        traceback.print_exc()
        raise ValueError("Recurse Error : "+str(e))

    logger.debug(f"RECURSE end: graph {g_output} has {len(store.graph(g_output))} triples")
    # for t in store.graph(g_output).triples((None,None,None)):
    #     logger.debug(f"  {t[0].n3()} {t[1].n3()} {t[2].n3()} .")
    return g_output

