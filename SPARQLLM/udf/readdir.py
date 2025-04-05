from rdflib import BNode, Graph, Literal, URIRef
from rdflib.namespace import XSD
from rdflib.plugins.sparql import prepareQuery
from rdflib.plugins.sparql.operators import register_custom_function

from string import Template
from urllib.parse import urlencode,quote
from urllib.request import Request, urlopen

from SPARQLLM.udf.SPARQLLM import store
from SPARQLLM.utils.utils import named_graph_exists


import os
import json


import logging
logger = logging.getLogger(__name__)

from pathlib import Path

def gettype(path):
    entry = Path(path)
    if entry.is_file():
        return Literal('file')
    elif entry.is_dir():
        return Literal('directory')
    elif entry.is_symlink():
        return Literal('symlink')
    else:
        return Literal('unknown')

def RDIR(dir,link_to):
    logger.debug(f"dir:{dir}, link_to:{link_to}")    

    if not os.path.isdir(dir[7:]):
        logger.debug(f"dir:{dir} is not a directory")
        return None

    graph_uri=URIRef(dir)
    if  named_graph_exists(store, graph_uri):
        logger.debug(f"Graph {graph_uri} already exists (good)")
        return None
    else:
        named_graph = store.get_context(graph_uri)

    try:
        ## quite brutal... remove the file://
        local_dir=str(dir)[7:]
        logger.debug(f"dir:{dir},local_dir:{local_dir}")
   
        files = os.listdir(local_dir)
        for file in files:
#            logging.debug(f"RDIR : found {file}, in {local_dir}")
            path=URIRef("file://"+os.path.join(local_dir, file))
            size=os.path.getsize(os.path.join(local_dir, file))
#            logging.debug(f"RDIR: path {path}, type {type(path)}")
            named_graph.add((link_to, URIRef("http://example.org/has_path"), path))
            named_graph.add((path, URIRef("http://example.org/has_size"), Literal(size, datatype=XSD.integer)))
            named_graph.add((path, URIRef("http://example.org/has_type"), gettype(os.path.join(local_dir, file))))  
        logger.debug(f"graph:{graph_uri}, len:{len(named_graph)}")
#        for s, p, o in named_graph:
#            print(f"Subject: {s}, Predicate: {p}, Object: {o}")

    except Exception as e:
        raise ValueError("RDIR Error : "+str(e))
    
    #don't forget
    return graph_uri


