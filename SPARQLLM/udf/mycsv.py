from rdflib import Graph, Literal, URIRef
from rdflib.namespace import XSD
from rdflib.plugins.sparql import prepareQuery
from rdflib.plugins.sparql.operators import register_custom_function

from string import Template
from urllib.parse import urlencode, quote
from urllib.request import Request, urlopen

from urllib.parse import urlparse, unquote

from SPARQLLM.udf.SPARQLLM import store
from SPARQLLM.utils.utils import print_result_as_table, named_graph_exists

import os
import json

import pandas as pd
from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDF, XSD

import traceback

import logging

logger = logging.getLogger(__name__)


def slm_csv(file_url, mappings_url=None):
    logger.debug(f"slm_csv called with file: {file_url}, mappings: {mappings_url}")
    try:

        file_path = unquote(urlparse(str(file_url)).path)
        logger.debug(f"Resolved file path: {file_path}")

        mappings_path = None
        mapping_name = 'default'
        if mappings_url is not None:
            mappings_path = unquote(urlparse(str(mappings_url)).path)
            logger.debug(f"Resolved mappings path: {mappings_path}")
            mapping_name = os.path.basename(mappings_path)

        graph_uri_str = f"{file_url}#{mapping_name}"
        graph_uri = URIRef(graph_uri_str)

        if named_graph_exists(store, graph_uri):
            logger.debug(f"Graph {graph_uri} already exists (good)")
            return graph_uri  # Return the existing graph URI if it exists
        else:
            named_graph = store.get_context(graph_uri)

        df = pd.read_csv(file_path)

        # If no mappings are provided, create a default mapping
        if mappings_url is None:

            logger.debug("No mappings provided, using default CSV to RDF conversion.")
            n = Namespace("http://example.org/")

            # Define a generic class for the CSV records
            Record = URIRef(n.Record)

            # Create properties for each column
            properties = {col: URIRef(n[col.replace(' ', '_').strip()]) for col in df.columns}

            for index, row in df.iterrows():
                record_uri = URIRef(n[f"record_{index}"])
                named_graph.add((record_uri, RDF.type, Record))

                for col, value in row.items():
                    if pd.notna(value):
                        prop = properties[col]
                        if isinstance(value, int):
                            datatype = XSD.integer
                        elif isinstance(value, float):
                            datatype = XSD.float
                        else:
                            datatype = XSD.string
                        named_graph.add((record_uri, prop, Literal(value, datatype=datatype)))

            logger.debug(f"Default graph {graph_uri} created with {len(named_graph)} triples.")
            return graph_uri

        # If mappings are provided, apply the CONSTRUCT query
        else:

            logger.debug(f"Mappings file provided: {mappings_url}. Applying CONSTRUCT query.")

            temp_graph = Graph()
            n = Namespace("http://example.org/")  # Espace de nom par défaut pour le CSV brut
            Record = URIRef(n.Record)

            properties = {col: URIRef(n[col.replace(' ', '_').strip()]) for col in df.columns}

            for index, row in df.iterrows():
                record_uri = URIRef(n[f"record_{index}"])
                temp_graph.add((record_uri, RDF.type, Record))
                for col, value in row.items():
                    if pd.notna(value):
                        prop = properties[col.replace(' ', '_').strip()]
                        if isinstance(value, int):
                            datatype = XSD.integer
                        elif isinstance(value, float):
                            datatype = XSD.float
                        else:
                            datatype = XSD.string
                        temp_graph.add((record_uri, prop, Literal(value, datatype=datatype)))

            logger.debug(f"Temporary graph created with {len(temp_graph)} triples.")

            # Load the CONSTRUCT query from the mappings file
            try:
                with open(mappings_path, 'r') as f:
                    construct_query_str = f.read()
            except Exception as e:
                logger.error(f"Could not read mappings file {mappings_url}: {e}")
                traceback.print_exc()
                return None

            logger.debug(f"Mapping query loaded:\n{construct_query_str}")

            try:
                # Apply the CONSTRUCT query to the temporary graph
                init_ns = {
                    "ex": Namespace("http://example.org/"),
                    "mycsv": Namespace("http://mycsv.org/"),
                    "rdf": RDF,
                    "xsd": XSD
                }

                # Prepare and execute the CONSTRUCT query
                result_graph = temp_graph.query(construct_query_str, initNs=init_ns)

                # Add the resulting triples to the named graph
                for triple in result_graph:
                    named_graph.add(triple)

                logger.debug(f"Mapped graph {graph_uri} created with {len(named_graph)} triples.")
                return graph_uri

            except Exception as e:
                logger.error(f"Error applying CONSTRUCT query: {e}")
                traceback.print_exc()
                return None

    except Exception as e:
        logger.error(f"Error reading file: {e}")
        traceback.print_exc()
        return None