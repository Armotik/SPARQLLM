
import hashlib
import warnings
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import XSD
from rdflib.plugins.sparql import prepareQuery
from rdflib.plugins.sparql.operators import register_custom_function

from string import Template
from rdflib import Graph, ConjunctiveGraph, URIRef, Literal, Namespace

from SPARQLLM.config import ConfigSingleton
from SPARQLLM.utils.utils import named_graph_exists, print_result_as_table
from SPARQLLM.udf.SPARQLLM import store

import logging
import time
logger = logging.getLogger(__name__)

from groq import Groq
import os

config = ConfigSingleton()
model = config.config['Requests']['SLM-GROQ-MODEL']

api_key = os.environ.get("GROQ_API_KEY", "default-api-key")
client = Groq(api_key=api_key)

def call_groq_api(client, model, prompt, max_retries=5):
    """Call GROQ while managing 429 error."""
    retry_delay = 1  # Délai initial en secondes
    time.sleep(retry_delay)
    for attempt in range(max_retries):
        try:
            chat_response = client.chat.completions.create(
                model=model,
                messages=[{ "role": "system",
                            "content": "You are a JSON-LD API. Always reply with a JSON-LD object using schema.org context. Do not include any Markdown formatting (like triple backticks) or explanations. Output only raw JSON."
                            },{
                            "role": "user", 
                           "content": prompt}]
            )
            return chat_response  # Succès, on retourne la réponse

        except Exception as e:
            if "429" in str(e) or "Rate limit" in str(e):
                logger.warning(f"Rate limit exceeded. Retrying in {retry_delay} seconds... (attempt {attempt+1}/{max_retries})")
                time.sleep(retry_delay)  # Attendre avant de réessayer
                retry_delay *= 2  # Exponential backoff
            else:
                logger.error(f"Error in calling  API: {e}")
                raise ValueError(f"Error in calling  API: {e}")

    raise ValueError("Max retries exceeded. GROQ API is still returning 429.")


def llm_graph_groq(prompt,uri):
    global store

    assert model != "", "GROQ Model not set in config.ini"
    if api_key == "default-api-key":
        raise ValueError("GROQ_API_KEY is not set. Using default value, which may not work for real API calls.")

    logger.debug(f"uri: {uri}, model: {model}, Prompt: {prompt[:50]} <...>")


    graph_name = prompt + ":"+str(uri)
    graph_uri = URIRef("http://groq.org/"+hashlib.sha256(graph_name.encode()).hexdigest())
    if  named_graph_exists(store, graph_uri):
        logger.debug(f"Graph {graph_uri} already exists (good)")
        return graph_uri
    else:
        named_graph = store.get_context(graph_uri)

    response = call_groq_api(client, model, prompt)
    # response = client.chat.completions.create(
    #     model=model,
    #     messages=[
    #         {
    #             "role": "user",
    #             "content": prompt
    #         }
    #     ],
    #     temperature=0.0
    # )
    logger.debug(f"Response: {response.choices[0].message.content}")
    graph_uri=URIRef(uri)
    named_graph = store.get_context(graph_uri)

    try:
        jsonld_data = response.choices[0].message.content
        named_graph.parse(data=jsonld_data, format="json-ld")

        #link new triple to bag of mappings
        insert_query_str = f"""
            INSERT  {{
                <{uri}> <http://example.org/has_schema_type> ?subject .}}
            WHERE {{
                ?subject a ?type .
            }}"""
        named_graph.update(insert_query_str)

        res=named_graph.query("""SELECT ?s ?o WHERE { ?s <http://example.org/has_schema_type> ?o }""")

    except Exception as e:
        logger.error(f"Error in parsing JSON-LD: {e}")
        named_graph.add((uri, URIRef("http://example.org/has_error"), Literal("Error {e}", datatype=XSD.string)))

    return graph_uri 



