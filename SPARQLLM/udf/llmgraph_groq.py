
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
client = Groq(api_key=api_key,max_retries=0,)

def parse_reset_duration(duration_str):
    """
    Convert strings like '16m10.288s' or '57.912s' to seconds (float).
    """
    match = re.match(r"(?:(\d+)m)?([\d.]+)s", duration_str)
    if not match:
        return 60  # fallback
    minutes = int(match.group(1)) if match.group(1) else 0
    seconds = float(match.group(2))
    return minutes * 60 + seconds

def call_groq_api(client, model, prompt, max_retries=5, max_wait=120):
    import hashlib

    retry_delay = 1
    prompt_hash = hashlib.sha256(prompt.encode('utf-8')).hexdigest()

    for attempt in range(1, max_retries + 1):
        try:
            chat_response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a JSON-LD API. Always reply with a JSON-LD object "
                            "using schema.org context. Do not include any Markdown formatting "
                            "(like triple backticks) or explanations. Output only raw JSON."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            headers = getattr(chat_response, 'response', {}).get('headers', {})
            tokens_left = int(headers.get("x-ratelimit-remaining-tokens", "9999"))
            if tokens_left < 300:
                logger.warning(f"⚠️ Only {tokens_left} tokens left — consider pausing.")

            logger.info(f"[GROQ] ✅ Success on attempt {attempt} — prompt hash {prompt_hash}")
            return chat_response

        except Exception as e:
            is_429 = "429" in str(e) or "Too Many Requests" in str(e)
            response = getattr(e, 'response', None)
            headers = getattr(response, 'headers', {}) if response else {}

            if is_429:
                wait = None

                if "retry-after" in headers:
                    try:
                        wait = int(float(headers["retry-after"]))
                        logger.warning(f"[GROQ] Retry-After header: waiting {wait}s")
                    except ValueError:
                        pass

                elif "x-ratelimit-reset-tokens" in headers:
                    wait = parse_reset_duration(headers["x-ratelimit-reset-tokens"])
                    logger.warning(f"[GROQ] Token limit: reset in {wait:.1f}s")

                elif "x-ratelimit-reset-requests" in headers:
                    wait = parse_reset_duration(headers["x-ratelimit-reset-requests"])
                    logger.warning(f"[GROQ] Request limit: reset in {wait:.1f}s")

                if wait is None:
                    wait = retry_delay
                    retry_delay = min(retry_delay * 2, max_wait)
                    logger.warning(f"[GROQ] No wait header found. Exponential backoff to {retry_delay}s")

                wait = min(wait, max_wait)
                logger.info(f"[GROQ] {prompt_hash} Waiting {wait}s before retry (attempt {attempt}/{max_retries})")
                time.sleep(wait)

            else:
                logger.error(f"[GROQ] API call failed: {e}")
                raise RuntimeError(f"Error calling GROQ API: {e}")

    raise RuntimeError("Max retries exceeded. GROQ API still rate-limited.")




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
    logger.debug(f"Response: {response.choices[0].message.content}")
    graph_uri=URIRef(uri)
    named_graph = store.get_context(graph_uri)

    try:
        jsonld_data = response.choices[0].message.content
        named_graph.parse(data=jsonld_data, format="json-ld")
        logger.info(f"reading: {jsonld_data}")


        #link new triple to bag of mappings
        insert_query_str = f"""
            INSERT  {{
                <{uri}> <http://example.org/has_schema_type> ?subject .}}
            WHERE {{
                ?subject a ?type .
            }}"""
        named_graph.update(insert_query_str)
#        logger.info(f"Inserted new triples into graph:{len(named_graph)}")

    except Exception as e:
#        logger.error(f"Error in parsing JSON-LD: {jsonld_data}")
        logger.error(f"Error in parsing JSON-LD: {e}")
        named_graph.add((uri, URIRef("http://example.org/has_error"), Literal("Error {e}", datatype=XSD.string)))

    return graph_uri 



