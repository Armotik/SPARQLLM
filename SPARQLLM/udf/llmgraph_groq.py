
import hashlib
import warnings
from orjson import JSONDecodeError
import orjson
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
import os, re, json
from pyshacl import validate
from rdflib import BNode

config = ConfigSingleton()
model = config.config['Requests']['SLM-GROQ-MODEL']

api_key = os.environ.get("GROQ_API_KEY", "default-api-key")

client = Groq(api_key=api_key,max_retries=0,)

def supports_response_format(client, model: str) -> bool:
    """
        Returns:
            bool: True if `response_format={"type": "json_object"}` supported.
    """
    try:
        _ = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Reply with a valid JSON object."},
                {"role": "user", "content": "Say hello as JSON."}
            ],
            temperature=0,
            response_format={"type": "json_object"},
            max_tokens=50
        )
        return True
    except Exception as e:
        logger.debug(f"[GROK] Model '{model}' does not support response_format=json_object: {e}")
        return False

support_json=False
if supports_response_format(client, model):
    logger.debug(f"[GROQ] Model '{model}' supports response_format=json_object.")
    support_json=True
else:
    logger.debug(f"[GROQ] Model '{model}' does NOT support response_format=json_object.")
    support_json=False


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

def validate_jsonld_with_shacl(jsonld_str: str, shacl_path: str):
    """
    Valide un JSON-LD contre une forme SHACL.
    
    Args:
        jsonld_str (str): Le JSON-LD à valider (sous forme de chaîne).
        shacl_path (str): Chemin vers le fichier SHACL (.ttl).

    Returns:
        tuple: (conforms: bool, report_text: str, report_graph: rdflib.Graph)
    """

    try:
        data_graph = Graph().parse(data=jsonld_str, format='json-ld')
        shacl_graph = Graph().parse(shacl_path, format='turtle')

        conforms, report_graph, report_text = validate(
            data_graph,
            shacl_graph=shacl_graph,
            inference='rdfs',
            abort_on_first=False,
            meta_shacl=False,
            advanced=True,
            debug=False
        )

        if conforms:
            logging.debug("JSON-LD is valid against SHACL.")
        else:
            logging.warning("JSON-LD is NOT valid:\n" + report_text)

        return conforms, report_text, report_graph

    except Exception as e:
        logging.error(f"Validation failed: {e}")
        return False, str(e), None



def call_groq_api(client, model, messages, max_retries=5, max_wait=120):

    retry_delay = 1
    last_user_content = next(
       (m["content"] for m in reversed(messages) if m["role"] == "user"),
        None
    )
    prompt_hash = hashlib.sha256(last_user_content.encode('utf-8')).hexdigest()

    params = {
        "model": model,
        "messages": messages,
        "temperature": 0
    }
    if support_json:
        params["stream"]=False
        params["response_format"] = {"type": "json_object"}
        params["stop"]=None


    for attempt in range(1, max_retries + 1):
        try:
            chat_response = client.chat.completions.create(**params)

            headers = getattr(chat_response, 'response', {}).get('headers', {})
            tokens_left = int(headers.get("x-ratelimit-remaining-tokens", "9999"))
            if tokens_left < 300:
                logger.warning(f" Only {tokens_left} tokens left — consider pausing.")

            logger.debug(f"[GROQ] Success on attempt {attempt} — prompt hash {prompt_hash}")
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
                logger.debug(f"[GROQ] {prompt_hash} Waiting {wait}s before retry (attempt {attempt}/{max_retries})")
                time.sleep(wait)

            else:
                logger.error(f"[GROQ] API call failed: {e}")
                raise RuntimeError(f"Error calling GROQ API: {e}")

    raise RuntimeError("Max retries exceeded. GROQ API still rate-limited.")


def extract_json_ld(text):
    match = re.search(r'\{[\s\S]*?\}', text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            logger.warning("cannot find JSON in LLM output")
    return None



def llm_graph_groq(prompt, uri=None):
    return llm_graph_groq_model(prompt, uri, model)

def llm_graph_groq_model(prompt,uri,model):
    global store

    assert model != "", "GROQ Model not set in config.ini"
    if api_key == "default-api-key":
        raise RuntimeError("GROQ_API_KEY is not set. Using default value, which may not work for real API calls.")

    logger.debug(f"uri: {uri}, model: {model}, Prompt: {prompt[:50]} <...>")


    if uri is None:
        graph_uri = BNode()
    else:
        graph_name = prompt + ":"+str(uri)+':'+model
        graph_uri = URIRef("http://groq.org/"+hashlib.sha256(graph_name.encode()).hexdigest())
        if  named_graph_exists(store, graph_uri):
            logger.debug(f"Graph {graph_uri} already exists (good)")
            return graph_uri

    named_graph = store.get_context(graph_uri)

    named_graph.add((URIRef("http://example.org/GROK"), URIRef("http://example.org/param_prompt"), Literal(str(prompt))))
    named_graph.add((URIRef("http://example.org/GROK"), URIRef("http://example.org/param_model"), Literal(str(model))))
    named_graph.add((URIRef("http://example.org/GROK"), URIRef("http://example.org/param_uri"), Literal(str(uri))))


    # Conversation setup
    messages = [
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

    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:

            response = call_groq_api(client, model, messages)
            content = response.choices[0].message.content
#            logger.debug(f"Response (attempt {attempt}): {content}")
            ## grrr, when the prompt is not enough...
            match = re.search(r'\{[\s\S]*?\}', content)
            if match:
                content = match.group(0)
            else:
                messages.append({
                    "role": "assistant",
                    "content": f"{content}"
                })
                messages.append({
                    "role": "user",
                    "content": f"there is no JSON in the response. Please provide a valid JSON-LD object."  
                })
                continue

            # Essaye de parser le JSON-LD
            # named_graph.remove((None, None, None))
            named_graph.parse(data=content, format="json-ld")
            logger.debug(f"Successfully parsed JSON-LD: {content}")

# SHACL check
#             conforms, report_text, _ = validate_jsonld_with_shacl(content, "./data/event-shape.ttl")
#             if not conforms:
#                 logger.warning(f"JSON-LD does not conform to SHACL: {report_text}")
#                 messages.append({
#                     "role": "assistant",
#                     "content": f"{content}"
#                 })
#                 messages.append({
#                     "role": "user",
# #                    "content": f"The JSON-LD produced is not valid according to SHACL : {report_text}. I need 'StartDate' and 'name' properties."  
#                     "content": f"The JSON-LD produced is not valid. Sure of the uppercase/lowercase ??"  
#                 })
#                 continue
#             else:
#                 logger.debug(f"JSON-LD is valid against SHACL: {report_text}")

            # Requête SPARQL d'insertion
            insert_query_str = f"""
                INSERT {{
                    <{uri}> <http://example.org/has_schema_type> ?subject .
                }}
                WHERE {{
                    ?subject a ?type .
                }}"""
            named_graph.update(insert_query_str)
# working
            # for s, p, o in named_graph:
            #     print(f"Named: {s} {p} {o}")
            break  # Succès → on sort de la boucle
        
        except orjson.JSONDecodeError as e:
            logger.warning(f"Attempt {attempt}: JSON-LD parsing failed — {e}, for uri: {uri}")
            if attempt == max_attempts:
                logger.error("Maximum retry attempts reached. Aborting.")
                named_graph.add((uri, URIRef("http://example.org/has_error"), Literal(e, datatype=XSD.string)))
#                raise  # Relever l'exception pour la traiter plus haut si besoin
            else:
                messages.append({
                    "role": "assistant",
                    "content": f"{content}"
                })
                messages.append({
                    "role": "user",
                    "content": f"The previous JSON was invalid: {e}. Did you forget to close a bracket ?."
                })

        except Exception as e:
            logger.warning(f"Attempt {attempt}: JSON-LD parsing failed — {e}, for uri: {uri}")
            if attempt == max_attempts:
                error_message =f"Maximum retry attempts reached. Error in parsing JSON-LD: {e}"
                logger.error(error_message)
                named_graph.add((uri, URIRef("http://example.org/has_error"), Literal(error_message, datatype=XSD.string)))
#                raise  # Relever l'exception pour la traiter plus haut si besoin

    return graph_uri 



