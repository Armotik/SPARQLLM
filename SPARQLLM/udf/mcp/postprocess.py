# post-traitement: JSON/JSON-LD -> graphe RDF nommé
import json, hashlib, datetime as dt
from rdflib import Graph

PROFILE_MAPPERS = {}         # tool_name -> fn(json)->jsonld
def register_mapper(tool, fn): PROFILE_MAPPERS[tool]=fn

def _heuristic_json_to_jsonld(data):
    def to_item(o):
        if not isinstance(o, dict): return None
        iid = o.get("html_url") or o.get("url") or o.get("id")
        item = {"@type":"Thing", **{k:v for k,v in o.items() if v is not None}}
        if iid: item["@id"]=str(iid)
        return item
    items=[]
    if isinstance(data, list): items=[x for x in map(to_item,data) if x]
    elif isinstance(data, dict):
        arr = next((v for v in data.values() if isinstance(v,list) and v and isinstance(v[0],dict)), None)
        items=[x for x in map(to_item, arr or [data]) if x]
    return {"@context":"https://schema.org/","@graph":items}

def _prov_wrap(jsonld, tool):
    if isinstance(jsonld, dict):
        ctx = jsonld.setdefault("@context", {"@vocab":"https://schema.org/","prov":"http://www.w3.org/ns/prov#"})
        if isinstance(ctx, dict): ctx.setdefault("prov","http://www.w3.org/ns/prov#")
        g = jsonld.setdefault("@graph", [])
        g.append({
          "@id": f"urn:prov:{hashlib.sha1(tool.encode()).hexdigest()[:8]}",
          "@type": "prov:Activity",
          "prov:wasGeneratedBy": f"urn:mcp:tool:{tool}",
          "prov:generatedAtTime": dt.datetime.utcnow().isoformat()+"Z"
        })
    return jsonld

def _materialize(engine, jsonld_obj, name=None):
    s = json.dumps(jsonld_obj) if isinstance(jsonld_obj, dict) else jsonld_obj
    g = Graph(); g.parse(data=s, format="json-ld")
    iri = name or f"urn:mcp:graph:{hashlib.sha256(s.encode()).hexdigest()[:16]}"
    engine.current_dataset.add_named_graph(iri, g)  # adapte à ton dataset
    return iri

def postprocess_to_rdf(engine, tool_name, result):
    # 1) JSON-LD natif
    if isinstance(result, dict) and result.get("media_type")=="application/ld+json":
        jl = _prov_wrap(result.get("jsonld"), tool_name)
        return {"ok": True, "graph": _materialize(engine, jl, result.get("graph_name"))}
    # 2) Mapper dédié
    mapper = PROFILE_MAPPERS.get(tool_name)
    if mapper and isinstance(result, (dict,list)):
        jl = _prov_wrap(mapper(result), tool_name)
        return {"ok": True, "graph": _materialize(engine, jl)}
    # 3) Heuristique générique
    if isinstance(result, (dict,list)):
        jl = _prov_wrap(_heuristic_json_to_jsonld(result), tool_name)
        return {"ok": True, "graph": _materialize(engine, jl)}
    # 4) Fallback
    return {"ok": True, "data": result}
