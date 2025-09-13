from rdflib import URIRef, BNode, Literal, Namespace, RDF, XSD
from collections import deque
from SPARQLLM.udf.SPARQLLM import store

SLM = Namespace("http://sparqllm/slm#")

def slm_bfs(start, goal, pred, ginit):
    """
    SLM-BFS(start, goal, pred, ginit) -> graph (named) contenant un chemin ordonné s -> ... -> t.
    Sortie (dans le graph retourné):
      _:path a slm:Path ; slm:start <s> ; slm:goal <t> ; slm:predicate <p> ; slm:length N ; slm:hasStep _:s_i .
      _:s_i a slm:PathStep ; slm:pos i ; slm:src <u> ; slm:dst <v> ; slm:pred <p> .
    """
    # Coercition
    def to_iri(x):
        if isinstance(x, URIRef): return x
        try:
            return URIRef(str(x))
        except Exception:
            return URIRef(str(x))
    s = to_iri(start); t = to_iri(goal); p = to_iri(pred)

    # Graphe d’entrée (contexte)
    gctx = store.graph(ginit)

    # BFS (plus court chemin en nombre d’arêtes)
    prev = {s: None}
    q = deque([s])
    found = False
    while q and not found:
        u = q.popleft()
        for _, _, v in gctx.triples((u, p, None)):
            if v not in prev:
                prev[v] = u
                if v == t:
                    found = True
                    break
                q.append(v)

    # Graphe de sortie
    out_id = BNode()
    gout = store.graph(out_id)

    path_node = BNode()
    gout.add((path_node, RDF.type, SLM.Path))
    gout.add((path_node, SLM.start, s))
    gout.add((path_node, SLM.goal, t))
    gout.add((path_node, SLM.predicate, p))

    if not found:
        gout.add((path_node, SLM.length, Literal(0, datatype=XSD.integer)))
        return out_id

    # Reconstruire s -> ... -> t
    path = []
    v = t
    while v is not None:
        path.append(v)
        v = prev[v]
    path.reverse()

    gout.add((path_node, SLM.length, Literal(len(path) - 1, datatype=XSD.integer)))

    # Étapes ordonnées
    for i in range(1, len(path)):
        step = BNode()
        gout.add((step, RDF.type, SLM.PathStep))
        gout.add((step, SLM.pos, Literal(i, datatype=XSD.integer)))
        gout.add((step, SLM.src, path[i - 1]))
        gout.add((step, SLM.dst, path[i]))
        gout.add((step, SLM.pred, p))
        gout.add((path_node, SLM.hasStep, step))

    return out_id
# ...existing code...