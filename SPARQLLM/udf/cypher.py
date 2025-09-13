from rdflib import URIRef, BNode, Literal, Namespace, RDF, XSD
from typing import Dict, List, Tuple, Optional
import re

from SPARQLLM.udf.SPARQLLM import store

import logging
logger = logging.getLogger(__name__)


SLM = Namespace("http://sparqllm/slm#")

# NOTE: On suppose qu'un objet 'store' (ConjunctiveGraph/Dataset) est disponible
# comme dans vos autres UDF (cf. recurse.py). Sinon, adaptez pour le récupérer.

def _parse_prefixes(pref_str: Optional[str]) -> Dict[str, str]:
    if not pref_str:
        return {}
    d = {}
    for part in pref_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        d[k.strip()] = v.strip()
    return d

def _resolve_pred(token: str, prefixes: Dict[str, str]) -> URIRef:
    token = token.strip()
    if token.startswith("<") and token.endswith(">"):
        return URIRef(token[1:-1])
    if ":" in token:
        pref, local = token.split(":", 1)
        base = prefixes.get(pref)
        if not base:
            # on laisse tel quel si c’est déjà un IRI complet accolé (peu probable)
            raise ValueError(f"Préfixe inconnu pour le prédicat: {token}")
        return URIRef(base + local)
    raise ValueError(f"Prédicat non résolu: {token}")
def _parse_chain_to_triples(chain: str, prefixes: Dict[str, str]) -> List[Tuple[str, URIRef, str]]:
    # Supporte: (a)-[:ex:p]->(b), (a)<-[:ex:p]-(b), (a)-[:ex:p]-(b)
    s = 0
    chain = chain.strip()

    def ws():
        nonlocal s
        while s < len(chain) and chain[s].isspace():
            s += 1

    def expect(tok: str):
        nonlocal s
        ws()
        if not chain.startswith(tok, s):
            raise ValueError(f"Attendu '{tok}' dans {chain!r} à {s}")
        s += len(tok)

    def read_var() -> str:
        nonlocal s
        ws(); expect("("); ws()
        m = re.match(r"\w+", chain[s:])
        if not m:
            raise ValueError("Variable attendue dans (var)")
        var = m.group(0); s += len(var); ws(); expect(")")
        return var

    def read_arrow_and_pred() -> Tuple[str, URIRef, str]:
        # Supporte: -[:pred]->  |  <-[:pred]-  |  -[:pred]-
        nonlocal s
        ws()
        if chain.startswith("<-", s):
            s += 2
            ws(); expect("["); ws(); expect(":")
            m = re.match(r"[A-Za-z_]\w*:[A-Za-z_][\w.-]*|<[^>]+>", chain[s:])
            if not m:
                raise ValueError(f"Prédicat attendu après ':' ou <IRI> à {s} dans {chain!r}")
            pred_tok = m.group(0); s += len(pred_tok); ws(); expect("]")
            ws(); expect("-")
            pred = _resolve_pred(pred_tok, prefixes)
            return ("<-", pred, "")
        else:
            expect("-")
            ws(); expect("["); ws(); expect(":")
            m = re.match(r"[A-Za-z_]\w*:[A-Za-z_][\w.-]*|<[^>]+>", chain[s:])
            if not m:
                raise ValueError(f"Prédicat attendu après ':' ou <IRI> à {s} dans {chain!r}")
            pred_tok = m.group(0); s += len(pred_tok); ws(); expect("]")
            ws()
            if chain.startswith("->", s):
                s += 2
                pred = _resolve_pred(pred_tok, prefixes)
                return ("->", pred, "")
            elif chain.startswith("-", s):
                s += 1
                pred = _resolve_pred(pred_tok, prefixes)
                return ("-", pred, "")
            else:
                raise ValueError(f"Direction attendue après [:pred] à {s} (-> ou -) dans {chain!r}")

    triples: List[Tuple[str, URIRef, str]] = []
    left = read_var()
    while True:
        ws()
        if s >= len(chain):
            break
        arrow, pred, _ = read_arrow_and_pred()
        right = read_var()
        if arrow in ("->", "-"):
            triples.append((left, pred, right))
        else:  # "<-"
            triples.append((right, pred, left))
        left = right
    return triples


def _parse_where(where_str: str) -> List[Tuple[str, str]]:
    # WHERE a = ex:a AND b = <...> AND c = "literal"
    if not where_str:
        return []
    conds = []
    for cond in [c.strip() for c in where_str.split("AND")]:
        m = re.match(r"^(?P<v>\w+)\s*=\s*(?P<const>.+)$", cond)
        if not m:
            raise ValueError(f"WHERE non supporté: {cond}")
        conds.append((m.group("v"), m.group("const").strip()))
    return conds

def _resolve_const(const: str, prefixes: Dict[str, str]):
    const = const.strip()
    if const.startswith("<") and const.endswith(">"):
        return URIRef(const[1:-1])
    if const.startswith('"') and const.endswith('"'):
        return Literal(const[1:-1])
    if ":" in const:
        pref, local = const.split(":", 1)
        base = prefixes.get(pref)
        if not base:
            raise ValueError(f"Préfixe inconnu dans WHERE: {const}")
        return URIRef(base + local)
    # fallback littéral brut
    return Literal(const)

def _parse_cypher(cypher: str, prefixes: Dict[str, str]):
    # MATCH ... [WHERE ...] RETURN ...
    m = re.match(r"(?is)^MATCH\s+(?P<pat>.+?)\s+(WHERE\s+(?P<where>.+?)\s+)?RETURN\s+(?P<ret>.+)$", cypher.strip())
    if not m:
        raise ValueError("Cypher minimal attendu: MATCH ... [WHERE ...] RETURN ...")
    pat_str = m.group("pat").strip()
    where_str = (m.group("where") or "").strip()
    ret_str = m.group("ret").strip()

    triples: List[Tuple[str, URIRef, str]] = []
    for p in [p.strip() for p in pat_str.split(",") if p.strip()]:
        triples.extend(_parse_chain_to_triples(p, prefixes))

    where_conds = _parse_where(where_str)

    ret_vars = [v.strip() for v in ret_str.split(",") if v.strip()]
    for v in ret_vars:
        if not re.match(r"^\w+$", v):
            raise ValueError(f"Nom de variable invalide dans RETURN: {v}")
    return triples, where_conds, ret_vars

def _backtrack_join(gctx, triples: List[Tuple[str, URIRef, str]], where_conds, prefixes: Dict[str, str]):
    # Exécute un JOIN par backtracking sur gctx
    # Bindings: dict var -> node
    equals = [(v, _resolve_const(c, prefixes)) for v, c in where_conds]

    order = list(range(len(triples)))  # tout simple; on peut heuristiquement trier par sélectivité
    bindings_list = []

    def consistent(b):
        for v, c in equals:
            if v in b and b[v] != c:
                return False
        return True

    def extend(i, b):
        if i == len(order):
            if consistent(b):
                bindings_list.append(dict(b))
            return
        s_var, pred, o_var = triples[order[i]]
        s_val = b.get(s_var)
        o_val = b.get(o_var)

        # Itère sur triples RDFLib
        if s_val is not None and o_val is not None:
            if (s_val, pred, o_val) in gctx:
                extend(i + 1, b)
            return

        if s_val is not None and o_val is None:
            for _, _, o in gctx.triples((s_val, pred, None)):
                if any(v == o_var and c != o for v, c in equals):
                    continue
                if o_var in b and b[o_var] != o:
                    continue
                b[o_var] = o
                if consistent(b):
                    extend(i + 1, b)
                del b[o_var]
            return

        if s_val is None and o_val is not None:
            for s, _, _ in gctx.triples((None, pred, o_val)):
                if any(v == s_var and c != s for v, c in equals):
                    continue
                if s_var in b and b[s_var] != s:
                    continue
                b[s_var] = s
                if consistent(b):
                    extend(i + 1, b)
                del b[s_var]
            return

        # aucun lié
        for s, _, o in gctx.triples((None, pred, None)):
            sv_ok = all(not (v == s_var and c != s) for v, c in equals)
            ov_ok = all(not (v == o_var and c != o) for v, c in equals)
            if not (sv_ok and ov_ok):
                continue
            prev_s = b.get(s_var); prev_o = b.get(o_var)
            if prev_s is not None and prev_s != s:
                continue
            if prev_o is not None and prev_o != o:
                continue
            b[s_var] = s; b[o_var] = o
            if consistent(b):
                extend(i + 1, b)
            if prev_s is None: del b[s_var]
            if prev_o is None: del b[o_var]

    extend(0, {})
    return bindings_list

def slm_cypher(ginit, cypher_query: str, prefixes: str = None):
    """
    GGF: SLM-CYPHER(ginit, cypher, prefixes) -> graphe nommé avec lignes slm:Row
    prefixes: "ex=http://ex.org/,schema=https://schema.org/"
    """
    pf = _parse_prefixes(prefixes)
    gctx = store.graph(ginit)

    triples, where_conds, ret_vars = _parse_cypher(cypher_query, pf)
    rows = _backtrack_join(gctx, triples, where_conds, pf)

    out_id = BNode()
    gout = store.graph(out_id)

    path_node = BNode()
    gout.add((path_node, RDF.type, SLM.ResultSet))
    gout.add((path_node, SLM.column, Literal(",".join(ret_vars))))

    for i, row in enumerate(rows, start=1):
        r = BNode()
        gout.add((r, RDF.type, SLM.Row))
        gout.add((r, SLM.pos, Literal(i, datatype=XSD.integer)))
        for v in ret_vars:
            b = BNode()
            gout.add((b, RDF.type, SLM.Binding))
            gout.add((b, SLM.name, Literal(v)))
            val = row.get(v)
            if val is not None:
                gout.add((b, SLM.value, val))
            gout.add((r, SLM.hasBinding, b))
        gout.add((path_node, SLM.hasRow, r))

    return out_id