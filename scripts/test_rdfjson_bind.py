#!/usr/bin/env python3
"""Minimal test of a SPARQL BIND using an rdf:JSON typed literal with rdflib.

Note: rdflib (7.1.3) does not implement SPARQL 1.2 JSON functions, but it will
still parse and bind a literal whose datatype IRI is rdf:JSON.

Expected outcome:
- Query runs successfully.
- Variables ?pr ?title ?time remain unbound (None) because no triple patterns produce them.
- ?params is bound inside the pattern but we did not project it; we add a second
  query variant that selects it to show the value.
"""
from rdflib import Graph

QUERY_ORIGINAL = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
SELECT ?pr ?title ?time WHERE {
    BIND(\"\"\"{ 'owner' : 'openwrt', 'repo' : 'openwrt', 'state' : 'open', 'limit' : 10 }\"\"\"^^rdf:JSON as ?params)
}
"""

QUERY_SHOW_PARAMS = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
SELECT ?params WHERE {
  BIND(\"\"\"{ 'owner' : 'openwrt', 'repo' : 'openwrt', 'state' : 'open', 'limit' : 10 }\"\"\"^^rdf:JSON as ?params)
}
"""

def main():
    g = Graph()
    print("-- Running original query (projects ?pr ?title ?time) --")
    res = list(g.query(QUERY_ORIGINAL))
    # Expect 1 row with all Nones
    for row in res:
        print(row)  # Should show (None, None, None)
    print(f"Rows: {len(res)}")

    print("\n-- Running variant that selects ?params --")
    res2 = list(g.query(QUERY_SHOW_PARAMS))
    for row in res2:
        print("?params literal:", row.params, "datatype=", getattr(row.params, 'datatype', None))
        # Raw lexical form
        print("lexical form:", str(row.params))
    print(f"Rows: {len(res2)}")

    if res2:
        lit = res2[0].params
        # rdflib will not automatically JSON-parse an rdf:JSON datatype.
        text = str(lit)
        # Optionally try to parse (convert single quotes to double quotes first)
        import json
        json_like = text.replace("'", '"')
        try:
            data = json.loads(json_like)
            print("Parsed JSON keys:", list(data.keys()))
        except Exception as e:
            print("JSON parse failed:", e)

if __name__ == "__main__":
    main()
