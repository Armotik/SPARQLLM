# SPARQLLM/udf/mcp/mappers/github.py
def map_github_prs_to_jsonld(api_json: dict) -> dict:
    items = []
    arr = api_json.get("items", api_json.get("pulls", [])) or []
    for pr in arr:
        url   = pr.get("html_url") or pr.get("url")
        title = pr.get("title")
        state = pr.get("state")
        created = pr.get("created_at")  # ISO
        author = (pr.get("user") or {}).get("login")
        items.append({
            "@id": url,
            "@type": "CreativeWork",
            "name": title,
            "dateCreated": created,
            "author": author and {"@type":"Person","name": author},
            "additionalType": state
        })
    return {"@context":"https://schema.org/","@graph":items}
