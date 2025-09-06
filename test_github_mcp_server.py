## Not a MCP server -> just standard API call to GitHub

import requests
import json
import os
from rdflib import Graph

def json_to_jsonld(data):
    """
    Convert JSON data to JSON-LD format.
    """
    context = "https://schema.org/"
    jsonld_data = {
        "@context": context,
        "@graph": data
    }
    return jsonld_data

def test_github_public_api():
    # URL of the GitHub public API to list pull requests
    owner = "momo54"  # Updated for SPARQLLM
    repo = "SPARQLLM"  # Updated for SPARQLLM
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls"

    # Request parameters
    params = {
        "state": "open",  # Open pull requests
        "per_page": 10    # Limit to 10 results
    }

    # Read the GitHub token from an environment variable
    token = os.getenv("GITHUB_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    # Perform the GET request
    response = requests.get(url, params=params, headers=headers)

    # Check the response status
    if response.status_code == 200:
        print("Response received:")
        data = response.json()
        print(json.dumps(data, indent=2))

        # Convert JSON to JSON-LD
        jsonld_data = json_to_jsonld(data)
        print("\nConverted to JSON-LD:")
        print(json.dumps(jsonld_data, indent=2))

        # Optional: Parse JSON-LD with rdflib
        g = Graph()
        g.parse(data=json.dumps(jsonld_data), format="json-ld")
        print("\nRDF Triples:")
        print(g.serialize(format="turtle"))  # Removed .decode("utf-8")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    test_github_public_api()
