# tools/echo_mcp_server.py  (MCP stdio jouet)
import sys, json

TOOLS = {
  "echo.ping": {"name":"echo.ping","description":"ping","input_schema":{"type":"object","properties":{"text":{"type":"string"}}}}
}

def write(msg): sys.stdout.write(json.dumps(msg)+"\n"); sys.stdout.flush()

for line in sys.stdin:
    req = json.loads(line)
    mid = req.get("id")
    method = req.get("method")
    params = req.get("params",{})

    if method == "tools/list":
        write({"jsonrpc":"2.0","id":mid,"result":{"tools": list(TOOLS.values())}})
    elif method == "tools/call":
        name = params["name"]; args = params.get("arguments",{})
        if name == "echo.ping":
            # renvoyer du JSON-LD pour tester la chaîne complète
            result = {
              "media_type":"application/ld+json",
              "jsonld": {
                 "@context":"https://schema.org/",
                 "@graph":[{"@id":"urn:ping","@type":"CreativeWork","name": args.get("text","pong")}]
              }
            }
            write({"jsonrpc":"2.0","id":mid,"result": result})
        else:
            write({"jsonrpc":"2.0","id":mid,"error":{"code":-32601,"message":"unknown tool"}})
    else:
        write({"jsonrpc":"2.0","id":mid,"error":{"code":-32601,"message":"unknown method"}})
