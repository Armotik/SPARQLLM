# client MCP générique (HTTP + STDIO)
import json, uuid, threading, queue, subprocess, requests, os
from pprint import pprint 

class MCPHandle:
    def __init__(self, kind, target, token=None, proc=None, registry=None):
        self.kind, self.target, self.token, self.proc = kind, target, token, proc
        self.registry = registry or {}
        self.q = queue.Queue()

class MCPClient:
    def __init__(self):
        self._handles = {}

    def connect_stdio(self, name, cmd):
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        h = MCPHandle("stdio", " ".join(cmd), proc=p)
        self._handles[name] = h
        threading.Thread(target=self._reader, args=(h,), daemon=True).start()
        return name

    def connect_http(self, name, base_url, token=None):
        self._handles[name] = MCPHandle("http", base_url, token)
        return name

    # --- Ajout: handle statique (pas un vrai serveur MCP) ---
    def connect_static(self, name):
        if name in self._handles:
            return name
        self._handles[name] = MCPHandle("static", target=name, registry={})
        return name

    def register_static_tool(self, handle_name, tool_name, func):
        h = self._handles.get(handle_name)
        if not h or h.kind != "static":
            raise ValueError(f"Handle {handle_name} n'est pas de type static")
        h.registry[tool_name] = func
        return tool_name

    def tools_list(self, name):
        return self._rpc(name, "tools/list", {})


    
    def tools_call(self, name, tool, arguments):
        h = self._handles[str(name)]
#        print(f"self._handles: {self._handles},self._handles[name]= {self._handles[str(name)]}")
#        print(f"Calling tool {tool} on handle {name} with args {arguments}")
        if h.kind == "static":
#            print(f"Static handle {name} with registry: {h.registry}")
#            print("REGISTRY KEYS =", list(h.registry.keys()))
            func = h.registry.get(str(tool))
            if not func:
                raise RuntimeError(f"Tool statique introuvable: {tool}")
            return func(arguments)
        return self._rpc(name, "tools/call", {"name": tool, "arguments": arguments})

    def has_tool(self, name, tool):
        h = self._handles.get(str(name))
        if not h:
            return False
        if h.kind != "static":
            return False
        return tool in h.registry

    def _reader(self, h):
        for line in h.proc.stdout:
            try: h.q.put(json.loads(line))
            except: pass

    def _rpc(self, name, method, params):
        h = self._handles[str(name)]
        rid = str(uuid.uuid4())
        req = {"jsonrpc":"2.0","id":rid,"method":method,"params":params}
        if h.kind == "stdio":
            h.proc.stdin.write(json.dumps(req)+"\n"); h.proc.stdin.flush()
            while True:
                msg = h.q.get()
                if msg.get("id")==rid:
                    if "error" in msg: raise RuntimeError(msg["error"])
                    return msg.get("result", {})
        else:
            headers = {"Content-Type":"application/json"}
            if h.token: headers["Authorization"]=h.token
            r = requests.post(h.target.rstrip("/")+"/json-rpc", json=req, headers=headers, timeout=60)
            r.raise_for_status()
            msg = r.json()
            if "error" in msg: raise RuntimeError(msg["error"])
            return msg.get("result", {})
