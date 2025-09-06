# sparqllm/udf/json_helpers.py
import json
def OBJ(*kv):
    # usage: OBJ("owner","octocat","repo","hello-world")
    it = iter(kv); return json.dumps({k:next(it) for k in it})
