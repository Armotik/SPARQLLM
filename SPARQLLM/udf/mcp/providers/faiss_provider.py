# -*- coding: utf-8 -*-
"""FAISS MCP provider for SPARQLLM

Provides two tools:
- faiss.index_file: index a TTL file (schema:text) into a FAISS index and optionally persist it
- faiss.search_index: search a persisted FAISS index for a query string

Return contract: a minimal JSON-LD dict with media_type and jsonld payload.

This provider is intended to be called from the MCP tool layer, e.g.
  slm:SLM-MCP-TOOL("faiss","faiss.index_file", slm:SLM-OBJ("file","./events.nt.ttl","index_path","./events.index","persist",true))
and
  slm:SLM-MCP-TOOL("faiss","faiss.search_index", slm:SLM-OBJ("index_path","./events.index","query","music in Berlin","k",3))
"""
from __future__ import annotations
import os
import json
import logging
import tempfile
import hashlib
from typing import List, Dict, Any

from rdflib import Graph, Namespace, URIRef

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np
except Exception:
    # lazy import for environments without packages; provider will raise on use
    SentenceTransformer = None
    faiss = None
    np = None

logger = logging.getLogger("faiss_provider")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(asctime)s %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


class FaissProvider:
    SCHEMA = Namespace("http://schema.org/")

    def __init__(self):
        if SentenceTransformer is None or faiss is None:
            logger.warning("sentence-transformers or faiss not available; calls will fail until installed")

    def _extract_texts(self, ttl_path: str) -> List[Dict[str, Any]]:
        g = Graph()
        g.parse(ttl_path, format="turtle")
        entries = []
        for s in g.subjects(predicate=self.SCHEMA.text):
            text = g.value(subject=s, predicate=self.SCHEMA.text)
            props = {str(p): str(o) for p, o in g.predicate_objects(subject=s)}
            entries.append({"uri": str(s), "text": str(text), "props": props})
        return entries

    def tool_index_file(self, file: str, model: str = "all-MiniLM-L6-v2", index_path: str | None = None, persist: bool = False, normalize: bool = True) -> dict:
        """Index TTL file into FAISS.

        Args:
            file: path to ttl file
            model: sentence-transformers model name
            index_path: optional path to write faiss index and metadata (without extension)
            persist: whether to persist index and metadata to disk
            normalize: normalize embeddings (recommended for IndexFlatIP)

        Returns:
            dict: JSON-like contract with media_type and jsonld
        """
        logger.info("Indexing file %s (model=%s)" % (file, model))
        if SentenceTransformer is None:
            return {"error": "missing_dependency", "message": "Install sentence-transformers and faiss"}
        if not os.path.exists(file):
            return {"error": "file_not_found", "file": file}

        entries = self._extract_texts(file)
        texts = [e["text"] for e in entries]
        if not texts:
            return {"error": "no_texts_found", "file": file}

        model_obj = SentenceTransformer(model)
        emb = model_obj.encode(texts, normalize_embeddings=normalize)
        emb = np.array(emb, dtype='float32')
        dim = emb.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(emb)

        meta = {"entries": [{"uri": e["uri"], "props": e["props"]} for e in entries], "model": model, "count": len(entries)}

        if persist:
            if not index_path:
                # create temp base path
                base = tempfile.NamedTemporaryFile(delete=False).name
                index_path = base
            idx_file = index_path if index_path.endswith('.index') else index_path + '.index'
            meta_file = index_path + '.meta.json'
            # ensure directory exists
            d = os.path.dirname(os.path.abspath(idx_file))
            if d and not os.path.exists(d):
                os.makedirs(d, exist_ok=True)
            faiss.write_index(index, idx_file)
            with open(meta_file, 'w', encoding='utf-8') as fh:
                json.dump(meta, fh, ensure_ascii=False, indent=2)
            logger.info("Persisted index to %s and metadata to %s" % (idx_file, meta_file))
            jsonld = {"@context": {"schema": str(self.SCHEMA)}, "@type": "FaissIndex", "index_path": idx_file, "meta_path": meta_file, "count": len(entries)}
            return {"media_type": "application/ld+json", "jsonld": jsonld, "graph_anchor": idx_file}

        # In-memory return: not persisted, but provide metadata in payload
        jsonld = {"@context": {"schema": str(self.SCHEMA)}, "@type": "FaissIndex", "index_in_memory": True, "count": len(entries)}
        # Attach small sample metadata (uris)
        jsonld["uris"] = [e["uri"] for e in entries]
        return {"media_type": "application/ld+json", "jsonld": jsonld, "graph_anchor": file}

    def tool_search_index(self, index_path: str | None, query: str, k: int = 5, model: str = "all-MiniLM-L6-v2", file: str | None = None) -> dict:
        """Search a persisted FAISS index.

        Args:
            index_path: base path used when persisting (without extension) or full .index path
            query: query string
            k: number of results
            model: embedding model

        Returns:
            dict: JSON-LD with matches
        """
        logger.info("Searching index %s for query '%s'" % (index_path, query))
        if SentenceTransformer is None:
            return {"error": "missing_dependency", "message": "Install sentence-transformers and faiss"}
        # allow passing a ttl 'file' to create the index on demand
        if not index_path and file:
            # default index base is the file path without extension + '.faiss'
            index_path = os.path.splitext(file)[0] + '.faiss'

        if not index_path:
            return {"error": "missing_index_path", "message": "Provide index_path or file to build"}

        idx_file = index_path if index_path.endswith('.index') else index_path + '.index'
        meta_file = index_path + '.meta.json'

        # If index missing but a TTL file is provided, build and persist it first
        if (not os.path.exists(idx_file) or not os.path.exists(meta_file)) and file:
            logger.info("Index not found, creating index from file %s" % file)
            res = self.tool_index_file(file=file, model=model, index_path=index_path, persist=True)
            if res.get('error'):
                return {"error": "index_build_failed", "cause": res}

        if not os.path.exists(idx_file) or not os.path.exists(meta_file):
            return {"error": "index_not_found", "idx_file": idx_file, "meta_file": meta_file}

        try:
            index = faiss.read_index(idx_file)
        except Exception as e:
            logger.error("Failed to read index: %s", e)
            return {"error": "read_index_failed", "message": str(e)}

        with open(meta_file, 'r', encoding='utf-8') as fh:
            meta = json.load(fh)

        model_obj = SentenceTransformer(model)
        q_emb = model_obj.encode([query], normalize_embeddings=True)
        q_emb = np.array(q_emb, dtype='float32')
        D, I = index.search(q_emb, k)

        matches = []
        for score, idx in zip(D[0].tolist(), I[0].tolist()):
            if idx < 0 or idx >= len(meta.get('entries', [])):
                continue
            entry = meta['entries'][idx]
            matches.append({"uri": entry.get('uri'), "score": float(score), "props": entry.get('props')})

        # Build RDF-ready JSON-LD with @graph: one main result node linking to match nodes
        try:
            h = hashlib.sha256(json.dumps({"query": query, "matches": matches}, ensure_ascii=False).encode('utf-8')).hexdigest()[:16]
        except Exception:
            h = "tmp"

        result_id = f"urn:faiss:search:{h}"
        graph_obj = {"@context": {"schema": str(self.SCHEMA), "ex": "http://example.org/"}, "@graph": []}

        # main result node
        main_node = {"@id": result_id, "@type": "FaissSearchResult"}
        main_node[str(self.SCHEMA.query) if hasattr(self.SCHEMA, 'query') else "query"] = query

        match_nodes = []
        for m in matches:
            m_id = m.get('uri') or f"urn:faiss:match:{hashlib.sha256(json.dumps(m, ensure_ascii=False).encode('utf-8')).hexdigest()[:12]}"
            node = {"@id": m_id}
            if 'score' in m:
                node["http://example.org/score"] = m.get('score')
            props = m.get('props') or {}
            rdf_type = props.pop("http://www.w3.org/1999/02/22-rdf-syntax-ns#type", None)
            if rdf_type:
                node["@type"] = rdf_type
            for p, v in props.items():
                node[p] = v

            match_nodes.append(node)

            # link from main node to match using schema:hasPart
            key = str(self.SCHEMA.hasPart)
            main_node.setdefault(key, [])
            main_node[key].append({"@id": m_id})

        graph_obj["@graph"].append(main_node)
        graph_obj["@graph"].extend(match_nodes)

        return {"media_type": "application/ld+json", "jsonld": graph_obj, "graph_anchor": idx_file}

    # registry
    def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {"name": "faiss.index_file", "description": "Index a TTL file into FAISS", "inputs": ["file","model","index_path","persist"]},
            {"name": "faiss.search_index", "description": "Search a persisted FAISS index (or provide file to auto-build)", "inputs": ["index_path","file","query","k","model"]},
        ]

    def call(self, tool_name: str, args: dict) -> dict:
        if tool_name == "faiss.index_file":
            return self.tool_index_file(
                file=args.get('file'),
                model=args.get('model','all-MiniLM-L6-v2'),
                index_path=args.get('index_path'),
                persist=bool(args.get('persist', False))
            )
        if tool_name == "faiss.search_index":
            return self.tool_search_index(
                index_path=args.get('index_path'),
                query=args.get('query',''),
                k=int(args.get('k',5)),
                model=args.get('model','all-MiniLM-L6-v2'),
                file=args.get('file')
            )
        return {"error": "unknown_tool", "tool": tool_name}


# singleton factory
_provider_singleton: FaissProvider | None = None

def get_provider() -> FaissProvider:
    global _provider_singleton
    if _provider_singleton is None:
        _provider_singleton = FaissProvider()
    return _provider_singleton
