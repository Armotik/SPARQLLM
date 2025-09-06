# -*- coding: utf-8 -*-
"""PostgreSQL MCP-like provider.

Expose un tool principal: postgres.tables.list
Arguments attendus: host, port, db, user, password (ou DSN), schema (optionnel)
Retour: JSON-LD ancré sur l'URI logique du schéma.
"""
from __future__ import annotations
import os, typing as t, hashlib, re
import psycopg

class PostgresProvider:
    def __init__(self, dsn: str|None=None):
        self.default_dsn = dsn or os.getenv("PG_DSN")

    def _connect(self, args: dict):
        dsn = self.default_dsn
        if not dsn:
            # Construire un DSN à partir des args si fourni
            host = args.get("host","localhost")
            port = args.get("port",5432)
            db   = args.get("db") or args.get("database") or os.getenv("PGDATABASE")
            user = args.get("user") or os.getenv("PGUSER")
            password = args.get("password") or os.getenv("PGPASSWORD")
            # DSN minimal
            parts = []
            if host: parts.append(f"host={host}")
            if port: parts.append(f"port={port}")
            if db: parts.append(f"dbname={db}")
            if user: parts.append(f"user={user}")
            if password: parts.append(f"password={password}")
            dsn = " ".join(parts)
        if not dsn:
            raise ValueError("No DSN or connection parameters provided")
        return psycopg.connect(dsn)

    def tool_tables_list(self, **args) -> dict:
        schema = args.get("schema") or "public"
        try:
            with self._connect(args) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT table_name
                        FROM information_schema.tables
                        WHERE table_schema = %s AND table_type='BASE TABLE'
                        ORDER BY table_name
                    """, (schema,))
                    rows = [r[0] for r in cur.fetchall()]
        except Exception as e:
            return {"error": "pg_error", "message": str(e)}

        anchor = f"urn:postgres:schema:{schema}"
        jsonld = {
            "@context": {"schema": "https://schema.org/"},
            "@id": anchor,
            "@type": "schema:DataCatalog",
            "schema:hasPart": [
                {
                    "@id": f"{anchor}:table:{t}",
                    "@type": "schema:Table",
                    "schema:identifier": t,
                    "schema:name": t
                } for t in rows
            ],
            "schema:name": schema
        }
        return {"media_type": "application/ld+json", "jsonld": jsonld, "graph_anchor": anchor}

    def tool_table_preview(self, **args) -> dict:
        """Retourne un échantillon de lignes pour une table donnée sous forme JSON-LD.

        Args attendus: schema, table, limit (+ mêmes paramètres de connexion que list).
        """
        schema = args.get("schema") or "public"
        table = args.get("table") or args.get("table_name")
        limit = int(args.get("limit", 5))
        if not table:
            return {"error": "missing_table"}
        # Sécurité minimaliste: whitelist pattern simple (alphanum + underscore)
        import re
        if not re.fullmatch(r"[A-Za-z0-9_]+", table):
            return {"error": "invalid_table_name"}
        try:
            with self._connect(args) as conn:
                with conn.cursor() as cur:
                    # Colonnes
                    cur.execute("""
                        SELECT column_name
                        FROM information_schema.columns
                        WHERE table_schema=%s AND table_name=%s
                        ORDER BY ordinal_position
                    """, (schema, table))
                    columns = [r[0] for r in cur.fetchall()]
                    # Lignes
                    col_list = ",".join([f'"{c}"' for c in columns]) if columns else '*'
                    cur.execute(f'SELECT {col_list} FROM "{schema}"."{table}" LIMIT %s', (limit,))
                    rows = cur.fetchall()
        except Exception as e:
            return {"error": "pg_error", "message": str(e)}

        table_anchor = f"urn:postgres:schema:{schema}:table:{table}"
        # Construire JSON-LD
        parts = []
        for idx, row in enumerate(rows):
            props = []
            for c, v in zip(columns, row):
                if v is None:
                    continue
                props.append({
                    "@type": "schema:PropertyValue",
                    "schema:name": c,
                    "schema:value": str(v)
                })
            parts.append({
                "@type": "schema:Observation",
                "schema:position": idx,
                "schema:additionalProperty": props
            })

        jsonld = {
            "@context": {"schema": "https://schema.org/"},
            "@id": table_anchor,
            "@type": "schema:Table",
            "schema:name": table,
            "schema:hasPart": parts
        }
        return {"media_type": "application/ld+json", "jsonld": jsonld, "graph_anchor": table_anchor}

    def tool_sql_query(self, **args) -> dict:
        """Exécute une requête SQL en lecture (SELECT) et retourne un fragment JSON-LD.

        Args:
          sql: requête SELECT (obligatoire)
          limit: nombre max de lignes à retourner (fallback si pas de LIMIT dans la requête)
          schema: (optionnel) pour l'ancre (sinon public)
        Sécurisation minimale: refuse si non-SELECT ou présence de mots clés sensibles.
        """
        sql = args.get("sql")
        if not sql:
            return {"error": "missing_sql"}
        sql_stripped = sql.strip().rstrip(';')
        if not sql_stripped.lower().startswith("select"):
            return {"error": "only_select_allowed"}
        forbidden = re.compile(r"\\b(insert|update|delete|drop|alter|grant|revoke|truncate)\\b", re.IGNORECASE)
        if forbidden.search(sql_stripped):
            return {"error": "forbidden_keyword"}
        limit = int(args.get("limit", 50))
        applied_limit = False
        if re.search(r"limit\\s+\\d+", sql_stripped, re.IGNORECASE):
            applied_limit = True
        rows = []
        columns: list[str] = []
        try:
            with self._connect(args) as conn:
                with conn.cursor() as cur:
                    cur.execute(sql_stripped)
                    if cur.description:
                        columns = [d.name for d in cur.description]
                    fetch_rows = cur.fetchmany(limit) if not applied_limit else cur.fetchall()
                    if not applied_limit and len(fetch_rows) > limit:
                        fetch_rows = fetch_rows[:limit]
                    rows = fetch_rows
        except Exception as e:
            return {"error": "pg_error", "message": str(e)}

        schema = args.get("schema") or "public"
        anchor = f"urn:postgres:query:{hashlib.sha256(sql_stripped.encode()).hexdigest()[:16]}"
        parts = []
        for idx, row in enumerate(rows):
            props = []
            for c, v in zip(columns, row):
                if v is None:
                    continue
                props.append({
                    "@type": "schema:PropertyValue",
                    "schema:name": c,
                    "schema:value": str(v)
                })
            parts.append({
                "@type": "schema:Observation",
                "schema:position": idx,
                "schema:additionalProperty": props
            })
        jsonld = {
            "@context": {"schema": "https://schema.org/"},
            "@id": anchor,
            "@type": "schema:Dataset",
            "schema:name": f"SQL result ({schema})",
            "schema:hasPart": parts,
            "schema:variableMeasured": columns
        }
        return {"media_type": "application/ld+json", "jsonld": jsonld, "graph_anchor": anchor}

    def list_tools(self) -> list[dict]:
        return [
            {"name": "postgres.tables.list", "description": "List tables in a schema", "inputs": ["schema", "host","port","db","user","password"]},
            {"name": "postgres.table.preview", "description": "Preview rows of a table", "inputs": ["schema","table","limit","host","port","db","user","password"]},
            {"name": "postgres.sql.query", "description": "Run a read-only SQL SELECT", "inputs": ["sql","limit","schema","host","port","db","user","password"]}
        ]

    def call(self, tool_name: str, args: dict) -> dict:
        if tool_name == "postgres.tables.list":
            return self.tool_tables_list(**args)
        if tool_name == "postgres.table.preview":
            return self.tool_table_preview(**args)
        if tool_name == "postgres.sql.query":
            return self.tool_sql_query(**args)
        return {"error": "unknown_tool", "tool": tool_name}

_pg_singleton: PostgresProvider|None = None

def get_pg_provider() -> PostgresProvider:
    global _pg_singleton
    if _pg_singleton is None:
        _pg_singleton = PostgresProvider()
    return _pg_singleton
