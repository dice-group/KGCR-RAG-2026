"""
Mini Freebase server for the ARoG demo.

Plays the role of the Virtuoso/Freebase SPARQL endpoint from the paper, but serves only the small
knowledge graph in mini_kg/mini_freebase.nt, so it runs on a normal laptop.

Start it from the ARoG folder and leave the window open:
    python demo_kit\\mini_kg_server.py

It listens on http://127.0.0.1:8890/sparql (the same port Virtuoso uses in the paper's setup).
"""
import argparse
import logging
import os
import re

from flask import Flask, Response, request
from rdflib import Graph

KIT = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
GRAPH = Graph()


def to_standard_sparql(query):
    """ARoG's queries were written for Virtuoso, which also accepts the words OR / AND inside FILTER.
    Standard SPARQL (and rdflib) needs || and &&, so translate them. The meaning is unchanged."""
    return re.sub(r"\s+OR\s+", " || ", re.sub(r"\s+AND\s+", " && ", query))


@app.route("/sparql", methods=["GET", "POST"])
def sparql():
    query = request.values.get("query")
    if not query and request.data:
        query = request.data.decode("utf-8")
    if not query:
        return Response(f"Mini Freebase SPARQL endpoint: {len(GRAPH)} triples. Send a SPARQL query as ?query=...",
                        mimetype="text/plain")
    try:
        result = GRAPH.query(to_standard_sparql(query))
        body = result.serialize(format="json")
        log_query(query, len(result))
        return Response(body, mimetype="application/sparql-results+json")
    except Exception as exc:  # report bad queries instead of crashing
        print(f" query error: {exc}", flush=True)
        return Response(f"Query error: {exc}", status=400, mimetype="text/plain")


QUERY_COUNT = [0]


def log_query(query, n_results):
    """Print one short line per query, so the audience can see ARoG asking the knowledge graph."""
    QUERY_COUNT[0] += 1
    ids = re.findall(r"ns:([mg]\.[0-9a-z_]+)", query)
    rels = [r for r in re.findall(r"ns:([a-z0-9_]+(?:\.[a-z0-9_]+){2,})", query) if r != "type.object.name"]
    if "type.object.name" in query:
        what = "name of"
    elif "?relation" in query:
        what = "relations of"
    else:
        what = "facts for"
    text = f" query {QUERY_COUNT[0]:>4}: {what} {', '.join(dict.fromkeys(ids)) or '(all)'}"
    if rels:
        text += f" via {rels[0]}"
    print(f"{text}  -> {n_results} result{'s' if n_results != 1 else ''}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Mini Freebase SPARQL endpoint for the ARoG demo")
    parser.add_argument("--port", type=int, default=8890)
    parser.add_argument("--kg", default=os.path.join(KIT, "mini_kg", "mini_freebase.nt"))
    args = parser.parse_args()

    GRAPH.parse(args.kg, format="nt")
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    print("=" * 70)
    print(f" Mini Freebase is running: {len(GRAPH)} triples loaded from {os.path.basename(args.kg)}")
    print(f" Address: http://127.0.0.1:{args.port}/sparql")
    print(" Leave this window open while you run the demo. Press Ctrl+C to stop.")
    print("=" * 70)
    app.run(host="127.0.0.1", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
