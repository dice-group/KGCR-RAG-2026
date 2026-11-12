"""
Builds the mini knowledge graph and the demo question list used by the ARoG demo.

You do NOT need to run this: its output (mini_kg/mini_freebase.nt, mini_kg/facts_readable.txt and
demo_questions.json) is already in the kit. It is included so anyone can see exactly where every
fact comes from.

Where the facts come from (each fact is tagged in facts_readable.txt):
  [WebQSP]   answers annotated in the WebQSP benchmark (data/WebQSP.json in the ARoG repository)
  [CWQ]      the first hop of a CWQ benchmark question, read from its SPARQL query (data/cwq_1000.json)
  [Figure2]  the worked example in Figure 2 of the ARoG paper (artist and daughter of the tour)
  [reverse]  the reverse direction of a fact above, added by this kit because real Freebase stores
             most facts in both directions (for example people.person.children / people.person.parents)

Usage (from the ARoG folder):  python demo_kit\\build_mini_kg.py
"""
import json
import os
import re
import sys

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import XSD

KIT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(KIT) if len(sys.argv) < 2 else sys.argv[1]
NS = Namespace("http://rdf.freebase.com/ns/")

# Countries whose simple WebQSP facts go into the graph (more countries = more realistic distractors).
TOPICS = {
    "m.02k54": "Egypt", "m.03_3d": "Japan", "m.015fr": "Brazil", "m.0chghy": "Australia",
    "m.03rjj": "Italy", "m.06mkj": "Spain", "m.0f8l9c": "France", "m.0345h": "Germany",
    "m.0b90_r": "Mexico", "m.09c7w0": "United States of America",
}

# The 8 demo questions: 4 one-hop WebQSP questions and 4 two-hop CWQ questions.
DEMO_WEBQSP = [
    ("W1", "what language does egyptian people speak?"),
    ("W2", "what kind of government is egypt?"),
    ("W3", "what do they call money in japan?"),
    ("W4", "what is the official language of brazil now?"),
]
DEMO_CWQ = [
    ("C1", 'What is the name of the daughter of the artist who had the "The Mrs. Carter Show World Tour"?'),
    ("C2", "Where was the main artist featured in the Rihanna: Live in Concert Tour raised?"),
    ("C3", "What is the currency for the place where the Egyptian Mau breed originated?"),
    ("C4", "What type of government does the country with Central Western Time Zone have?"),
]

# Reverse properties as named in Freebase. Used only to give entities more than one relation.
REVERSE = {
    "people.person.children": "people.person.parents",
    "music.artist.concert_tours": "music.concert_tour.artist",
    "location.country.currency_used": "finance.currency.countries_used",
    "location.country.languages_spoken": "language.human_language.countries_spoken_in",
    "location.country.form_of_government": "government.form_of_government.countries",
    "location.location.time_zones": "time.time_zone.locations_in_this_time_zone",
    "people.person.place_of_birth": "location.location.people_born_here",
    "location.location.containedby": "location.location.contains",
    "biology.breed_origin.breeds_originating_here": "biology.animal_breed.place_of_origin",
    "base.locations.countries.continent": "base.locations.continents.countries_within",
}


def simple_parse(p, max_answers=6):
    return (len(p.get("InferentialChain") or []) == 1 and not p["Constraints"] and p["Time"] is None
            and p["Order"] is None and p["Answers"] and len(p["Answers"]) <= max_answers
            and all(a["AnswerType"] == "Entity" and a["EntityName"] for a in p["Answers"]))


def main():
    webqsp = json.load(open(os.path.join(REPO, "data", "WebQSP.json"), encoding="utf-8"))
    cwq = json.load(open(os.path.join(REPO, "data", "cwq_1000.json"), encoding="utf-8"))
    wq_by_id = {q["QuestionId"]: q for q in webqsp}

    names = dict(TOPICS)
    facts = {}  # (s, p, o) -> source tag

    def add(s, p, o, tag):
        if (s, p, o) not in facts:
            facts[(s, p, o)] = tag

    # 1) Simple WebQSP facts about the chosen countries.
    for q in webqsp:
        for p in q["Parses"]:
            if p["TopicEntityMid"] in TOPICS and simple_parse(p):
                for a in p["Answers"]:
                    names[a["AnswerArgument"]] = a["EntityName"]
                    add(p["TopicEntityMid"], p["InferentialChain"][0], a["AnswerArgument"], "WebQSP")

    # 2) CWQ composition questions: first hop from the SPARQL query, second hop from the WebQSP parse.
    demo_cwq_q = {q for _, q in DEMO_CWQ}
    for x in cwq:
        if x.get("compositionality_type") != "composition":
            continue
        wq = wq_by_id.get(x.get("webqsp_ID"))
        if not wq or not simple_parse(wq["Parses"][0]):
            continue
        p = wq["Parses"][0]
        c_mid = p["TopicEntityMid"]
        pats = re.findall(r"(\S+) ns:(\S+) (\S+) \.", x["sparql"])
        if len(pats) != 2 or "ORDER" in x["sparql"] or "LIMIT" in x["sparql"]:
            continue
        if not (c_mid in TOPICS or x["question"] in demo_cwq_q):
            continue
        first = [t for t in pats if t[0] == "?c" and t[2].startswith("ns:")]
        if not first or not x["topic_entity"]:
            continue
        rel, topic = first[0][1], first[0][2][3:]
        names[c_mid] = p["TopicEntityName"]
        names.update(x["topic_entity"])
        add(c_mid, rel, topic, "CWQ")
        for a in p["Answers"]:
            names[a["AnswerArgument"]] = a["EntityName"]
            add(c_mid, p["InferentialChain"][0], a["AnswerArgument"], "WebQSP")

    # 3) Figure 2 of the paper: the tour's artist and her daughter.
    names.update({"m.0qftnyr": "The Mrs. Carter Show World Tour", "m.01mpq7s": "Beyoncé Knowles",
                  "m.0hvsrmc": "Blue Ivy Carter"})
    add("m.01mpq7s", "music.artist.concert_tours", "m.0qftnyr", "Figure2")
    add("m.01mpq7s", "people.person.children", "m.0hvsrmc", "Figure2")

    # 4) Reverse directions.
    for (s, p, o), tag in list(facts.items()):
        if p in REVERSE:
            add(o, REVERSE[p], s, "reverse")

    # Build the RDF graph. Every entity gets an English name (as in Freebase's type.object.name).
    g = Graph()
    used = set()
    for (s, p, o) in facts:
        g.add((NS[s], NS[p], NS[o]))
        used.update([s, o])
    # One literal value, as in Figure 2 (the tour's start date), so the "specific value" hiding is visible.
    g.add((NS["m.0qftnyr"], NS["music.concert_tour.start_date"], Literal("2013-04-15", datatype=XSD.date)))
    for mid in used:
        g.add((NS[mid], NS["type.object.name"], Literal(names[mid], lang="en")))

    os.makedirs(os.path.join(KIT, "mini_kg"), exist_ok=True)
    g.serialize(destination=os.path.join(KIT, "mini_kg", "mini_freebase.nt"), format="nt", encoding="utf-8")

    # Human-readable listing for the presentation.
    lines = ["Mini knowledge graph used in the ARoG demo",
             "Format: subject  --relation-->  object   [source]", ""]
    for (s, p, o), tag in sorted(facts.items(), key=lambda kv: (names[kv[0][0]], kv[0][1], names[kv[0][2]])):
        lines.append(f"{names[s]} ({s})  --{p}-->  {names[o]} ({o})   [{tag}]")
    lines.append("The Mrs. Carter Show World Tour (m.0qftnyr)  --music.concert_tour.start_date-->  2013-04-15   [Figure2]")
    with open(os.path.join(KIT, "mini_kg", "facts_readable.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # Demo question records, copied unchanged from the benchmark files (plus a demo_id).
    wq_by_q = {q["RawQuestion"]: q for q in webqsp}
    cwq_by_q = {q["question"]: q for q in cwq}
    demo = {"webqsp": [], "cwq": []}
    for did, q in DEMO_WEBQSP:
        rec = dict(wq_by_q[q]); rec["demo_id"] = did; demo["webqsp"].append(rec)
    for did, q in DEMO_CWQ:
        rec = dict(cwq_by_q[q]); rec["demo_id"] = did; demo["cwq"].append(rec)
    with open(os.path.join(KIT, "demo_questions.json"), "w", encoding="utf-8") as f:
        json.dump(demo, f, ensure_ascii=False, indent=1)

    tags = {}
    for t in facts.values():
        tags[t] = tags.get(t, 0) + 1
    print(f"Entities: {len(used)}  Facts: {len(facts)}  RDF triples incl. names: {len(g)}  by source: {tags}")


if __name__ == "__main__":
    main()
