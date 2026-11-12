"""
ARoG demo launcher (helper kit)
===============================

Runs the authors' ARoG code UNCHANGED (main_freebase.py, utils.py, freebase_func.py) and applies three
fixes while it runs, without editing their files:

  1. OpenAI calls go straight to OpenAI. The original code sends them through a proxy at
     socks5://127.0.0.1:7891 that only existed on the authors' computer, so it would retry forever.
  2. The Freebase (SPARQL) and SentenceTransformer addresses point to your servers. In the original
     files they are placeholders such as "http://xxx.xxx.xxx.xxx:8890/sparql".
  3. The SentenceTransformer address uses port 5001, the port lm_server.py really listens on
     (utils.py says 5000).

It also prints a readable trace of what each ARoG module does, checks every prompt sent to the LLM for
hidden entity names (a privacy check), scores answers with the authors' own matching rule, and saves
everything in demo_kit\\outputs.

Commands (run them from the ARoG folder, with the "arog" environment active):

  python demo_kit\\arog_demo.py check              check that everything is ready (costs < $0.001)
  python demo_kit\\arog_demo.py run --only C1      run one demo question with a step-by-step trace
  python demo_kit\\arog_demo.py run                run all 8 demo questions (about 3 to 6 minutes)
  python demo_kit\\arog_demo.py show               show the latest run again (no internet needed)
  python demo_kit\\arog_demo.py kg                 print the mini knowledge graph
  python demo_kit\\arog_demo.py benchmark --sparql http://SERVER:8890/sparql --dataset webqsp --limit 30
                                                   run real benchmark questions on a full Freebase server
"""
import argparse
import datetime
import glob
import importlib.util
import json
import math
import os
import random
import re
import shutil
import sys
import time
import traceback

KIT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(KIT_DIR)
OUT_ROOT = os.path.join(KIT_DIR, "outputs")
DEFAULT_MODEL = "gpt-4o-mini-2024-07-18"          # the model used in the paper
PRICE_IN, PRICE_OUT = 0.15, 0.60                  # USD per 1M tokens for gpt-4o-mini (OpenAI pricing, Oct 2026)
PAPER = {"webqsp": (74.7, 58.9), "cwq": (60.0, 36.3), "grailqa": (78.7, 71.8)}  # Table 2, Hits@1 total / filtered
COT_RIGHT = {"webqsp": "CoT/cot_webqsp_right.json", "cwq": "CoT/cot_cwq_alias_right.json",
             "grailqa": "CoT/cot_grailqa_right.json"}
QUESTION_KEY = {"webqsp": "RawQuestion", "cwq": "question", "grailqa": "question"}
LINE = "-" * 78

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

TRANSCRIPT = []


def say(msg=""):
    print(msg, flush=True)
    TRANSCRIPT.append(msg)


# --------------------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------------------
def read_key(cli_key=None):
    key = cli_key or os.environ.get("OPENAI_API_KEY", "")
    path = os.path.join(KIT_DIR, "openai_key.txt")
    if not key and os.path.exists(path):
        with open(path, encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    key = line
                    break
    if not key or "PASTE" in key.upper():
        return None
    return key


def clean_results(string):
    """Copied from evaluation/eval_arog/utils.py: take the text inside the first {...}."""
    string = string.replace("{Yes}", "").replace("{No}", "")
    if "{" in string:
        start = string.find("{") + 1
        end = string.find("}")
        return string[start:end]
    return string


def exact_match(response, answers):
    """Copied from evaluation/eval_arog/utils.py: the authors' answer-matching rule."""
    response = re.sub(r"\(.*?\)", "", response).strip()
    clean_result = response.strip().replace(" ", "").lower()
    for answer in answers:
        clean_answer = answer.strip().replace(" ", "").lower()
        if clean_result == clean_answer or clean_result in clean_answer or clean_answer in clean_result:
            return True
    return False


def score_record(rec):
    """Same decision as the authors' eval.py for one saved answer. Returns (status, answer_text)."""
    results = str(rec.get("results") or "")
    if "{" in str(rec.get("reasoning_chains_str", "")):
        return "skipped", results            # eval.py skips these (constraints_refuse rule)
    if "{" in results:
        answer = clean_results(results)
        if answer == "NULL":
            return "skipped", answer
        return ("correct" if exact_match(answer, rec["answer_list"]) else "wrong"), answer
    return ("correct" if exact_match(results, rec["answer_list"]) else "wrong"), results


def variable_part(prompt):
    """The part of a prompt that changes per question (after the fixed examples of the template)."""
    i = max(prompt.rfind("Q: "), prompt.rfind("Context: "))
    return prompt[i:] if i >= 0 else prompt


def find_names(text, names):
    low = text.lower()
    hits = []
    for name in names:
        n = name.lower()
        if re.search(r"(?<![a-z0-9])" + re.escape(n) + r"(?![a-z0-9])", low):
            hits.append(name)
    return hits


def money(tokens_in, tokens_out):
    return tokens_in / 1e6 * PRICE_IN + tokens_out / 1e6 * PRICE_OUT


def new_run_dir(tag):
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(OUT_ROOT, f"run_{stamp}_{tag}")
    os.makedirs(path, exist_ok=True)
    return path


def arog_output_files():
    return glob.glob(os.path.join(REPO_DIR, "ARoG_0720_naive_1_*.jsonl")) + \
        glob.glob(os.path.join(REPO_DIR, "ARoG_0720_naive_1_*.json"))


def move_old_outputs():
    old = arog_output_files()
    if old:
        dest = os.path.join(OUT_ROOT, "older_arog_files_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(dest, exist_ok=True)
        for f in old:
            shutil.move(f, os.path.join(dest, os.path.basename(f)))


# --------------------------------------------------------------------------------------------------
# check
# --------------------------------------------------------------------------------------------------
def cmd_check(a):
    all_ok = True

    def report(ok, label, detail="", fix=""):
        nonlocal all_ok
        all_ok = all_ok and ok
        say(f"[{'  OK  ' if ok else ' FAIL '}] {label}" + (f": {detail}" if detail else ""))
        if not ok and fix:
            say(f"         How to fix: {fix}")

    say(LINE)
    say("ARoG demo: readiness check")
    say(LINE)
    v = sys.version_info
    report((3, 9) <= v[:2] <= (3, 12), "Python version",
           f"{v.major}.{v.minor}.{v.micro}" + ("" if v[:2] == (3, 10) else " (the guide uses 3.10)"),
           "Activate the environment first:  conda activate arog")

    needed = ["main_freebase.py", "utils.py", "freebase_func.py", "prompt_list.py", "lm_server.py",
              "data/WebQSP.json", "data/cwq_1000_alias.json", "CoT/cot_webqsp_right.json"]
    missing = [f for f in needed if not os.path.exists(os.path.join(REPO_DIR, f))]
    report(not missing, "ARoG code found", REPO_DIR if not missing else "missing: " + ", ".join(missing),
           "The demo_kit folder must sit inside the ARoG folder, next to main_freebase.py.")

    modules = ["openai", "httpx", "httpx_socks", "zhipuai", "nltk", "rank_bm25", "sentence_transformers",
               "torch", "SPARQLWrapper", "rdflib", "flask", "requests", "tqdm"]
    missing = [m for m in modules if importlib.util.find_spec(m) is None]
    report(not missing, "Python packages", "all installed" if not missing else "missing: " + ", ".join(missing),
           "pip install -r demo_kit\\requirements_windows.txt")

    try:
        from SPARQLWrapper import SPARQLWrapper, JSON
        s = SPARQLWrapper(a.sparql)
        s.setTimeout(60)
        s.setReturnFormat(JSON)
        # A cheap test that works on both the mini graph and the full Freebase: the name of Egypt (m.02k54).
        s.setQuery("PREFIX ns: <http://rdf.freebase.com/ns/> SELECT ?n WHERE { ns:m.02k54 ns:type.object.name ?n } LIMIT 3")
        names = [b["n"]["value"] for b in s.query().convert()["results"]["bindings"]]
        detail = f"answering at {a.sparql} (m.02k54 is '{names[0] if names else 'not found'}')"
        if "127.0.0.1" in a.sparql or "localhost" in a.sparql:
            s.setQuery("SELECT (COUNT(*) AS ?n) WHERE { ?s ?p ?o }")
            n = s.query().convert()["results"]["bindings"][0]["n"]["value"]
            detail = f"{int(n):,} triples at {a.sparql}"
        report(bool(names), "Knowledge graph server", detail,
               "The server answers but does not know Egypt (m.02k54). Is it a Freebase server?")
    except Exception as exc:
        local = "127.0.0.1" in a.sparql or "localhost" in a.sparql
        report(False, "Knowledge graph server", f"no answer from {a.sparql} ({type(exc).__name__})",
               "Start it in another window:  python demo_kit\\mini_kg_server.py" if local else
               "Check the server address, and that your network or the university VPN lets you reach it.")

    try:
        import requests
        r = requests.post(a.embed.rstrip("/") + "/sentence_transformer", timeout=120,
                          json={"width": 1, "question": "What is the capital city of this country?",
                                "total_relations": "location.country.capital&;& music.artist.track"})
        top = r.json()["topn_relations"][0]
        report(True, "Embedding server (SentenceTransformer)",
               f"working at {a.embed} (for a question about a capital city it ranked '{top}' first)")
    except Exception as exc:
        report(False, "Embedding server (SentenceTransformer)", f"no answer from {a.embed} ({type(exc).__name__})",
               "Start it in another window:  python lm_server.py   (the first start downloads a 260 MB model)")

    key = read_key(a.key)
    if not key:
        report(False, "OpenAI API key", "not found",
               "Paste your key into demo_kit\\openai_key.txt (replace the PASTE line) and save the file.")
    else:
        try:
            import httpx
            import openai
            client = openai.OpenAI(api_key=key, http_client=httpx.Client(timeout=60.0))
            r = client.chat.completions.create(model=a.model, max_tokens=5, temperature=0,
                                               messages=[{"role": "user", "content": "Reply with the single word OK."}])
            report(True, "OpenAI API", f"key accepted, model {a.model} replied '{r.choices[0].message.content.strip()}'")
        except Exception as exc:
            text = str(exc)
            name = type(exc).__name__
            if "insufficient_quota" in text or "quota" in text.lower():
                fix = "Your OpenAI account has no credit. Add at least $5 under Settings > Billing."
            elif name == "AuthenticationError":
                fix = "The key was rejected. Create a new key and paste it into demo_kit\\openai_key.txt."
            elif name in ("NotFoundError", "BadRequestError", "PermissionDeniedError"):
                fix = (f"The model '{a.model}' is not available to your account. Add --model gpt-4o-mini to the "
                       "command, or pick a current model from platform.openai.com/docs/models.")
            elif name in ("APIConnectionError", "APITimeoutError"):
                fix = "No connection to OpenAI. Check your internet connection (and any VPN)."
            else:
                fix = "Read the error above; if it mentions billing, add credit to your OpenAI account."
            report(False, "OpenAI API", f"{name}: {text[:160]}", fix)

    say(LINE)
    say("Everything is ready. Next:  python demo_kit\\arog_demo.py run --only C1" if all_ok
        else "Fix the FAIL lines above, then run the check again.")
    return 0 if all_ok else 1


# --------------------------------------------------------------------------------------------------
# Loading the authors' code with the three fixes and the trace
# --------------------------------------------------------------------------------------------------
class State:
    def __init__(self):
        self.trace = True
        self.calls = []          # every LLM call of the current question
        self.all_calls = []
        self.names = {}          # id -> name, the user-side map (never sent to the LLM)
        self.hop = 0
        self.hop_printed = 0

    def begin(self):
        self.calls, self.names, self.hop, self.hop_printed = [], {}, 1, 0


STATE = State()


def arog_args(dataset, model, key, width=3, depth=3):
    """The same settings as the defaults in main_freebase.py (the paper's settings)."""
    return argparse.Namespace(
        dataset=dataset, max_length=256, temperature_exploration=0.4, temperature_reasoning=0, width=width,
        depth=depth, remove_unnecessary_rel=True, question_abs=True, entity_abs=True, filter_cot=False,
        filter_judge=True, reason_with_name=False, naive_retrieve=True, slm_triple_prune=True, LLM_type=model,
        opeani_api_keys=key, num_retain_entity=5, prune_tools="llm", max_worker=1)


def load_arog(a, run_dir):
    os.chdir(REPO_DIR)
    if REPO_DIR not in sys.path:
        sys.path.insert(0, REPO_DIR)
    say("Loading the ARoG code (this takes 10 to 30 seconds the first time)...")
    import logging
    logging.basicConfig(filename=os.path.join(run_dir, "arog_log.txt"), filemode="w", level=logging.INFO,
                        encoding="utf-8", format="%(asctime)s %(message)s")
    import httpx
    import freebase_func
    import utils
    import main_freebase

    # Fixes 2 and 3: server addresses (set in every module that holds a copy of the name).
    embed = a.embed.rstrip("/")
    for m in (freebase_func, utils, main_freebase):
        m.SPARQLPATH = a.sparql
    for m in (utils, main_freebase):
        m.SLMPATH = embed + "/sentence_transformer"
        m.SLMPATH_COS = embed + "/sentence_transformer_cos"
        m.SLMPATH_COS_LIST = embed + "/sentence_transformer_cos_list"
    freebase_func.SLMPATH = embed + "/sentence_transformer"
    # Fix 1: a direct connection to OpenAI instead of the authors' local proxy.
    utils.http_client = httpx.Client(timeout=120.0)

    # The original code prints every prompt and query; send that to a log file instead of the screen.
    console_log = open(os.path.join(run_dir, "arog_console_log.txt"), "a", encoding="utf-8")

    def quiet_print(*args, **kwargs):
        text = " ".join(str(x) for x in args)
        console_log.write(text + "\n")
        console_log.flush()
        low = text.lower()
        if "error" in low or text.startswith("we pass"):
            sys.stdout.write("   ! " + text[:300] + "\n")
            sys.stdout.flush()

    for m in (utils, freebase_func, main_freebase):
        m.print = quiet_print
    main_freebase.tqdm = lambda it, *args, **kwargs: it

    # Record every LLM call (prompt and answer) for the privacy check.
    original_run_llm = utils.run_llm

    def run_llm(prompt, temperature, max_tokens, opeani_api_keys, engine="gpt-3.5-turbo"):
        result, token_num = original_run_llm(prompt, temperature, max_tokens, opeani_api_keys, engine)
        call = {"prompt": prompt, "response": result, "tokens": token_num}
        STATE.calls.append(call)
        STATE.all_calls.append(call)
        return result, token_num

    utils.run_llm = run_llm

    # Trace wrappers around the four modules.
    orig_abs = main_freebase.generate_abs_question
    orig_rel = main_freebase.relation_search_prune
    orig_ann = utils.entity_annotation_all
    orig_rea = main_freebase.reasoning

    def generate_abs_question(question, args):
        out = orig_abs(question, args)
        if STATE.trace:
            say("  Module 2  Concept path written by the LLM (its own guess; used on your side to rank facts):")
            for part in (out[1] or "").split("; "):
                say("              " + part.strip().strip("{}"))
        return out

    def relation_search_prune(entity_id, entity_name, pre_relations, pre_head, question, args):
        out = orig_rel(entity_id, entity_name, pre_relations, pre_head, question, args)
        if STATE.trace:
            if STATE.hop_printed < STATE.hop:
                say(f"  --- Hop {STATE.hop} " + "-" * 60)
                STATE.hop_printed = STATE.hop
            rels = ", ".join(r["relation"] for r in out[0]) or "(none)"
            say(f"  Module 3  LLM picked relations next to '{entity_name}':")
            say(f"              {rels}")
        return out

    def entity_annotation_all(topic_entity_name, question, entity_ids, head_relation, tail_relation,
                              id_description_dict, id_type_dict, id_name_dict, value_entity_dict,
                              value_entity_list, args):
        before = dict(id_type_dict)
        out = orig_ann(topic_entity_name, question, entity_ids, head_relation, tail_relation,
                       id_description_dict, id_type_dict, id_name_dict, value_entity_dict, value_entity_list, args)
        new_type_dict, new_name_dict = out[1], out[2]
        for eid in entity_ids:
            name = new_name_dict.get(eid) or value_entity_dict.get(eid)
            if name:
                STATE.names[eid] = str(name)
            if STATE.trace and new_type_dict.get(eid) != before.get(eid):
                kind = "Hidden value " if eid in value_entity_dict else "Hidden entity"
                say(f"  Module 1  {kind} {eid} described to the LLM as '{new_type_dict.get(eid)}'"
                    f"   (your private map: {name})")
        return out

    def reasoning(question, cluster_chain_of_entities, id_name_dict, current_retrieve_depth, end_flag,
                  call_num, args):
        if STATE.trace and cluster_chain_of_entities:
            say("  Module 3  Evidence kept for this hop (exactly what the LLM receives):")
            for t in cluster_chain_of_entities[-1][:6] or ["(nothing found)"]:
                say("              " + t)
        out = orig_rea(question, cluster_chain_of_entities, id_name_dict, current_retrieve_depth, end_flag,
                       call_num, args)
        if STATE.trace:
            if out[0]:
                ans = str(out[1] or out[2] or "")
                say("  Module 4  Enough evidence? Yes. The LLM answers: " + (ans.split("}")[0] + "}" if "}" in ans else ans[:120]))
            else:
                say("  Module 4  Enough evidence? No, so ARoG takes another hop.")
        STATE.hop += 1
        return out

    main_freebase.generate_abs_question = generate_abs_question
    main_freebase.relation_search_prune = relation_search_prune
    utils.entity_annotation_all = entity_annotation_all
    main_freebase.reasoning = reasoning
    return main_freebase, utils


def output_jsonl(dataset, width=3, depth=3):
    return os.path.join(REPO_DIR, f"ARoG_0720_naive_1_{dataset}_True_True_depth_{depth}_width_{width}_-1.jsonl")


def count_lines(path):
    if not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for _ in f)


def run_one(main_freebase, args, rec, label, question_key, cot_right, trace=True):
    STATE.begin()
    STATE.trace = trace
    question = rec[question_key]
    topic_names = list(rec.get("topic_entity", {}).values())
    path = output_jsonl(args.dataset, args.width, args.depth)
    before = count_lines(path)
    filtered = question not in cot_right
    if trace:
        say(LINE)
        say(f"[{label}] {question}")
        say(f"     Entity named in the question (the LLM may see it): {', '.join(topic_names) or '(none)'}")
        say(f"     Setting: {'filtered (the LLM cannot answer it from memory)' if filtered else 'total only (plain CoT already answers it)'}")
    t0 = time.time()
    main_freebase.task_run([rec], question_key, args, -1)
    elapsed = time.time() - t0

    row = {"id": label, "question": question, "filtered": filtered, "status": "error", "answer": "",
           "gold": [], "calls": len(STATE.calls), "tokens_in": 0, "tokens_out": 0, "seconds": round(elapsed, 1),
           "leaks": [], "prompts": len(STATE.calls), "ids_seen": 0}
    if count_lines(path) > before:
        with open(path, encoding="utf-8") as f:
            rec_out = json.loads(f.readlines()[-1])
        status, answer = score_record(rec_out)
        tok = rec_out.get("token_num") or {}
        row.update(status=status, answer=answer.strip(), gold=rec_out.get("answer_list", []),
                   calls=rec_out.get("call_num", len(STATE.calls)), tokens_in=tok.get("input", 0),
                   tokens_out=tok.get("output", 0))
    # Privacy check: did any hidden name appear in what was sent to the LLM?
    hidden = [n for n in set(STATE.names.values()) if len(n) >= 4 and n not in topic_names
              and not find_names(question, [n])]
    leaks, ids = set(), set()
    for call in STATE.calls:
        part = variable_part(call["prompt"])
        leaks.update(find_names(part, hidden))
        ids.update(re.findall(r"\bm\.[0-9a-z_]+", part))
    row["leaks"], row["ids_seen"] = sorted(leaks), len(ids)

    if trace:
        mark = {"correct": "CORRECT", "wrong": "WRONG", "skipped": "SKIPPED", "error": "NO ANSWER (see log)"}[row["status"]]
        say(f"  Result    After your private ID-to-name map: {row['answer'] or '-'}   -> {mark}")
        say(f"            Benchmark answer: {', '.join(row['gold']) or '-'}")
        say(f"            LLM calls: {row['calls']} | tokens: {row['tokens_in'] + row['tokens_out']:,} | "
            f"time: {elapsed:.0f} s | cost: ${money(row['tokens_in'], row['tokens_out']):.4f}")
        say(f"  Privacy   {row['prompts']} prompts sent to the LLM; they contain {row['ids_seen']} hidden IDs and "
            + ("no hidden names." if not row["leaks"] else f"these hidden names: {', '.join(row['leaks'])}"))
    else:
        say(f"[{label}] {'OK ' if row['status'] == 'correct' else row['status'].upper()[:5]:5} "
            f"{'F' if filtered else 'T'}  {question[:60]}")
    return row


def summarise(rows, run_dir, title):
    say("")
    say("=" * 78)
    say(title)
    say("=" * 78)
    say(f"{'ID':<5}{'Setting':<10}{'Result':<9}{'Calls':>6}{'Tokens':>9}{'Hidden names leaked':>22}")
    for r in rows:
        say(f"{r['id']:<5}{('filtered' if r['filtered'] else 'total'):<10}{r['status']:<9}{r['calls']:>6}"
            f"{r['tokens_in'] + r['tokens_out']:>9,}{(', '.join(r['leaks']) or 'none'):>22}")
    scored = [r for r in rows if r["status"] in ("correct", "wrong")]
    right = sum(r["status"] == "correct" for r in scored)
    filt = [r for r in scored if r["filtered"]]
    tin, tout = sum(r["tokens_in"] for r in rows), sum(r["tokens_out"] for r in rows)
    say(LINE)
    say(f"Correct: {right} of {len(scored)}"
        + (f"   (filtered questions: {sum(r['status'] == 'correct' for r in filt)} of {len(filt)})" if filt else ""))
    say(f"LLM calls: {sum(r['calls'] for r in rows)} | tokens: {tin + tout:,} | estimated cost: ${money(tin, tout):.3f}")
    say(f"Privacy check: hidden names found in prompts for {sum(bool(r['leaks']) for r in rows)} of {len(rows)} questions.")
    say(f"Saved in: {run_dir}")
    with open(os.path.join(run_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    return right, len(scored)


def finish_run(run_dir):
    for f in arog_output_files():
        shutil.move(f, os.path.join(run_dir, os.path.basename(f)))
    with open(os.path.join(run_dir, "llm_calls.jsonl"), "w", encoding="utf-8") as f:
        for c in STATE.all_calls:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    with open(os.path.join(run_dir, "transcript.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(TRANSCRIPT) + "\n")


def load_cot_right(dataset):
    try:
        with open(os.path.join(REPO_DIR, COT_RIGHT[dataset]), encoding="utf-8") as f:
            return set(json.load(f))
    except Exception:
        return set()


# --------------------------------------------------------------------------------------------------
# run (demo questions on the mini knowledge graph)
# --------------------------------------------------------------------------------------------------
def cmd_run(a):
    key = read_key(a.key)
    if not key:
        say("No OpenAI key found. Paste it into demo_kit\\openai_key.txt, then run:  python demo_kit\\arog_demo.py check")
        return 1
    with open(os.path.join(KIT_DIR, "demo_questions.json"), encoding="utf-8") as f:
        demo = json.load(f)
    only = {x.strip().upper() for x in a.only.split(",")} if a.only else None
    run_dir = new_run_dir("demo")
    move_old_outputs()
    main_freebase, _ = load_arog(a, run_dir)
    random.seed(12345)
    rows = []
    try:
        for dataset in ("webqsp", "cwq"):
            cot_right = load_cot_right(dataset)
            for rec in demo[dataset]:
                if only and rec["demo_id"].upper() not in only:
                    continue
                args = arog_args(dataset, a.model, key)
                rows.append(run_one(main_freebase, args, rec, rec["demo_id"], QUESTION_KEY[dataset], cot_right))
    except KeyboardInterrupt:
        say("\nStopped by you (Ctrl+C). Saving what was finished.")
    if not rows:
        say("No question matched --only. Use IDs from:  W1 W2 W3 W4 C1 C2 C3 C4")
    else:
        summarise(rows, run_dir, "ARoG demo on the mini knowledge graph")
        say("Note: hand-picked questions on a small graph built from benchmark data. This demonstrates the "
            "method; it does not reproduce the paper's accuracy, which needs the full Freebase.")
    finish_run(run_dir)
    return 0


# --------------------------------------------------------------------------------------------------
# benchmark (real questions on a full Freebase server)
# --------------------------------------------------------------------------------------------------
def cmd_benchmark(a):
    key = read_key(a.key)
    if not key:
        say("No OpenAI key found. Paste it into demo_kit\\openai_key.txt first.")
        return 1
    if "127.0.0.1:8890" in a.sparql or "localhost:8890" in a.sparql:
        say("Warning: this looks like the mini knowledge graph. Benchmark questions need the full Freebase,")
        say("so most of them will fail here. Pass the address of a Freebase server with --sparql.")
    per_q = 0.0016
    say(f"Running {a.limit} {a.dataset} questions. Estimated cost: about ${a.limit * per_q:.2f}; "
        f"time: about {a.limit * 0.5:.0f} to {a.limit * 1.5:.0f} minutes.")
    run_dir = new_run_dir(f"benchmark_{a.dataset}")
    move_old_outputs()
    main_freebase, utils = load_arog(a, run_dir)
    datas, question_key = utils.prepare_dataset(a.dataset)
    random.seed(12345)                     # the same shuffling as main_freebase.py
    if a.dataset == "webqsp":
        datas = random.sample(datas, len(datas))
    else:
        datas = random.sample(datas, 1000)
    datas = datas[a.start:a.start + a.limit]
    cot_right = load_cot_right(a.dataset)
    rows = []
    try:
        for i, rec in enumerate(datas, a.start + 1):
            args = arog_args(a.dataset, a.model, key)
            rows.append(run_one(main_freebase, args, rec, str(i), question_key, cot_right, trace=a.trace))
    except KeyboardInterrupt:
        say("\nStopped by you (Ctrl+C). Scoring what was finished.")
    if rows:
        right, n = summarise(rows, run_dir, f"ARoG on {len(rows)} {a.dataset} questions (full Freebase)")
        filt = [r for r in rows if r["filtered"] and r["status"] in ("correct", "wrong")]
        if n:
            p = right / n
            margin = 1.96 * math.sqrt(max(p * (1 - p), 0.1) / n) * 100
            note = f"plus or minus about {margin:.0f} points with {n} questions"
            if n < 30:
                note += "; use 30 or more questions for a number worth quoting"
            say(f"Hits@1, total setting:    {100 * p:.1f}%  ({note})")
        if filt:
            pf = sum(r["status"] == "correct" for r in filt) / len(filt)
            say(f"Hits@1, filtered setting: {100 * pf:.1f}%  ({len(filt)} question{'s' if len(filt) != 1 else ''})")
        if a.dataset in PAPER:
            say(f"Paper (Table 2):          {PAPER[a.dataset][0]}% total, {PAPER[a.dataset][1]}% filtered")
    finish_run(run_dir)
    return 0


# --------------------------------------------------------------------------------------------------
# show and kg
# --------------------------------------------------------------------------------------------------
def cmd_show(a):
    runs = sorted(glob.glob(os.path.join(OUT_ROOT, "run_*")))
    if not runs:
        print("No runs saved yet. Run:  python demo_kit\\arog_demo.py run")
        return 1
    if a.run == "list":
        print("Saved runs (oldest first). Show one with:  python demo_kit\\arog_demo.py show --run <name>")
        for r in runs:
            print("  " + os.path.basename(r))
        return 0
    run_dir = runs[-1] if a.run == "latest" else next((r for r in runs if a.run in os.path.basename(r)), None)
    if not run_dir:
        print("Run not found. Saved runs:\n  " + "\n  ".join(os.path.basename(r) for r in runs))
        return 1
    path = os.path.join(run_dir, "transcript.txt")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if a.summary and "=" * 78 in text:
        text = text[text.find("=" * 78):]
    print(text)
    print(f"(from {run_dir})")
    if a.open and os.name == "nt":
        os.startfile(run_dir)
    return 0


def cmd_kg(a):
    with open(os.path.join(KIT_DIR, "mini_kg", "facts_readable.txt"), encoding="utf-8") as f:
        lines = f.read().splitlines()
    if a.grep:
        lines = lines[:3] + [l for l in lines[3:] if a.grep.lower() in l.lower()]
    print("\n".join(lines))
    return 0


def main():
    p = argparse.ArgumentParser(description="ARoG demo launcher. See the guide for step-by-step use.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--sparql", default="http://127.0.0.1:8890/sparql", help="SPARQL endpoint (knowledge graph)")
        sp.add_argument("--embed", default="http://127.0.0.1:5001", help="address of lm_server.py")
        sp.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model (paper: gpt-4o-mini-2024-07-18)")
        sp.add_argument("--key", default=None, help="OpenAI key (normally read from demo_kit\\openai_key.txt)")

    sp = sub.add_parser("check", help="check that everything is ready")
    common(sp)
    sp = sub.add_parser("run", help="run the demo questions")
    common(sp)
    sp.add_argument("--only", default=None, help="comma-separated IDs, e.g. C1 or W1,C1")
    sp = sub.add_parser("benchmark", help="run benchmark questions on a full Freebase server")
    common(sp)
    sp.add_argument("--dataset", choices=["webqsp", "cwq", "grailqa"], default="webqsp")
    sp.add_argument("--limit", type=int, default=20)
    sp.add_argument("--start", type=int, default=0)
    sp.add_argument("--trace", action="store_true", help="show the full step-by-step trace")
    sp = sub.add_parser("show", help="show a saved run")
    sp.add_argument("--run", default="latest", help="latest, list, or part of a run folder name")
    sp.add_argument("--summary", action="store_true", help="only the summary table")
    sp.add_argument("--open", action="store_true", help="open the run folder in Explorer")
    sp = sub.add_parser("kg", help="print the mini knowledge graph")
    sp.add_argument("--grep", default=None, help="only lines containing this text, e.g. Egypt")

    a = p.parse_args()
    if getattr(a, "model", None) and "gpt" not in a.model.lower():
        # utils.run_llm only calls OpenAI when the model name contains "gpt"; otherwise it returns nothing.
        print(f"The authors' code only sends a model to OpenAI if its name contains 'gpt' ('{a.model}' does not).\n"
              "Pick a model whose name contains gpt, for example --model gpt-4o-mini.")
        return 1
    try:
        return {"check": cmd_check, "run": cmd_run, "benchmark": cmd_benchmark, "show": cmd_show,
                "kg": cmd_kg}[a.cmd](a)
    except Exception:
        traceback.print_exc()
        print("\nSomething went wrong. The guide's troubleshooting table covers the common errors.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
