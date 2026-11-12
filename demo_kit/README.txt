ARoG demo kit
=============

This folder helps you run the ARoG code (Ning et al., AAAI 2026) on a normal Windows laptop and show it
working in a group meeting. Follow the step-by-step guide; this file only explains what is in the folder.

Where it goes
-------------
Put the whole demo_kit folder inside the ARoG code folder, next to main_freebase.py:

    ARoG-main\
        main_freebase.py, utils.py, freebase_func.py, lm_server.py, data\, CoT\, ...   (authors' files)
        demo_kit\                                                                     (this folder)

Nothing in the authors' files is changed. Every command is run from the ARoG-main folder.

What is in this folder
----------------------
arog_demo.py              The launcher. It loads the authors' code as it is and runs it with three fixes
                          applied in memory (see below). Commands:
                              python demo_kit\arog_demo.py check
                              python demo_kit\arog_demo.py run --only C1
                              python demo_kit\arog_demo.py run
                              python demo_kit\arog_demo.py show
                              python demo_kit\arog_demo.py kg
                              python demo_kit\arog_demo.py benchmark --sparql http://SERVER:8890/sparql --dataset webqsp --limit 30
mini_kg_server.py         A small SPARQL server that stands in for Freebase. It answers the same queries
                          ARoG sends to Freebase, at http://127.0.0.1:8890/sparql.
mini_kg\mini_freebase.nt  The mini knowledge graph (69 entities, 156 facts) in Freebase format.
mini_kg\facts_readable.txt The same facts in plain English-like lines, for showing on screen.
demo_questions.json       The 8 demo questions (W1-W4 from WebQSP, C1-C4 from CWQ), taken unchanged from
                          the authors' data files.
build_mini_kg.py          The script that built the three files above (you do not need to run it).
requirements_windows.txt  The Python packages to install on Windows.
openai_key.txt            Paste your OpenAI API key here. Never share this file or put it on GitHub.
outputs\                  Created on the first run. One folder per run, with:
                              transcript.txt     everything printed on screen
                              results.json       the score table
                              llm_calls.jsonl    every prompt sent to OpenAI and every reply
                              ARoG_0720_*.jsonl  the authors' own output file for that run
                              arog_log.txt, arog_console_log.txt   the authors' log output

The three fixes the launcher applies while the code runs
--------------------------------------------------------
1. OpenAI connection. utils.py sends every OpenAI call through a proxy at socks5://127.0.0.1:7891 that only
   existed on the authors' computer. Without the fix every call fails and run_llm retries forever.
2. Server addresses. freebase_func.py and utils.py contain placeholders ("http://xxx.xxx.xxx.xxx:8890/sparql"
   and "http://{{SentenceTransformer Server}}:5000/..."). The launcher fills in the real addresses.
3. Port number. lm_server.py listens on port 5001, but utils.py says 5000. The launcher uses 5001.

Where the mini knowledge graph comes from
-----------------------------------------
The facts are the gold reasoning paths written into the WebQSP and CWQ benchmark files that ship with the
code (data\WebQSP.json, data\cwq_1000.json), plus their reverse relations, entity names and a few facts
from Figure 2 of the paper (the Beyonce example). Facts are tagged [WebQSP], [CWQ], [Figure2] or [reverse]
in facts_readable.txt. No answer was invented.

What the demo shows and what it does not
----------------------------------------
It shows the real ARoG pipeline (all four modules, real GPT-4o-mini calls, the paper's settings: width 3,
depth 3, temperature 0 and 0.4) answering real benchmark questions while the LLM only sees entity IDs.
It does not reproduce the paper's accuracy numbers. Those need the full Freebase (about 125 GB filtered,
400 GB unpacked) on a server, which the "benchmark" command can use if you get access to one.
