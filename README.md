# ARoG Reproduction

Reproduction of **ARoG** (Abstraction Reasoning on Graph) from:

> Ning et al. *Privacy-protected Retrieval-Augmented Generation for Knowledge Graph Question Answering.* AAAI 2026. [arXiv:2508.08785](https://arxiv.org/abs/2508.08785)

Original code: [github.com/NLPGM/ARoG](https://github.com/NLPGM/ARoG) (commit `4867cf5`, 10 Nov 2025)

Task 2 of the KGCR-RAG project group, WiSe 2026/27.

## What ARoG does

ARoG answers questions over a knowledge graph with an LLM, but the LLM never sees the names of the entities it retrieves, only their Freebase IDs (for example `m.01mpq7s`). ARoG has four steps:

1. **Concept path:** the LLM turns the question into an abstract path, for example `artist -> has daughter -> person`.
2. **Retrieval:** for each hop, the LLM picks up to 3 relations, and a SentenceTransformer ranks the facts that come back.
3. **Entity abstraction:** each hidden entity is described by its relations, and the LLM gives it a type, for example `m.01mpq7s (musical artist)`.
4. **Generator:** the LLM decides whether there is enough evidence. If there is, it answers with IDs, which are mapped back to names on the user's side.

## Repository contents

```
ARoG-reproduction/
├── README.md
├── demo_kit/                    launcher and local knowledge graph
│   ├── arog_demo.py             runs the original ARoG code with the runtime fixes
│   ├── mini_kg_server.py        local SPARQL endpoint (Freebase format)
│   ├── mini_kg/                 the subgraph: 69 entities, 226 triples
│   ├── demo_questions.json      the 8 evaluated questions (4 WebQSP, 4 CWQ)
│   ├── build_mini_kg.py         script that built the subgraph from the benchmark files
│   └── requirements_windows.txt
├── results/
│   ├── run1/                    results.json, transcript.txt, LLM prompts, ARoG output
│   └── run2/
└── report/
    └── ARoG_Reproduction_Report.pdf
```

The original ARoG code is **not** copied here. Get it from the authors' repository and put `demo_kit/` inside it, next to `main_freebase.py`.

## Setup

Tested on Windows with Python 3.10 (Miniconda) and 8 GB RAM. No GPU is needed.

```bash
git clone https://github.com/NLPGM/ARoG.git
cd ARoG
# copy the demo_kit folder from this branch into this folder
conda create -n arog python=3.10 -c conda-forge --override-channels -y
conda activate arog
pip install -r demo_kit/requirements_windows.txt
```

Put your OpenAI API key in `demo_kit/openai_key.txt`, or set the `OPENAI_API_KEY` environment variable. Never commit the key.

## How to run

Open three terminals, all in the ARoG folder with the `arog` environment active:

```bash
python demo_kit/mini_kg_server.py        # terminal 1: knowledge graph on port 8890
python lm_server.py                      # terminal 2: SentenceTransformer on port 5001
python demo_kit/arog_demo.py check       # terminal 3: check that everything works
python demo_kit/arog_demo.py run         # run all 8 questions
python demo_kit/arog_demo.py show        # show the latest run again
```

To run against a full Freebase SPARQL endpoint instead:

```bash
python demo_kit/arog_demo.py benchmark --sparql http://<server>:8890/sparql --dataset webqsp --limit 30
```

## Settings

These are the same as the paper: `gpt-4o-mini-2024-07-18`, width 3, depth 3, temperature 0 for reasoning and 0.4 for exploration.

## Results

The evaluation used 8 questions, 2 runs, and the local subgraph. Metric: Hits@1, scored with the authors' matching rule.

| | Run 1 | Run 2 |
|---|---|---|
| Correct (all 8) | 8/8 | 7/8 |
| Correct (6 filtered questions) | 6/6 | 5/6 |
| LLM calls | 77 | 76 |
| Hidden entity names found in prompts | 0 | 0 |

The paper reports these figures on the full Freebase (Total / Filtered): WebQSP 74.7 / 58.9, CWQ 60.0 / 36.3, GrailQA 78.7 / 71.8. Our numbers are not comparable, because we used 8 questions and a small subgraph.

## Issues found in the original code

The launcher fixes these at runtime, without editing the authors' files:

- OpenAI calls go through a proxy (`socks5://127.0.0.1:7891`) that only existed on the authors' machine.
- The SPARQL and SentenceTransformer addresses are placeholders.
- `utils.py` expects port 5000, but `lm_server.py` runs on port 5001.
- `requirements.txt` includes Linux-only packages (vllm, ray) and is missing `httpx-socks`, `nltk`, `requests` and `tqdm`.
- The evaluation command in the README uses the wrong output file name.
- `eval.py` only scores questions that plain chain-of-thought already answers.

## Key findings and limitations

- **Type errors propagate.** In W1 the correct entity was retrieved, but it was typed as a "geographic location", so it was lost and the LLM answered from memory instead.
- **The scoring rule is lenient.** The same memory answer was scored correct in one run and wrong in the other, depending only on answer format.
- **Privacy covers names only.** The question, the relation names and the LLM's own knowledge still reveal information, and Freebase IDs are public.
- **Fallback is silent.** When the search finds nothing, the code answers from the LLM's memory without flagging it.
- **Reproducibility.** Full Freebase needs about 400 GB of disk, and `gpt-4o-mini-2024-07-18` shuts down on 14 January 2027.

See `report/ARoG_Reproduction_Report.pdf` for details and proposed fixes.

## Author

Ahmed, KGCR-RAG Project Group, Paderborn University
