# NKI optimization heuristics

A rule database for agents (and humans) writing or rewriting NKI kernels on Trainium2: what to do,
why, how to detect a violation, and where the rule came from. In this repo it is the natural input for
the "propose a faster rewrite" side of the optimization loop that `nkiequiv` checks.

| file | what |
|---|---|
| [`NKI_HEURISTICS.md`](NKI_HEURISTICS.md) | curated read: regime decision, where custom kernels pay off, DMA, SBUF/PSUM, matmul, engines, pipelining, LNC=2, numerics, API migration, profile reading |
| [`APPENDIX.md`](APPENDIX.md) | every rule, grouped by category, with source and confidence |
| [`rules.json`](rules.json) | 724 rules, one schema: `id, category, rule, why, trigger, fix, detectable, check_sketch, numbers, source, confidence, generation, tier` |
| [`cards/`](cards/) | one short card per category, for retrieval into a prompt |
| [`heuristics.py`](heuristics.py) | `card()`, `retrieve(query, k)`, `for_error(text)`, `instructions_for_error(text)` |
| [`nki_constants.json`](nki_constants.json) | 70 numeric hardware/tile constants with units and sources |
| [`sources/`](sources/) | the five per-source extraction passes the merge consumes |
| [`merge_heuristics.py`](merge_heuristics.py) | rebuilds `rules.json`, `APPENDIX.md` and `cards/` from `sources/` |

Sources, extracted 2026-10-10 against NKI 0.6.0: the NKI documentation tree (architecture guide,
performance guide, deep dives, tutorials, NKI Library docs), docstrings of the installed `nki` package
and the migration guide, the `aws-neuron/nki-samples` repo, the installed `nkilib` production kernels,
and the agent skill files AWS ships in `neuron-agentic-development`. Every record cites its source;
`confidence: inferred` marks rules derived rather than stated.

```bash
python3 heuristics/heuristics.py "psum accumulate across K tiles"     # card + top-5 retrieval
python3 heuristics/merge_heuristics.py heuristics/sources heuristics  # rebuild derived files
```

No dependencies beyond the standard library.
