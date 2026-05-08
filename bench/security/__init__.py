"""Security benchmark — direct prompt-injection evaluation.

Three configurations are compared:

* `aether_taint_on`   — `aetherc check` with Pass 6 enabled. Vulnerable
  programs (untrusted reaches prompt without sanitize) are rejected at
  compile time and counted as caught pre-deploy. ASR == 0 by construction.
* `aether_taint_off`  — `aetherc check --no-taint-check`. Same Aether
  toolchain, Pass 6 disabled. Programs that would have been rejected
  compile clean and run against the live LLM. Isolates the contribution
  of taint tracking.
* `langchain_baseline` — Equivalent LangChain LCEL pipeline. No taint
  tracking by design. Provides the absolute reference point.

Attack content is adapted from InjecAgent (arXiv:2403.02691, MIT).
See `data/_make_subset.py` for provenance.
"""
