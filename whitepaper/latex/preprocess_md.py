#!/usr/bin/env python3
"""
whitepaper/latex/preprocess_md.py

Convert WHITEPAPER_ACADEMIC.md (numeric markdown references [1]..[23] and
a hand-written references list at the bottom) into a pandoc-ready stream
where:

  - Inline references like `[7]` become pandoc-citeproc references
    `[@ref7]`. Citation keys mirror the original numbers and are defined
    in whitepaper/latex/references.bib.
  - The hand-written `## References` body is replaced by a pandoc
    citeproc placeholder (`::: {#refs}\n:::`), which pandoc fills with
    the typeset bibliography.
  - Code blocks are NOT touched (so EBNF/pseudocode containing `[1]`
    survives).

The source markdown is READ-ONLY; this script writes its output to
stdout (or a file with -o). This keeps the whitepaper canonical and
makes the LaTeX path purely additive — hard rule #2 (no whitepaper
edits except those that don't change measurements) is preserved.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


CITE_RE = re.compile(r"\[(\d+)\]")

# Body-text Unicode glyphs that the default Latin Modern Roman font does
# not include. We translate them to inline math at preprocess time so
# the PDF renders proper glyphs from the math font rather than tofu.
BODY_TEXT_MATH = {
    "≈": r"$\approx$",
    "≠": r"$\neq$",
    "≤": r"$\leq$",
    "≥": r"$\geq$",
}

# Code-block character translation for the typing-rule blocks. Listings
# / fancyvrb under XeTeX cannot apply per-character literate substitution
# reliably for arbitrary Unicode, so we transform the source at the markdown
# level into a math-display environment that Pandoc passes through verbatim.
def _typing_rule_to_latex(body: str) -> str:
    """
    Convert a fenced code block whose content is a typing-style rule
    (recognised by ``Γ ⊢`` and ``─`` characters) into a raw-LaTeX
    centred math display, replacing the horizontal rule of dashes with
    ``\\rule{0.7\\linewidth}{0.4pt}``.
    """
    lines = [ln for ln in body.splitlines() if ln.strip()]
    # split into premises (above ─), separator, conclusion (below)
    above: list[str] = []
    below: list[str] = []
    in_below = False
    for ln in lines:
        if set(ln.strip()) <= {"─"}:
            in_below = True
            continue
        (below if in_below else above).append(ln)

    def to_math(ln: str) -> str:
        # Convert ascii-art math notation into LaTeX inline math syntax.
        s = ln
        s = s.replace("Γ", r"\Gamma ")
        s = s.replace("⊢", r"\vdash ")
        s = s.replace("τ", r"\tau ")
        s = s.replace("α", r"\alpha ")
        s = s.replace("β", r"\beta ")
        s = s.replace("γ", r"\gamma ")
        s = s.replace("ε", r"\varepsilon ")
        s = s.replace("→", r"\rightarrow ")
        s = s.replace("↦", r"\mapsto ")
        s = s.replace("⊥", r"\bot ")
        s = s.replace("∈", r"\in ")
        s = s.replace("⊕", r"\oplus ")
        s = s.replace("≤", r"\leq ")
        s = s.replace("≥", r"\geq ")
        s = s.replace("≠", r"\neq ")
        s = s.replace("≈", r"\approx ")
        s = s.replace("∀", r"\forall ")
        s = s.replace("∃", r"\exists ")
        # Subscripts: τ₁, eᵢ, etc.
        for sub_src, sub_dst in [
            ("₀", "_0"), ("₁", "_1"), ("₂", "_2"), ("₃", "_3"),
            ("₄", "_4"), ("₅", "_5"), ("ᵢ", "_i"), ("ₙ", "_n"),
        ]:
            s = s.replace(sub_src, sub_dst)
        s = s.replace("...", r"\ldots ")
        # Wrap each whitespace-separated chunk in \text{} when it's pure
        # alphabetic to keep upright; the above replacements have already
        # turned Greek letters into TeX commands so a chunk like
        # ``f(x_1: e_1, ..., x_n: e_n)`` stays mostly intact in math mode.
        # Escape some characters that would otherwise misparse in math.
        s = s.replace("{", r"\{").replace("}", r"\}")
        return s

    out = ["", "\\begin{center}"]
    out.append(" \\quad ".join(f"$ {to_math(p)} $" for p in above))
    out.append("\\\\[2pt]")
    out.append("\\rule{0.7\\linewidth}{0.4pt}")
    out.append("\\\\[2pt]")
    out.append(" \\quad ".join(f"$ {to_math(c)} $" for c in below))
    out.append("\\end{center}")
    out.append("")
    return "\n".join(out)


def _is_typing_rule_block(body: str) -> bool:
    return ("Γ" in body) and ("⊢" in body) and ("─" in body)


def preprocess(text: str) -> str:
    out: list[str] = []
    in_fence = False
    fence_marker = ""
    in_refs_section = False

    lines = text.splitlines(keepends=False)
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()

        # Track fenced code blocks (``` or ~~~ openers / closers).
        if not in_fence and (stripped.startswith("```") or stripped.startswith("~~~")):
            fence_marker = stripped[:3]
            # Capture the fenced block body up to its closer so we can
            # decide whether to emit a typing-rule replacement instead.
            j = i + 1
            body_lines: list[str] = []
            while j < len(lines):
                if lines[j].lstrip().startswith(fence_marker):
                    break
                body_lines.append(lines[j])
                j += 1
            block_body = "\n".join(body_lines)
            if _is_typing_rule_block(block_body):
                out.append(_typing_rule_to_latex(block_body))
            else:
                out.append(line)
                out.extend(body_lines)
                if j < len(lines):
                    out.append(lines[j])
            i = j + 1 if j < len(lines) else j
            fence_marker = ""
            continue

        # Detect the start of the manual References section.
        if line.strip() == "## References" and not in_refs_section:
            in_refs_section = True
            out.append(line)
            out.append("")
            out.append("::: {#refs}")
            out.append(":::")
            out.append("")
            i += 1
            continue

        # Inside the References section, swallow lines until we hit the
        # next top-level horizontal rule (---) or the next `##` heading
        # (Appendix A in the source). Re-emit that delimiter line.
        if in_refs_section:
            if line.strip() == "---" or line.startswith("## "):
                in_refs_section = False
                out.append(line)
            # else: drop the line silently
            i += 1
            continue

        # Substitute `[N]` -> `[@refN]` everywhere outside code/refs.
        rewritten = CITE_RE.sub(r"[@ref\1]", line)
        # Body-text Unicode that the upright text font lacks; render as
        # inline math so the math font's glyph is used.
        for src, dst in BODY_TEXT_MATH.items():
            rewritten = rewritten.replace(src, dst)
        out.append(rewritten)
        i += 1

    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", help="Path to WHITEPAPER_ACADEMIC.md")
    ap.add_argument("-o", "--output", default="-",
                    help="Output path (default: stdout)")
    args = ap.parse_args()

    text = Path(args.input).read_text(encoding="utf-8")
    rewritten = preprocess(text)

    if args.output == "-":
        sys.stdout.write(rewritten)
    else:
        Path(args.output).write_text(rewritten, encoding="utf-8")


if __name__ == "__main__":
    main()
