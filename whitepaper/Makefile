# whitepaper/Makefile
#
# Build the academic LaTeX/PDF artefacts from WHITEPAPER_ACADEMIC.md.
#
# Targets:
#   make tex     -> regenerate latex/aether.tex from the markdown source
#   make pdf     -> build latex/aether.pdf via tectonic (auto-fetches packages)
#   make clean   -> remove generated .tex/.pdf and tectonic auxiliary files
#
# Toolchain:
#   pandoc   3.x  (https://pandoc.org/)         — markdown -> LaTeX
#   tectonic 0.16+ (https://tectonic-typesetting.github.io/) — LaTeX -> PDF
#
# Both are documented in REPRODUCIBILITY.md §2 and §8. Tectonic's first
# run downloads ~600MB of TeX Live packages on demand; subsequent builds
# reuse the cache (Windows: %LOCALAPPDATA%\TectonicProject; macOS/Linux:
# ~/.cache/Tectonic).
#
# This Makefile assumes it's invoked from the repo root as:
#   make -C whitepaper pdf
# (i.e. -C cd's into whitepaper/ first). Paths below are relative to
# whitepaper/.

SRC      := WHITEPAPER_ACADEMIC.md
BIB      := latex/references.bib
HEADER   := latex/header.tex
PREP     := latex/preprocess_md.py
INTER    := latex/aether.preprocessed.md
TEX      := latex/aether.tex
PDF      := latex/aether.pdf
PYTHON   ?= python

PANDOC_FLAGS = \
  --from=markdown+yaml_metadata_block+tex_math_dollars+raw_tex+pipe_tables \
  --to=latex \
  --listings \
  -V documentclass=article \
  -V geometry= \
  --include-in-header=$(HEADER) \
  --bibliography=$(BIB) \
  --citeproc \
  --standalone

.PHONY: all tex pdf clean
all: pdf

tex: $(TEX)

pdf: $(PDF)

$(INTER): $(SRC) $(PREP)
	$(PYTHON) $(PREP) $(SRC) -o $(INTER)

$(TEX): $(INTER) $(BIB) $(HEADER)
	pandoc $(INTER) $(PANDOC_FLAGS) -o $(TEX)

$(PDF): $(TEX) latex/arxiv.sty
	cd latex && tectonic --keep-logs --keep-intermediates aether.tex

clean:
	rm -f $(INTER) $(TEX) $(PDF)
	rm -f latex/aether.aux latex/aether.log latex/aether.out latex/aether.toc
	rm -f latex/aether.fdb_latexmk latex/aether.fls latex/aether.synctex.gz
