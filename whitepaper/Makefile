# whitepaper/Makefile
#
# Single-target docx build. The pandoc/docx pipeline replaces the prior
# LaTeX/PDF pipeline (archived under whitepaper/archive/latex/).
#
# Usage (from repo root or anywhere -- recipes run from whitepaper/):
#   make -C whitepaper docx
#   make -C whitepaper clean
#
# Toolchain:
#   pandoc 3.x (https://pandoc.org/) -- markdown -> docx
#
# Pandoc bundles skylighting for code highlighting; the tango style is
# used because it provides high-contrast colour for fn/let/match/struct
# tokens which is what readers care about for the rust-fenced Aether
# code in this paper.

.PHONY: docx clean

docx: aether.docx

aether.docx: WHITEPAPER_ACADEMIC.md metadata.yml reference.docx
	pandoc WHITEPAPER_ACADEMIC.md \
	  --from=markdown+pipe_tables+grid_tables+raw_html \
	  --to=docx \
	  --reference-doc=reference.docx \
	  --resource-path=.:figures \
	  --syntax-highlighting=tango \
	  --metadata-file=metadata.yml \
	  --toc --toc-depth=2 \
	  -o aether.docx

clean:
	rm -f aether.docx
