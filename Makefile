# Paper build pipeline. Generated artifacts (numbers.tex, table_*.tex,
# figures/*.pdf) are committed so the paper compiles standalone (Overleaf,
# arXiv, collaborators without the Python toolchain); `make check` fails if
# they are stale relative to the benchmark CSVs.

.PHONY: paper numbers figures check clean

paper: numbers figures
	cd paper && latexmk -pdf -interaction=nonstopmode main.tex

numbers:
	uv run python paper/build_numbers.py

figures:
	uv run --extra paper python paper/build_figures.py

check: numbers figures
	git diff --exit-code paper/numbers.tex paper/table_main.tex \
		paper/table_baselines.tex paper/table_accuracy.tex paper/figures/

clean:
	cd paper && latexmk -C
