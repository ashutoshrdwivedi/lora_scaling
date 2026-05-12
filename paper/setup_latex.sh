#!/usr/bin/env bash
# =============================================================================
# setup_latex.sh  —  install all LaTeX packages needed to compile main.tex
#
# Run AFTER BasicTeX is installed:
#   bash paper/setup_latex.sh
#
# Requires sudo (BasicTeX's tlmgr lives in /Library/TeX/texbin/).
# =============================================================================
set -euo pipefail

TLMGR="/Library/TeX/texbin/tlmgr"

if [[ ! -x "$TLMGR" ]]; then
  echo "ERROR: tlmgr not found at $TLMGR"
  echo "  Install BasicTeX first:  brew install --cask basictex"
  echo "  Then reload your PATH:   eval \"\$(/usr/libexec/path_helper)\""
  exit 1
fi

echo "==> Updating tlmgr itself..."
sudo "$TLMGR" update --self

echo "==> Installing required packages..."
sudo "$TLMGR" install \
  latexmk \
  collection-latexrecommended \
  natbib \
  xcolor \
  geometry \
  caption \
  hyperref \
  url \
  microtype \
  inconsolata \
  booktabs \
  multirow \
  todonotes \
  lineno \
  etoolbox \
  algorithms \
  algorithmicx \
  amsmath \
  # algorithms/algorithmicx kept for when pseudocode is added
  times \
  helvetic

echo ""
echo "==> Done. Verifying pdflatex..."
pdflatex --version | head -1

echo ""
echo "All set. Compile the paper with:"
echo "  cd paper && make"
echo "  (or: make quick  for a single-pass draft check)"
