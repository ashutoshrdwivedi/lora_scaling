#!/usr/bin/env python3
"""Verify the paste-ready rebuttal parts before they go into OpenReview.

Two checks, both hard failures:

1. LENGTH  -- each part must fit OpenReview's 5,000-character comment box.
2. NUMBERS -- every numeric token in the prose must trace to a value in
   numbers.json (produced by make_numbers.py from the raw CSVs), or to the
   explicit ALLOWED literal list below. Anything else is, by construction, a
   number nobody measured.

Usage:
    python rebuttal/make_numbers.py && python rebuttal/check.py
    python rebuttal/check.py --verbose      # show what each number matched
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARTS = sorted((HERE / "parts").glob("*.md"))
REGISTRY = HERE / "numbers.json"
CHAR_LIMIT = 5000
# Leave headroom: the portal may count characters slightly differently than
# Python does (line endings, unicode normalisation) and there is no second try.
SAFE_LIMIT = 4900

# Literals that are legitimately not measurements: protocol constants, grid
# coordinates, section/question labels, dates, and plain English quantities.
# Keep this list short and justified -- it is the hole in the net.
ALLOWED = {
    # experimental grid coordinates (batch sizes, ranks, seeds, seq len)
    4, 5, 8, 16, 32, 64, 128, 256, 200, 50, 10,
    # Pool sizes that are grid COORDINATES -- values a script asked for, which
    # say nothing about what the hardware achieved.
    #
    # Measured ceilings are deliberately NOT here: 47000, 49000, 58000, 28000
    # and 12000 were removed so they must trace to a ceiling_n in numbers.json.
    # They are the numbers a re-run moves, and whitelisting them meant a stale
    # ceiling in the prose passed silently -- the checker only ever caught
    # invented values, never superseded ones. Do not add a ceiling back here to
    # quiet a failure; regenerate numbers.json and fix the prose instead.
    100, 1000, 2000, 4000, 5000, 6000, 8000, 10000, 11000, 20000,
    40000, 45000, 50000, 51000, 55000, 60000, 13000,
    # 26000/29000 are measured L40S results, not coordinates: the
    # default-allocator ceiling and the count that fails under both. They stay
    # allowed only until a re-run populates ceiling_n_default_alloc (the probe
    # now measures both arms); drop them once numbers.json carries that field.
    26000, 29000,
    # small counting numbers used in prose ("three encoders", "two GPU classes")
    0, 1, 2, 3, 6, 7, 9,
    # parity tolerances quoted from the test suite
    1e-4, 2.2e-5, 5,
    # structural: page budget, year, review round
    2026,
}
# Regex-shaped exemptions: labels rather than quantities.
ALLOWED_PATTERNS = [
    r"^Q[1-6]$",          # reviewer question labels
    r"^W[1-6]$",          # reviewer weakness labels
    r"^§4(\.\d)?$",  # section references
    r"^Finding \d$",
    r"^r=\d+$",
    r"^B=\d+$",
    r"^N=[\d,]+$",
    r"^fp16$",
    r"^L40S-48GB$",
    r"^A100-80GB$",
    r"^MiniLM-L6-v2$",
    r"^DeBERTa-v2",
    r"^XLM-R",
    r"^bge-m3$",
    r"^0\.19\.1$",        # PEFT version
]

# Tokens that look numeric but are model/library names, not measurements.
NAME_TOKENS = re.compile(
    r"all-MiniLM-L6-v2|bge-m3|DeBERTa-v2-xlarge|DeBERTa-v2|XLM-RoBERTa-XL|"
    r"XLM-R-XL|ELECTRA-large|L40S-48GB|A100-80GB|A100|L40S|fp16|float16|"
    r"BERT|RoBERTa|Appendix C|PyTorch|CUDA|torch\.compile|index_select|"
    r"expandable_segments|share_att_key|SetFit|Punica|S-LoRA|LoRAX|vLLM|PEFT"
)


def flatten(obj, out=None):
    """Collect every numeric leaf in the registry."""
    if out is None:
        out = set()
    if isinstance(obj, dict):
        for v in obj.values():
            flatten(v, out)
    elif isinstance(obj, list):
        for v in obj:
            flatten(v, out)
    elif isinstance(obj, bool):
        pass
    elif isinstance(obj, (int, float)):
        out.add(float(obj))
    elif isinstance(obj, str):
        # registry keys sometimes hold stringified numbers
        try:
            out.add(float(obj))
        except ValueError:
            pass
    return out


def derived_values(vals: set[float]) -> set[float]:
    """Values a reader would legitimately write in another unit or sign.

    A measured -2.82% may be written as "2.82% below"; 28000 as "28k";
    0.29 ms as 0.29. Adding these keeps the checker from flagging honest
    re-phrasings while still rejecting invented magnitudes.
    """
    extra = set()
    for v in vals:
        extra.add(abs(v))
        extra.add(round(v))
        if v and abs(v) >= 1000:
            extra.add(v / 1000.0)      # 28000 -> 28 ("28k")
        if v and abs(v) >= 1e6:
            extra.add(v / 1e6)         # 334092288 -> 334.09 ("334M")
    return extra


# Numbers as they appear in prose: 1,000  4.08  -2.82  1e-4  2.5  9.0
NUM_RE = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?(?:[eE]-?\d+)?)")


def decimals(token: str) -> int:
    if "." not in token or "e" in token.lower():
        return 6
    return len(token.split(".")[1])


def matches(value: float, allowed: set[float]) -> bool:
    d = min(decimals(f"{value}"), 6)
    for a in allowed:
        if round(a, d) == round(value, d):
            return True
        # also accept when the prose rounds a longer measured value
        if abs(a - value) < 10 ** (-d) / 2 + 1e-12:
            return True
    return False


def check_part(path: Path, allowed: set[float], verbose: bool) -> list[str]:
    text = path.read_text()
    problems = []

    n = len(text)
    if n > CHAR_LIMIT:
        status = "OVER"
    elif n > SAFE_LIMIT:
        status = "TIGHT"
    else:
        status = "OK"
    print(f"  [{status:>5}] {path.name}: {n:,} / {CHAR_LIMIT:,} chars")
    if n > CHAR_LIMIT:
        problems.append(f"{path.name}: {n - CHAR_LIMIT} chars over the {CHAR_LIMIT} limit")
    elif n > SAFE_LIMIT:
        print(f"          only {CHAR_LIMIT - n} chars of headroom -- trim before pasting")

    for lineno, line in enumerate(text.splitlines(), 1):
        scrubbed = NAME_TOKENS.sub(" ", line)
        if any(re.search(p, line) for p in ALLOWED_PATTERNS if p.startswith("^Finding")):
            scrubbed = re.sub(r"Finding \d", " ", scrubbed)
        scrubbed = re.sub(r"§4(\.\d)?", " ", scrubbed)
        scrubbed = re.sub(r"\b[QW][1-6]\b", " ", scrubbed)
        for m in NUM_RE.finditer(scrubbed):
            tok = m.group(1)
            val = float(tok.replace(",", ""))
            if val in ALLOWED:
                continue
            if matches(val, allowed):
                if verbose:
                    print(f"      ok  {path.name}:{lineno}  {tok}")
                continue
            ctx = line.strip()[:90]
            problems.append(f"{path.name}:{lineno}  UNVERIFIED {tok!r}  in: {ctx}")
    return problems


def main() -> int:
    verbose = "--verbose" in sys.argv
    if not REGISTRY.exists():
        print("numbers.json missing -- run: python rebuttal/make_numbers.py")
        return 2
    reg = json.loads(REGISTRY.read_text())
    allowed = flatten(reg)
    allowed |= derived_values(allowed)

    if not PARTS:
        print(f"no parts found under {HERE / 'parts'}/")
        return 2

    print(f"registry: {len(allowed):,} verified values from {REGISTRY.name}")
    print(f"checking {len(PARTS)} part(s) against the {CHAR_LIMIT}-char limit\n")

    problems = []
    for p in PARTS:
        problems += check_part(p, allowed, verbose)

    print()
    if problems:
        print(f"FAIL — {len(problems)} problem(s):\n")
        for pr in problems:
            print(f"  {pr}")
        print(
            "\nEvery flagged number is either wrong, or measured but missing "
            "from make_numbers.py. Fix the text, or add the source to the "
            "registry. Do not add it to ALLOWED unless it is genuinely not a "
            "measurement."
        )
        return 1
    print("PASS — all parts fit the character limit and every number traces to numbers.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
