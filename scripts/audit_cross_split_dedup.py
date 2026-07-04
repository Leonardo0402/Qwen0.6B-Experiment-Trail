"""
scripts/audit_cross_split_dedup.py -- Cross-split semantic dedup audit for MBPP.

Reads ``data/external/mbpp/verified/{train,test,validation}.jsonl`` (one
Sample JSON per line) and runs 6 dedup checks across splits:

  1. instruction_hash   -- normalized instruction hash (exact match)
  2. code_hash          -- normalized target_code hash (exact match)
  3. test_hash          -- test-suite hash (exact match)
  4. func_signature     -- ast.FunctionDef name+args (HIGH_SIMILARITY)
  5. ast_structural     -- ast.dump with Name->VAR, Constant->CONST (HIGH_SIMILARITY)
  6. ngram_3            -- 3-gram Jaccard > 0.7 (HIGH_SIMILARITY)

Outputs (under <report-dir>/):
  cross-split-dedup-audit.json         -- statistics
  cross-split-dedup-review-queue.jsonl -- high-similarity pairs (one JSON per line)
  cross-split-dedup-quarantine.json    -- quarantined families (excluded from P3)

Per P3 plan Global Constraint #15: ``unresolved=0`` is required before any
Train/Val/Frozen partition is built.  All high-similarity pairs are
auto-quarantined so ``unresolved`` is always 0 after this script runs.

Usage
-----
    python scripts/audit_cross_split_dedup.py \\
        --input-dir data/external/mbpp \\
        --report-dir reports/p3

Exit codes
----------
    0   success
    1   no samples found / fatal error
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Project-root import guard (so the script works from any cwd)
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.schemas import Sample  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SPLITS: list[str] = ["train", "test", "validation"]

_CHECKS_PERFORMED: list[str] = [
    "instruction_hash",
    "code_hash",
    "test_hash",
    "func_signature",
    "ast_structural",
    "ngram_3",
]

_NGRAM_THRESHOLD: float = 0.7
_NGRAM_N: int = 3

_WHITESPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


# ---------------------------------------------------------------------------
# In-memory record (lighter than Sample for the O(n^2) n-gram pass)
# ---------------------------------------------------------------------------

class Rec:
    """Flat record holding only the fields needed by the dedup checks."""

    __slots__ = (
        "sample_id", "family_id", "split",
        "instruction", "target_code", "public_tests", "hidden_tests",
    )

    def __init__(self, sample: Sample, split: str):
        self.sample_id = sample.sample_id
        self.family_id = sample.family_id
        self.split = split
        self.instruction = sample.instruction
        self.target_code = sample.target_code
        self.public_tests = sample.public_tests
        self.hidden_tests = sample.hidden_tests


# ---------------------------------------------------------------------------
# Normalisation helpers (pure, importable, testable)
# ---------------------------------------------------------------------------

def normalize_instruction(text: str) -> str:
    """Lowercase, collapse whitespace, strip."""
    return _WHITESPACE_RE.sub(" ", (text or "").strip().lower()).strip()


def normalize_code(code: str) -> str:
    """Strip ``# ...`` comment lines, strip whitespace, drop blank lines.

    Per task-5-brief check 2:
    - Remove lines whose stripped form starts with ``#``.
    - Strip whitespace from every remaining line.
    - Drop blank lines entirely (so they don't affect the hash).
    """
    out_lines: list[str] = []
    for line in (code or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        out_lines.append(stripped)
    return "\n".join(out_lines)


def normalize_test_suite(public: str, hidden: str) -> str:
    """Concatenate ``public_tests + "\\n" + hidden_tests`` and collapse whitespace."""
    combined = (public or "") + "\n" + (hidden or "")
    return _WHITESPACE_RE.sub(" ", combined).strip()


def sha256_hex(text: str) -> str:
    """SHA-256 hex digest of *text* encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tokenize_instruction(text: str) -> list[str]:
    """Tokenize instruction: lowercase, split on non-alphanumeric."""
    return _TOKEN_RE.findall((text or "").lower())


def compute_ngrams(tokens: list[str], n: int = _NGRAM_N) -> set[tuple[str, ...]]:
    """Return the set of *n*-grams (as tuples) from *tokens*.

    Returns an empty set when *tokens* has fewer than *n* elements.
    """
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def jaccard(a: set, b: set) -> float:
    """Jaccard similarity between two sets.

    Returns 0.0 when both sets are empty (avoiding ZeroDivisionError).
    """
    if not a and not b:
        return 0.0
    union = len(a | b)
    if union == 0:
        return 0.0
    return len(a & b) / union


def _ngram_bucket_key(instruction_hash: str) -> str:
    """Bucket key for n-gram comparison (first 2 hex chars of instruction hash).

    Per task-5-brief check 6: blocking/bucketing to avoid O(n^2) on 955
    samples.  Tests may monkeypatch this function to force all samples
    into the same bucket.
    """
    return instruction_hash[:2]


# ---------------------------------------------------------------------------
# AST-based checks (pure, importable, testable)
# ---------------------------------------------------------------------------

class _VarNormalizer(ast.NodeTransformer):
    """Replace ``ast.Name.id`` with ``"VAR"`` and ``ast.Constant.value``
    with ``"CONST"`` so structural dumps ignore literal/var-name differences."""

    def visit_Name(self, node: ast.Name) -> ast.AST:
        node.id = "VAR"
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        node.value = "CONST"
        return node


def extract_func_signatures(code: str) -> Optional[str]:
    """Parse *code* and return a stable signature string.

    Format: ``"func_name(arg1,arg2,...)"``; multiple functions joined by
    ``;``.  Returns ``None`` when the code cannot be parsed (SyntaxError)
    or contains no ``ast.FunctionDef`` nodes.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        print(
            f"WARNING: func_signature: skipping sample (SyntaxError: "
            f"{exc.msg} at line {exc.lineno})",
            file=sys.stderr,
        )
        return None
    sigs: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            args = [a.arg for a in node.args.args]
            sigs.append(f"{node.name}({','.join(args)})")
    return ";".join(sigs) if sigs else None


def compute_ast_structural_hash(code: str) -> Optional[str]:
    """SHA-256 of ``ast.dump(tree, annotate_fields=False)`` after replacing
    all ``ast.Name.id`` -> ``"VAR"`` and ``ast.Constant.value`` -> ``"CONST"``.

    Returns ``None`` when the code cannot be parsed (SyntaxError).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        print(
            f"WARNING: ast_structural: skipping sample (SyntaxError: "
            f"{exc.msg} at line {exc.lineno})",
            file=sys.stderr,
        )
        return None
    _VarNormalizer().visit(tree)
    ast.fix_missing_locations(tree)
    dumped = ast.dump(tree, annotate_fields=False)
    return sha256_hex(dumped)


# ---------------------------------------------------------------------------
# Per-sample feature extraction
# ---------------------------------------------------------------------------

class SampleFeatures:
    """Pre-computed per-sample features needed by the dedup checks.

    AST-based fields are ``None`` when the code failed to parse; the
    sample is still counted in ``total_samples_checked`` but contributes
    no pairs to that method (per task-5-brief constraint).
    """

    __slots__ = (
        "rec",
        "instruction_hash",
        "code_hash",
        "test_hash",
        "func_signature",
        "ast_structural_hash",
        "ngram_set",
        "ngram_bucket",
    )

    def __init__(self, rec: Rec):
        self.rec = rec
        self.instruction_hash = sha256_hex(normalize_instruction(rec.instruction))
        self.code_hash = sha256_hex(normalize_code(rec.target_code))
        self.test_hash = sha256_hex(
            normalize_test_suite(rec.public_tests, rec.hidden_tests)
        )
        self.func_signature = extract_func_signatures(rec.target_code)
        self.ast_structural_hash = compute_ast_structural_hash(rec.target_code)
        self.ngram_set = compute_ngrams(tokenize_instruction(rec.instruction))
        # Bucket by first 2 chars of instruction hash (see _ngram_bucket_key).
        self.ngram_bucket = _ngram_bucket_key(self.instruction_hash)


# ---------------------------------------------------------------------------
# Pair generation
# ---------------------------------------------------------------------------

def _cross_split_pairs(
    samples: list[SampleFeatures],
) -> list[tuple[SampleFeatures, SampleFeatures]]:
    """Return all cross-split pairs from a list of samples in the same bucket.

    Pairs are ordered so ``(a.rec.split, a.rec.sample_id)`` <
    ``(b.rec.split, b.rec.sample_id)`` -- this makes the output
    deterministic and ensures each unordered pair appears exactly once.
    """
    pairs: list[tuple[SampleFeatures, SampleFeatures]] = []
    n = len(samples)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = samples[i], samples[j]
            if a.rec.split == b.rec.split:
                continue
            key_a = (a.rec.split, a.rec.sample_id)
            key_b = (b.rec.split, b.rec.sample_id)
            if key_a > key_b:
                a, b = b, a
            pairs.append((a, b))
    return pairs


# ---------------------------------------------------------------------------
# Core audit
# ---------------------------------------------------------------------------

def run_dedup_audit(
    split_to_samples: dict[str, list[Sample]],
) -> dict:
    """Run all 6 dedup checks on the given samples.

    Returns a dict with the public audit.json schema plus an internal
    ``_review_queue`` key (list of review-queue entries) that the CLI
    writes to ``cross-split-dedup-review-queue.jsonl``.  Callers that
    only need the public audit fields should ignore ``_review_queue``.
    """
    features: list[SampleFeatures] = []
    for split in _SPLITS:
        for s in split_to_samples.get(split, []):
            features.append(SampleFeatures(Rec(s, split)))

    total = len(features)

    exact_dup_pairs: list[dict] = []
    high_sim_pairs: list[dict] = []
    review_queue: list[dict] = []
    quarantined_families: set[str] = set()

    # ------------------------------------------------------------------
    # Check 1: instruction_hash (EXACT_DUPLICATE)
    # ------------------------------------------------------------------
    by_instr: dict[str, list[SampleFeatures]] = {}
    for f in features:
        by_instr.setdefault(f.instruction_hash, []).append(f)
    for bucket in by_instr.values():
        for a, b in _cross_split_pairs(bucket):
            exact_dup_pairs.append({
                "sample_a": a.rec.sample_id, "split_a": a.rec.split,
                "sample_b": b.rec.sample_id, "split_b": b.rec.split,
                "method": "instruction_hash",
            })
            quarantined_families.add(a.rec.family_id)
            quarantined_families.add(b.rec.family_id)

    # ------------------------------------------------------------------
    # Check 2: code_hash (EXACT_DUPLICATE)
    # ------------------------------------------------------------------
    by_code: dict[str, list[SampleFeatures]] = {}
    for f in features:
        by_code.setdefault(f.code_hash, []).append(f)
    for bucket in by_code.values():
        for a, b in _cross_split_pairs(bucket):
            exact_dup_pairs.append({
                "sample_a": a.rec.sample_id, "split_a": a.rec.split,
                "sample_b": b.rec.sample_id, "split_b": b.rec.split,
                "method": "code_hash",
            })
            quarantined_families.add(a.rec.family_id)
            quarantined_families.add(b.rec.family_id)

    # ------------------------------------------------------------------
    # Check 3: test_hash (EXACT_DUPLICATE)
    # ------------------------------------------------------------------
    by_test: dict[str, list[SampleFeatures]] = {}
    for f in features:
        by_test.setdefault(f.test_hash, []).append(f)
    for bucket in by_test.values():
        for a, b in _cross_split_pairs(bucket):
            exact_dup_pairs.append({
                "sample_a": a.rec.sample_id, "split_a": a.rec.split,
                "sample_b": b.rec.sample_id, "split_b": b.rec.split,
                "method": "test_hash",
            })
            quarantined_families.add(a.rec.family_id)
            quarantined_families.add(b.rec.family_id)

    # ------------------------------------------------------------------
    # Check 4: func_signature (HIGH_SIMILARITY)
    # ------------------------------------------------------------------
    by_sig: dict[str, list[SampleFeatures]] = {}
    for f in features:
        if f.func_signature is None:
            continue
        by_sig.setdefault(f.func_signature, []).append(f)
    for bucket in by_sig.values():
        for a, b in _cross_split_pairs(bucket):
            review_queue.append({
                "sample_a": a.rec.sample_id, "split_a": a.rec.split,
                "sample_b": b.rec.sample_id, "split_b": b.rec.split,
                "method": "func_signature", "score": 1.0,
                "status": "auto_quarantined",
            })
            high_sim_pairs.append({
                "sample_a": a.rec.sample_id, "sample_b": b.rec.sample_id,
                "method": "func_signature", "score": 1.0,
            })
            quarantined_families.add(a.rec.family_id)
            quarantined_families.add(b.rec.family_id)

    # ------------------------------------------------------------------
    # Check 5: ast_structural (HIGH_SIMILARITY)
    # ------------------------------------------------------------------
    by_ast: dict[str, list[SampleFeatures]] = {}
    for f in features:
        if f.ast_structural_hash is None:
            continue
        by_ast.setdefault(f.ast_structural_hash, []).append(f)
    for bucket in by_ast.values():
        for a, b in _cross_split_pairs(bucket):
            review_queue.append({
                "sample_a": a.rec.sample_id, "split_a": a.rec.split,
                "sample_b": b.rec.sample_id, "split_b": b.rec.split,
                "method": "ast_structural", "score": 1.0,
                "status": "auto_quarantined",
            })
            high_sim_pairs.append({
                "sample_a": a.rec.sample_id, "sample_b": b.rec.sample_id,
                "method": "ast_structural", "score": 1.0,
            })
            quarantined_families.add(a.rec.family_id)
            quarantined_families.add(b.rec.family_id)

    # ------------------------------------------------------------------
    # Check 6: ngram_3 (HIGH_SIMILARITY, Jaccard > 0.7)
    # ------------------------------------------------------------------
    by_bucket: dict[str, list[SampleFeatures]] = {}
    for f in features:
        by_bucket.setdefault(f.ngram_bucket, []).append(f)
    for bucket in by_bucket.values():
        for a, b in _cross_split_pairs(bucket):
            score = jaccard(a.ngram_set, b.ngram_set)
            if score > _NGRAM_THRESHOLD:
                score_rounded = round(score, 4)
                review_queue.append({
                    "sample_a": a.rec.sample_id, "split_a": a.rec.split,
                    "sample_b": b.rec.sample_id, "split_b": b.rec.split,
                    "method": "ngram_3", "score": score_rounded,
                    "status": "auto_quarantined",
                })
                high_sim_pairs.append({
                    "sample_a": a.rec.sample_id, "sample_b": b.rec.sample_id,
                    "method": "ngram_3", "score": score_rounded,
                })
                quarantined_families.add(a.rec.family_id)
                quarantined_families.add(b.rec.family_id)

    # ------------------------------------------------------------------
    # Build audit dict
    # ------------------------------------------------------------------
    quarantined_list = sorted(quarantined_families)
    unresolved_count = 0  # all high-sim auto-quarantined per P3 plan
    conclusion = "PASS" if unresolved_count == 0 else "FAIL"

    return {
        "total_samples_checked": total,
        "splits_checked": list(_SPLITS),
        "checks_performed": list(_CHECKS_PERFORMED),
        "exact_duplicates": {
            "count": len(exact_dup_pairs),
            "pairs": exact_dup_pairs,
        },
        "high_similarity": {
            "count": len(high_sim_pairs),
            "pairs": high_sim_pairs,
        },
        "quarantined_families": {
            "count": len(quarantined_list),
            "families": quarantined_list,
        },
        "unresolved": {
            "count": unresolved_count,
            "note": (
                "All high-similarity pairs auto-quarantined per P3 plan "
                "(unconfirmed \u2192 quarantine)"
            ),
        },
        "conclusion": conclusion,
        # Internal: not part of the public audit.json schema; the CLI
        # pops this and writes it to cross-split-dedup-review-queue.jsonl.
        "_review_queue": review_queue,
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_verified_samples(input_dir: Path) -> dict[str, list[Sample]]:
    """Load ``<input_dir>/verified/<split>.jsonl`` for each split in _SPLITS.

    Missing files are treated as empty splits (no error).
    """
    result: dict[str, list[Sample]] = {}
    for split in _SPLITS:
        path = input_dir / "verified" / f"{split}.jsonl"
        samples: list[Sample] = []
        if path.exists():
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        samples.append(Sample.from_json_line(line))
        result[split] = samples
    return result


def write_reports(
    audit: dict,
    review_queue: list[dict],
    report_dir: Path,
    *,
    generated_at: Optional[str] = None,
) -> None:
    """Write the 3 output files under *report_dir*.

    Files written:
      - ``cross-split-dedup-audit.json``
      - ``cross-split-dedup-review-queue.jsonl``
      - ``cross-split-dedup-quarantine.json``
    """
    if generated_at is None:
        generated_at = datetime.now(timezone.utc).isoformat()

    report_dir.mkdir(parents=True, exist_ok=True)

    # 1. audit.json (strip the internal _review_queue key).
    audit_public = {k: v for k, v in audit.items() if not k.startswith("_")}
    audit_public["generated_at"] = generated_at
    audit_path = report_dir / "cross-split-dedup-audit.json"
    with audit_path.open("w", encoding="utf-8") as fh:
        json.dump(audit_public, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    # 2. review-queue.jsonl (one JSON per line).
    review_path = report_dir / "cross-split-dedup-review-queue.jsonl"
    with review_path.open("w", encoding="utf-8", newline="\n") as fh:
        for entry in review_queue:
            fh.write(
                json.dumps(entry, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )

    # 3. quarantine.json
    quarantine = {
        "generated_at": generated_at,
        "quarantine_reason": (
            "Cross-split semantic duplicates or high-similarity pairs "
            "(unconfirmed). Excluded from P3 Frozen/Val/Train partition."
        ),
        "quarantined_families": audit["quarantined_families"]["families"],
        "count": audit["quarantined_families"]["count"],
        "auto_quarantine_rules": [
            "exact_duplicate",
            "high_similarity_unconfirmed",
        ],
    }
    quarantine_path = report_dir / "cross-split-dedup-quarantine.json"
    with quarantine_path.open("w", encoding="utf-8") as fh:
        json.dump(quarantine, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Cross-split semantic dedup audit for MBPP verified samples. "
            "Reads <input-dir>/verified/<split>.jsonl and writes 3 report "
            "files under <report-dir>/."
        ),
    )
    p.add_argument(
        "--input-dir", default="data/external/mbpp",
        help="MBPP root directory (with verified/<split>.jsonl). "
             "Default: data/external/mbpp.",
    )
    p.add_argument(
        "--report-dir", default="reports/p3",
        help="Output report directory. Default: reports/p3.",
    )
    return p


def main() -> int:
    """CLI entry point.  Returns 0 on success, 1 on fatal error."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")  # type: ignore[attr-defined]
    except Exception:
        pass

    args = _build_parser().parse_args()
    input_dir = Path(args.input_dir)
    report_dir = Path(args.report_dir)

    split_to_samples = load_verified_samples(input_dir)
    total = sum(len(v) for v in split_to_samples.values())
    if total == 0:
        print(
            f"ERROR: no verified samples found under "
            f"{input_dir / 'verified'}",
            file=sys.stderr,
        )
        return 1

    print(
        f"audit_cross_split_dedup: checking {total} samples across "
        f"{len(_SPLITS)} splits"
    )
    for split in _SPLITS:
        print(f"  {split:10}: {len(split_to_samples[split])} samples")

    audit = run_dedup_audit(split_to_samples)
    review_queue = audit.pop("_review_queue", [])
    write_reports(audit, review_queue, report_dir)

    # Per-method breakdowns for the stdout summary.
    method_exact: dict[str, int] = {m: 0 for m in _CHECKS_PERFORMED}
    for p in audit["exact_duplicates"]["pairs"]:
        method_exact[p["method"]] = method_exact.get(p["method"], 0) + 1
    method_high: dict[str, int] = {m: 0 for m in _CHECKS_PERFORMED}
    for p in audit["high_similarity"]["pairs"]:
        method_high[p["method"]] = method_high.get(p["method"], 0) + 1

    print()
    print("  exact_duplicates by method:")
    for m in _CHECKS_PERFORMED:
        if method_exact.get(m, 0):
            print(f"    {m:18}: {method_exact[m]}")
    print(f"  exact_duplicates total: {audit['exact_duplicates']['count']}")
    print("  high_similarity by method:")
    for m in _CHECKS_PERFORMED:
        if method_high.get(m, 0):
            print(f"    {m:18}: {method_high[m]}")
    print(f"  high_similarity total: {audit['high_similarity']['count']}")
    print(
        f"  quarantined_families:  "
        f"{audit['quarantined_families']['count']}"
    )
    print(f"  unresolved:            {audit['unresolved']['count']}")
    print(f"  conclusion:            {audit['conclusion']}")

    audit_path = report_dir / "cross-split-dedup-audit.json"
    review_path = report_dir / "cross-split-dedup-review-queue.jsonl"
    quarantine_path = report_dir / "cross-split-dedup-quarantine.json"
    print()
    print(f"  audit:        -> {audit_path}")
    print(f"  review queue: -> {review_path}")
    print(f"  quarantine:    -> {quarantine_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
