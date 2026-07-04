"""Tests for scripts/audit_cross_split_dedup.py -- Task 5.

Covers the 10 tests specified in ``.superpowers/sdd/task-5-brief.md``:

  1. test_exact_instruction_duplicate_detected
  2. test_exact_code_duplicate_detected
  3. test_exact_test_duplicate_detected
  4. test_func_signature_match_detected
  5. test_ast_structural_match_detected
  6. test_ngram_high_similarity_detected
  7. test_ngram_low_similarity_not_flagged
  8. test_quarantine_list_built
  9. test_unresolved_always_zero
 10. test_disjoint_splits_no_duplicates

All tests use synthetic Samples (no real MBPP data).  The n-gram tests
monkeypatch ``_ngram_bucket_key`` to force all samples into the same
bucket so Jaccard is actually computed (the production bucket key uses
the first 2 hex chars of the instruction hash, which would place
different instructions in different buckets and skip the comparison).
"""

from __future__ import annotations

from scripts import audit_cross_split_dedup as audit_mod
from scripts.audit_cross_split_dedup import run_dedup_audit
from src.schemas import Sample, Verification


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _verification() -> Verification:
    return Verification(syntax_ok=True, pytest_ok=True, ruff_ok=True,
                        timeout=False)


def _make_sample(
    sample_id: str,
    family_id: str,
    instruction: str,
    target_code: str,
    *,
    public_tests: str = "assert f(1) == 1\n\nassert f(2) == 2",
    hidden_tests: str = "assert f(3) == 3",
    source_split: str = "train",
) -> Sample:
    """Build a minimal valid code_generation Sample for dedup tests."""
    return Sample(
        sample_id=sample_id,
        family_id=family_id,
        difficulty=1,
        task_type="code_generation",
        language="python",
        skill_tags=["general"],
        instruction=instruction,
        broken_code=None,
        execution_feedback=None,
        target_code=target_code,
        public_tests=public_tests,
        hidden_tests=hidden_tests,
        verified=True,
        verification=_verification(),
        generator="test-harness",
        created_at="2026-01-01T00:00:00+00:00",
        dataset_version="v1",
        source_split=source_split,
    )


def _run(samples: dict[str, list[Sample]]) -> dict:
    """Shorthand: run the audit and pop the internal _review_queue key."""
    audit = run_dedup_audit(samples)
    audit.pop("_review_queue", None)
    return audit


# ---------------------------------------------------------------------------
# Tests 1-3: exact duplicate detection (instruction / code / test-suite)
# ---------------------------------------------------------------------------

class TestExactDuplicates:
    # --- Test 1 ---
    def test_exact_instruction_duplicate_detected(self) -> None:
        """Two samples with the same normalized instruction (different
        whitespace) -> exact duplicate via instruction_hash."""
        s1 = _make_sample(
            "mbpp_1", "mbpp_fam_1",
            instruction="Write a function to add two numbers.",
            target_code="def add(a, b):\n    return a + b\n",
            public_tests="assert add(1, 1) == 2\n\nassert add(2, 3) == 5",
            hidden_tests="assert add(0, 0) == 0",
            source_split="train",
        )
        s2 = _make_sample(
            "mbpp_2", "mbpp_fam_2",
            # Extra internal whitespace -- normalises to the same string.
            instruction="Write  a function   to add two numbers.",
            target_code="def multiply(x, y):\n    return x * y\n",
            public_tests="assert multiply(1, 1) == 1\n\nassert multiply(2, 3) == 6",
            hidden_tests="assert multiply(0, 5) == 0",
            source_split="test",
        )
        audit = _run({"train": [s1], "test": [s2], "validation": []})
        methods = [p["method"] for p in audit["exact_duplicates"]["pairs"]]
        assert "instruction_hash" in methods, (
            f"instruction_hash not in exact_duplicate methods: {methods}"
        )

    # --- Test 2 ---
    def test_exact_code_duplicate_detected(self) -> None:
        """Two samples with the same normalized target_code (one has a
        full-line comment, the other doesn't) -> exact duplicate via
        code_hash."""
        code1 = "def add(a, b):\n    return a + b\n"
        code2 = "# add two numbers\ndef add(a, b):\n    return a + b\n"
        s1 = _make_sample(
            "mbpp_1", "mbpp_fam_1",
            instruction="Write an add function.",
            target_code=code1,
            public_tests="assert add(1, 1) == 2\n\nassert add(2, 3) == 5",
            hidden_tests="assert add(0, 0) == 0",
            source_split="train",
        )
        s2 = _make_sample(
            "mbpp_2", "mbpp_fam_2",
            instruction="Compute the sum of two numbers.",
            target_code=code2,
            public_tests="assert add(10, 10) == 20\n\nassert add(20, 30) == 50",
            hidden_tests="assert add(100, 200) == 300",
            source_split="test",
        )
        audit = _run({"train": [s1], "test": [s2], "validation": []})
        methods = [p["method"] for p in audit["exact_duplicates"]["pairs"]]
        assert "code_hash" in methods, (
            f"code_hash not in exact_duplicate methods: {methods}"
        )

    # --- Test 3 ---
    def test_exact_test_duplicate_detected(self) -> None:
        """Two samples with the same test-suite (public + hidden) ->
        exact duplicate via test_hash."""
        public = "assert add(1, 1) == 2\n\nassert add(2, 3) == 5"
        hidden = "assert add(0, 0) == 0"
        s1 = _make_sample(
            "mbpp_1", "mbpp_fam_1",
            instruction="Write an add function.",
            target_code="def add(a, b):\n    return a + b\n",
            public_tests=public,
            hidden_tests=hidden,
            source_split="train",
        )
        s2 = _make_sample(
            "mbpp_2", "mbpp_fam_2",
            instruction="Compute the sum of two numbers.",
            target_code="def multiply(x, y):\n    return x * y\n",
            public_tests=public,  # SAME test-suite
            hidden_tests=hidden,
            source_split="test",
        )
        audit = _run({"train": [s1], "test": [s2], "validation": []})
        methods = [p["method"] for p in audit["exact_duplicates"]["pairs"]]
        assert "test_hash" in methods, (
            f"test_hash not in exact_duplicate methods: {methods}"
        )


# ---------------------------------------------------------------------------
# Tests 4-7: high-similarity detection (signature / AST / n-gram)
# ---------------------------------------------------------------------------

class TestHighSimilarity:
    # --- Test 4 ---
    def test_func_signature_match_detected(self) -> None:
        """Two samples with the same function signature (different body)
        -> high similarity via func_signature."""
        code1 = "def add(a, b):\n    return a + b\n"
        code2 = "def add(a, b):\n    return a * b\n"  # same sig, diff op
        s1 = _make_sample(
            "mbpp_1", "mbpp_fam_1",
            instruction="Write a function to add two numbers.",
            target_code=code1,
            public_tests="assert add(1, 1) == 2\n\nassert add(2, 3) == 5",
            hidden_tests="assert add(0, 0) == 0",
            source_split="train",
        )
        s2 = _make_sample(
            "mbpp_2", "mbpp_fam_2",
            instruction="Compute the product of two numbers.",
            target_code=code2,
            public_tests="assert add(1, 1) == 1\n\nassert add(2, 3) == 6",
            hidden_tests="assert add(0, 5) == 0",
            source_split="test",
        )
        audit = _run({"train": [s1], "test": [s2], "validation": []})
        methods = [p["method"] for p in audit["high_similarity"]["pairs"]]
        assert "func_signature" in methods, (
            f"func_signature not in high_similarity methods: {methods}"
        )

    # --- Test 5 ---
    def test_ast_structural_match_detected(self) -> None:
        """Two samples with the same AST structure (different local var
        names in the body) -> high similarity via ast_structural.

        After Name->VAR normalisation, ``c = a + b; return c`` and
        ``d = a + b; return d`` produce identical ``ast.dump`` output.
        Function name and parameter names are identical so func_signature
        also matches -- that's fine; this test only asserts
        ast_structural is present.
        """
        code1 = "def add(a, b):\n    c = a + b\n    return c\n"
        code2 = "def add(a, b):\n    d = a + b\n    return d\n"
        s1 = _make_sample(
            "mbpp_1", "mbpp_fam_1",
            instruction="Compute sum of two numbers using a temp variable.",
            target_code=code1,
            public_tests="assert add(1, 1) == 2\n\nassert add(2, 3) == 5",
            hidden_tests="assert add(0, 0) == 0",
            source_split="train",
        )
        s2 = _make_sample(
            "mbpp_2", "mbpp_fam_2",
            instruction="Calculate the sum of two numbers using a temporary variable.",
            target_code=code2,
            public_tests="assert add(10, 10) == 20\n\nassert add(20, 30) == 50",
            hidden_tests="assert add(100, 200) == 300",
            source_split="test",
        )
        audit = _run({"train": [s1], "test": [s2], "validation": []})
        methods = [p["method"] for p in audit["high_similarity"]["pairs"]]
        assert "ast_structural" in methods, (
            f"ast_structural not in high_similarity methods: {methods}"
        )

    # --- Test 6 ---
    def test_ngram_high_similarity_detected(self, monkeypatch) -> None:
        """Two samples with Jaccard > 0.7 -> high similarity via ngram_3.

        The bucket key is monkeypatched to force both samples into the
        same bucket so the Jaccard is actually computed (the production
        bucket key uses the first 2 hex chars of the instruction hash,
        which would place different instructions in different buckets).
        """
        # Force all samples into the same bucket.
        monkeypatch.setattr(audit_mod, "_ngram_bucket_key", lambda h: "XX")
        # Instructions differ by one word ("positive") -- Jaccard ~0.79.
        s1 = _make_sample(
            "mbpp_1", "mbpp_fam_1",
            instruction=(
                "Write a python function to find minimum sum of factors "
                "of a given number."
            ),
            target_code="def min_sum_factors(num):\n    return num + 1\n",
            public_tests="assert min_sum_factors(12) == 13\n\nassert min_sum_factors(105) == 106",
            hidden_tests="assert min_sum_factors(2) == 3",
            source_split="train",
        )
        s2 = _make_sample(
            "mbpp_2", "mbpp_fam_2",
            instruction=(
                "Write a python function to find minimum sum of factors "
                "of a given positive number."
            ),
            target_code="def max_sum_divs(value):\n    return value - 1\n",
            public_tests="assert max_sum_divs(10) == 9\n\nassert max_sum_divs(20) == 19",
            hidden_tests="assert max_sum_divs(5) == 4",
            source_split="test",
        )
        audit = _run({"train": [s1], "test": [s2], "validation": []})
        methods = [p["method"] for p in audit["high_similarity"]["pairs"]]
        assert "ngram_3" in methods, (
            f"ngram_3 not in high_similarity methods: {methods}"
        )
        # Sanity: the recorded score should be > 0.7.
        ngram_pairs = [p for p in audit["high_similarity"]["pairs"]
                      if p["method"] == "ngram_3"]
        assert ngram_pairs, "no ngram_3 pairs recorded"
        assert all(p["score"] > 0.7 for p in ngram_pairs)

    # --- Test 7 ---
    def test_ngram_low_similarity_not_flagged(self, monkeypatch) -> None:
        """Two samples with Jaccard < 0.7 -> NOT flagged by ngram_3.

        The bucket key is monkeypatched to force both samples into the
        same bucket so the Jaccard is actually computed (and found low).
        """
        # Force all samples into the same bucket.
        monkeypatch.setattr(audit_mod, "_ngram_bucket_key", lambda h: "XX")
        s1 = _make_sample(
            "mbpp_1", "mbpp_fam_1",
            instruction="Write a function to add two numbers.",
            target_code="def add(a, b):\n    return a + b\n",
            public_tests="assert add(1, 1) == 2\n\nassert add(2, 3) == 5",
            hidden_tests="assert add(0, 0) == 0",
            source_split="train",
        )
        s2 = _make_sample(
            "mbpp_2", "mbpp_fam_2",
            instruction="Reverse a linked list iteratively.",
            target_code=(
                "def reverse_list(head):\n"
                "    prev = None\n"
                "    while head:\n"
                "        head.next, prev, head = prev, head, head.next\n"
                "    return prev\n"
            ),
            public_tests=(
                "assert reverse_list([1,2,3]) == [3,2,1]\n\n"
                "assert reverse_list([]) == []"
            ),
            hidden_tests="assert reverse_list([1]) == [1]",
            source_split="test",
        )
        audit = _run({"train": [s1], "test": [s2], "validation": []})
        methods = [p["method"] for p in audit["high_similarity"]["pairs"]]
        assert "ngram_3" not in methods, (
            f"ngram_3 should not flag low-similarity pair: {methods}"
        )


# ---------------------------------------------------------------------------
# Tests 8-10: quarantine, unresolved, clean-split baseline
# ---------------------------------------------------------------------------

class TestQuarantineAndConclusion:
    # --- Test 8 ---
    def test_quarantine_list_built(self) -> None:
        """Families in any duplicate / high-sim pair -> quarantined.

        Uses an instruction-hash exact duplicate so both families land in
        the quarantine list.  (ngram_3 will also fire because identical
        instructions yield Jaccard = 1.0 -- both checks contribute to
        the same quarantine set, which is the intended behaviour.)
        """
        s1 = _make_sample(
            "mbpp_1", "mbpp_fam_1",
            instruction="Write a function to add two numbers.",
            target_code="def add(a, b):\n    return a + b\n",
            public_tests="assert add(1, 1) == 2\n\nassert add(2, 3) == 5",
            hidden_tests="assert add(0, 0) == 0",
            source_split="train",
        )
        s2 = _make_sample(
            "mbpp_2", "mbpp_fam_2",
            instruction="Write a function to add two numbers.",  # SAME
            target_code="def multiply(x, y):\n    return x * y\n",
            public_tests="assert multiply(1, 1) == 1\n\nassert multiply(2, 3) == 6",
            hidden_tests="assert multiply(0, 5) == 0",
            source_split="test",
        )
        audit = _run({"train": [s1], "test": [s2], "validation": []})
        quarantined = audit["quarantined_families"]["families"]
        assert "mbpp_fam_1" in quarantined, (
            f"mbpp_fam_1 not in quarantine list: {quarantined}"
        )
        assert "mbpp_fam_2" in quarantined, (
            f"mbpp_fam_2 not in quarantine list: {quarantined}"
        )
        assert audit["quarantined_families"]["count"] >= 2

    # --- Test 9 ---
    def test_unresolved_always_zero(self) -> None:
        """After the script runs, unresolved == 0 (all auto-quarantined)
        and conclusion == PASS, even when duplicates exist."""
        s1 = _make_sample(
            "mbpp_1", "mbpp_fam_1",
            instruction="Write a function to add two numbers.",
            target_code="def add(a, b):\n    return a + b\n",
            public_tests="assert add(1, 1) == 2\n\nassert add(2, 3) == 5",
            hidden_tests="assert add(0, 0) == 0",
            source_split="train",
        )
        s2 = _make_sample(
            "mbpp_2", "mbpp_fam_2",
            instruction="Write a function to add two numbers.",
            target_code="def multiply(x, y):\n    return x * y\n",
            public_tests="assert multiply(1, 1) == 1\n\nassert multiply(2, 3) == 6",
            hidden_tests="assert multiply(0, 5) == 0",
            source_split="test",
        )
        audit = _run({"train": [s1], "test": [s2], "validation": []})
        assert audit["unresolved"]["count"] == 0
        assert audit["conclusion"] == "PASS"

    # --- Test 10 ---
    def test_disjoint_splits_no_duplicates(self) -> None:
        """Three splits with no overlap -> 0 duplicates, 0 quarantine,
        conclusion PASS."""
        s1 = _make_sample(
            "mbpp_1", "mbpp_fam_1",
            instruction="Write a function to add two numbers.",
            target_code="def add(a, b):\n    return a + b\n",
            public_tests="assert add(1, 1) == 2\n\nassert add(2, 3) == 5",
            hidden_tests="assert add(0, 0) == 0",
            source_split="train",
        )
        s2 = _make_sample(
            "mbpp_2", "mbpp_fam_2",
            instruction="Reverse a linked list iteratively.",
            target_code=(
                "def reverse_list(head):\n"
                "    prev = None\n"
                "    while head:\n"
                "        head.next, prev, head = prev, head, head.next\n"
                "    return prev\n"
            ),
            public_tests=(
                "assert reverse_list([1,2,3]) == [3,2,1]\n\n"
                "assert reverse_list([]) == []"
            ),
            hidden_tests="assert reverse_list([1]) == [1]",
            source_split="test",
        )
        s3 = _make_sample(
            "mbpp_3", "mbpp_fam_3",
            instruction="Check if a string is a palindrome.",
            target_code="def is_palindrome(s):\n    return s == s[::-1]\n",
            public_tests=(
                "assert is_palindrome('aba')\n\n"
                "assert not is_palindrome('abc')"
            ),
            hidden_tests="assert is_palindrome('')",
            source_split="validation",
        )
        audit = _run({
            "train": [s1],
            "test": [s2],
            "validation": [s3],
        })
        assert audit["exact_duplicates"]["count"] == 0
        assert audit["high_similarity"]["count"] == 0
        assert audit["quarantined_families"]["count"] == 0
        assert audit["quarantined_families"]["families"] == []
        assert audit["unresolved"]["count"] == 0
        assert audit["conclusion"] == "PASS"
