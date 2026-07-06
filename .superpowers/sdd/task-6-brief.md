# Task 6 Brief: Build Family Registry

## Context
- Project: e:\agent\Qwen\qwen3-code-lab
- Branch: feat/p3-capability-expansion-v2 (Tasks 1-5 complete)
- Plan file: .superpowers/sdd/p3-plan.md (Global Constraints #8, #9, #15 bind this task)
- Task 5 produced `reports/p3/cross-split-dedup-quarantine.json` listing 58 quarantined family_ids that must NEVER enter Train/Val/Frozen partition.
- Task 6 sits between dedup audit and Frozen v3 candidate reservation. Task 7 will call `registry.claim(family_id, "frozen_v3_candidate")` for the 120 reserved families; Task 8 will upgrade those to `"frozen_v3"`; Task 9 will claim `"p3_train"` / `"p3_validation"` and assert pairwise disjoint.

## Goal
Build three new files:
1. `src/family_registry.py` — pure-Python API (dataclass + functions, no I/O side-effects in the API itself)
2. `scripts/build_family_registry.py` — CLI builder that reads P2 partition + MBPP verified JSONL + Task 5 quarantine list, produces `data/family-registry.json`
3. `tests/test_family_registry.py` — test suite

And one new data file produced by running the builder:
4. `data/family-registry.json` — the canonical registry (committed)

## family-registry.json Schema (top-level)
```json
{
  "generated_at": "<iso8601 utc>",
  "generator": "build_family_registry.py",
  "schema_version": 1,
  "total_families": <int>,
  "total_p2_used": <int>,
  "total_quarantined": <int>,
  "total_new_available": <int>,
  "families": {
    "<family_id>": {
      "source_task_id": "mbpp_<task_id>",
      "source_split": "train" | "test" | "validation",
      "usage": [<string>, ...],
      "first_commit": "<git sha short or 'unknown'>",
      "dataset_version": "mbpp-v1",
      "sample_ids": ["mbpp_<task_id>"]
    },
    ...
  }
}
```

### Family `usage` tag vocabulary (binding — use these exact strings)
- P2 backfill tags (from `data/p2-curriculum/family-partition.json`):
  - `"p2_train"` — 224 families in `train_families`
  - `"p2_validation"` — 75 families in `validation_families`
  - `"p2_frozen_v2"` — 75 families in `frozen_families`
- P3 quarantine tag (from `reports/p3/cross-split-dedup-quarantine.json`):
  - `"quarantine"` — 58 families in `quarantined_families`
- P3 future tags (NOT applied in this task; the API supports them but the builder does not invoke `claim()` with these — Tasks 7/8/9 will):
  - `"frozen_v3_candidate"` (Task 7)
  - `"frozen_v3"` (Task 8)
  - `"p3_train"` (Task 9)
  - `"p3_validation"` (Task 9)
  - `"p3_train_replay"` (Task 9, P2-replay subset)

### Backfill rules (binding)
1. **P2 backfill**: For each family_id in `train_families`, set `usage=["p2_train"]`. Same for `validation_families` → `["p2_validation"]`, `frozen_families` → `["p2_frozen_v2"]`. All 374 P2 families come from MBPP `train` source_split. Their `first_commit` is the git short SHA of the P2 partition file's introduction; for simplicity, use `"515c955"` (the merge commit that introduced P2 — see `git log --oneline -- data/p2-curriculum/family-partition.json`).
2. **MBPP verified families backfill**: Read `data/external/mbpp/verified/{train,test,validation}.jsonl` to discover NEW families not already in the registry. For each unique `family_id` encountered, IF the family does not exist in the registry yet (i.e. it is from the `test` or `validation` source_split, not from the 374 P2 train families), create an entry with `source_split` from the sample's `source_split` field (or derive from the filename if missing), `dataset_version="mbpp-v1"`, `first_commit="3dce2ce"` (the import commit). For families that ALREADY exist (P2 backfill), DO NOT create a duplicate — just verify the `source_split` matches (sanity check; if mismatch, abort with a clear error). `sample_ids` are derived from `family_id` per rule 4, not from the JSONL pass.
3. **Quarantine backfill**: For each family_id in `reports/p3/cross-split-dedup-quarantine.json::quarantined_families`, ADD `"quarantine"` to the `usage` list (append, do not replace existing P2 tags if any). These families are excluded from P3 partition but their P2 history is preserved.
4. **Sample IDs**: For MBPP families, `sample_ids = ["mbpp_<task_id>"]` where `task_id` is parsed from `family_id` via the bijective mapping `mbpp_fam_<n>` → `mbpp_<n>` (every MBPP family has exactly one source sample). This derivation works for ALL MBPP families — including the 3 train families whose samples were rejected at verify time (they still exist in the partition file with a P2 tag). Do NOT enumerate variant sample_ids (those live in stage manifests, not the registry). Do NOT read verified/rejected JSONL just to populate sample_ids — derive them from family_id instead.

### Builder invariants (asserted at end of builder)
After backfill, the builder MUST assert:
- `total_p2_used == 374` (224+75+75)
- `total_quarantined == 58` (matches Task 5 output)
- `total_new_available == (families with usage == [])` — families with no P2 tag, no quarantine, no P3 tag
- `total_families == 374 + (955 verified families' unique family_ids) - overlap_with_p2` (count distinct family_ids across all sources)
- No family has both `"quarantine"` AND a P3 future tag (`"frozen_v3"`, `"p3_train"`, `"p3_validation"`) — at this stage of the build, no P3 tags exist yet, so this is trivially true; assert anyway as a guard for future Tasks 7-9.

## src/family_registry.py API

```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json

@dataclass
class FamilyEntry:
    family_id: str
    source_task_id: str
    source_split: str
    usage: list[str] = field(default_factory=list)
    first_commit: str = "unknown"
    dataset_version: str = "mbpp-v1"
    sample_ids: list[str] = field(default_factory=list)

    def is_used(self) -> bool:
        """True iff usage list is non-empty."""
        return len(self.usage) > 0

    def has_usage(self, tag: str) -> bool:
        """True iff the family's usage list contains *tag*."""
        return tag in self.usage

    def claim(self, tag: str) -> None:
        """Add *tag* to usage list. Idempotent: claiming an existing tag again is a no-op."""
        if tag not in self.usage:
            self.usage.append(tag)


@dataclass
class FamilyRegistry:
    families: dict[str, FamilyEntry] = field(default_factory=dict)

    @classmethod
    def from_path(cls, path: Path | str) -> "FamilyRegistry":
        """Load registry from data/family-registry.json."""

    def to_path(self, path: Path | str) -> None:
        """Write registry to data/family-registry.json (pretty-printed, sorted keys)."""

    def get(self, family_id: str) -> Optional[FamilyEntry]:
        ...

    def claim(self, family_id: str, tag: str) -> None:
        """Claim a tag for a family. Raises KeyError if family_id not in registry."""

    def is_used(self, family_id: str) -> bool:
        ...

    def families_with_usage(self, tag: str) -> list[str]:
        """Return sorted list of family_ids whose usage contains *tag*."""

    def assert_pairwise_disjoint(
        self,
        usages: list[str],
        whitelist: list[tuple[str, str]] | None = None,
    ) -> None:
        """Assert that the family sets for each pair of usage tags in *usages*
        are pairwise disjoint, EXCEPT for pairs listed in *whitelist*.

        Example:
            registry.assert_pairwise_disjoint(
                usages=["p3_train", "p3_validation", "frozen_v3"],
                whitelist=[("p2_train", "p3_train_replay")],
            )

        Raises AssertionError with a message listing the violating family_ids
        and the offending pair.
        """
```

### Important API contract notes
- `claim()` is idempotent — claiming the same tag twice does NOT duplicate the entry. This is verified in the test suite.
- `assert_pairwise_disjoint` checks EACH PAIR (not just 3-way intersection). With `usages=["A","B","C"]`, it asserts `A∩B=∅`, `A∩C=∅`, `B∩C=∅` — three checks, not one.
- The whitelist is a list of `(tag_a, tag_b)` tuples. When checking the pair `(A, B)`, if `(A, B)` or `(B, A)` is in the whitelist, the check is skipped for that pair.
- The registry does NOT enforce immutability of P2 tags — but the builder only ever ADDS tags. Future Tasks (7-9) must follow the same convention.

## scripts/build_family_registry.py CLI

```
python scripts/build_family_registry.py \
    --p2-partition data/p2-curriculum/family-partition.json \
    --mbpp-verified-dir data/external/mbpp/verified \
    --quarantine reports/p3/cross-split-dedup-quarantine.json \
    --output data/family-registry.json
```

Exit codes: 0 = success, 1 = invariant violation or I/O error.

The builder must:
1. Load P2 partition → backfill 374 families with appropriate P2 tags.
2. Load all 3 verified JSONL files → add new families (or merge sample_ids for existing P2 families).
3. Load quarantine JSON → append `"quarantine"` tag to quarantined families.
4. Run all builder invariants above. Abort with exit 1 if any fails.
5. Write `data/family-registry.json` (pretty-printed, sorted keys, ends with newline).
6. Print a summary to stdout: total families, P2 used, quarantined, new available.

## tests/test_family_registry.py

Use synthetic registries built in-memory (no file I/O on real data). The builder test (test_builder_correctness) may use small synthetic input files in `tmp_path`.

Required tests (10):

1. `test_family_entry_is_used_empty_returns_false`: FamilyEntry with empty usage → `is_used()` returns False.
2. `test_family_entry_is_used_nonempty_returns_true`: FamilyEntry with `usage=["p2_train"]` → `is_used()` returns True.
3. `test_claim_idempotent`: calling `entry.claim("p3_train")` twice → usage has exactly one `"p3_train"`.
4. `test_registry_claim_adds_tag`: `registry.claim("mbpp_fam_42", "frozen_v3_candidate")` adds the tag to that family.
5. `test_registry_claim_unknown_family_raises`: `registry.claim("nonexistent", "x")` raises KeyError.
6. `test_families_with_usage_filters_correctly`: registry with 3 families (one p2_train, one quarantine, one empty) → `families_with_usage("quarantine")` returns only the quarantine family.
7. `test_assert_pairwise_disjoint_passes`: registry where the three usage sets are disjoint → no exception.
8. `test_assert_pairwise_disjoint_fails_on_overlap`: registry where one family has both `"p3_train"` and `"p3_validation"` → raises AssertionError with the family_id in the message.
9. `test_assert_pairwise_disjoint_whitelist_allows_overlap`: registry where one family has both `"p2_train"` and `"p3_train_replay"` → `assert_pairwise_disjoint(["p2_train","p3_train_replay"], whitelist=[("p2_train","p3_train_replay")])` does NOT raise.
10. `test_builder_correctness`: build a registry from synthetic P2 partition + synthetic verified JSONL + synthetic quarantine list (use `tmp_path`), assert: total_families matches expected count, P2 tags applied correctly, quarantine tag applied, new families have empty usage.

## Constraints
- Do NOT modify any existing files under `src/` other than creating the new `src/family_registry.py`.
- Do NOT modify any existing files under `tests/`, `scripts/`, `reports/`, or `data/`.
- New file only: `src/family_registry.py`, `scripts/build_family_registry.py`, `tests/test_family_registry.py`, and the generated `data/family-registry.json`.
- Use only stdlib + project-local imports (`from src.schemas import Sample` is OK if needed; do NOT import `src.validators`).
- Single git commit at the end.

## Test Verification
`cd e:\agent\Qwen\qwen3-code-lab ; python -m pytest tests/test_family_registry.py -v`

All 10 tests must pass.

## Run the Builder for Real
After tests pass, run the builder for real on the actual data:
```
python scripts/build_family_registry.py \
    --p2-partition data/p2-curriculum/family-partition.json \
    --mbpp-verified-dir data/external/mbpp/verified \
    --quarantine reports/p3/cross-split-dedup-quarantine.json \
    --output data/family-registry.json
```

The builder must succeed (exit 0) and produce the registry file with these expected counts:
- 955 verified samples = 371 train + 494 test + 90 validation (post-Task 4 verification; 9 rejected samples are NOT in verified JSONL)
- 374 P2 families come from the partition file (all from MBPP train source_split, task_ids 601-974); the 3 train families whose samples were rejected at verify time STILL get their P2 tag because the partition file lists them — P2 history predates the P3 rejection.
- The 371 train-verified families are a subset of the 374 P2 train families (3 families have verified JSONL but their original P2 samples were generated from the pre-verify import).
- Therefore:
  - `total_families` == 374 (P2 train) + 494 (test) + 90 (validation) = **958**
  - `total_p2_used` == **374** (224 p2_train + 75 p2_validation + 75 p2_frozen_v2)
  - `total_quarantined` == **58** (matches Task 5 output)
  - `total_new_available` == 958 - 374 - 58 = **526** (families with empty usage that Task 7 can claim as frozen_v3_candidate)

If the builder's invariants fail, abort and report. Do NOT silently write a broken registry.

## Report File
Write to: `.superpowers/sdd/task-6-report.md`
Include:
- Files created (4 paths)
- total_families / total_p2_used / total_quarantined / total_new_available
- Confirmation that all 4 builder invariants pass
- Confirmation that pairwise disjoint assertion passes for P2 tags (p2_train ∩ p2_validation = ∅, p2_train ∩ p2_frozen_v2 = ∅, p2_validation ∩ p2_frozen_v2 = ∅)
- The commit hash
- Test summary
- Any concerns

## Commit
- Stage: `src/family_registry.py`, `scripts/build_family_registry.py`, `tests/test_family_registry.py`, `data/family-registry.json`
- Commit message: `feat(p3): family registry with P2 backfill + quarantine`
- Single commit.

## Working Directory
e:\agent\Qwen\qwen3-code-lab
