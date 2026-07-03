"""tests/test_p2_evidence_hardening.py -- Issue #1 P0-7 automated guards.

These tests enforce the evidence-chain invariants required by Issue #1:

1. ``data/p2-curriculum/frozen-eval-v2/`` MUST NOT contain
   ``train.jsonl`` or ``validation.jsonl`` (legacy duplicates of
   ``test_raw.jsonl`` that were mistakenly counted as training data
   in the old audit).

2. ``reports/p2/dataset-audit.json`` totals.train MUST equal
   84 + 280 + 560 = 924 (frozen-eval-v2 NOT counted as train).

3. ``reports/p2/adapter-evidence.json`` MUST contain the explicit
   ``parent_adapter_weight_sha256`` and ``parent_adapter_config_sha256``
   fields (Issue #1 P0-6 split). The legacy ``parent_adapter_sha256``
   field is kept as an alias of ``parent_adapter_config_sha256``.

4. Parent adapter weight SHA chain MUST be verifiable:
   Stage(N).parent_adapter_weight_sha256 == Stage(N-1).weight_sha256.

5. No training config under ``configs/curriculum/p2-*.yaml`` may
   reference a path that contains ``frozen-eval`` as train_file.

6. The fixed stratified-120 subset MUST exist with a manifest that
   records sample_id / family_id / task_type / seed / SHA256.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


FE_DIR = _ROOT / "data" / "p2-curriculum" / "frozen-eval-v2"
STRATIFIED_DIR = FE_DIR / "stratified-120"
ADAPTER_EVIDENCE = _ROOT / "reports" / "p2" / "adapter-evidence.json"
DATASET_AUDIT = _ROOT / "reports" / "p2" / "dataset-audit.json"
CONFIGS_DIR = _ROOT / "configs" / "curriculum"


# ---------------------------------------------------------------------------
# P0-2: frozen-eval-v2 must not contain train.jsonl / validation.jsonl
# ---------------------------------------------------------------------------

class TestFrozenEvalPurity:
    """Frozen-eval-v2 is an EVALUATION set; never a training set."""

    def test_no_train_jsonl_in_frozen_eval(self):
        """train.jsonl was a legacy duplicate of test_raw.jsonl (576 lines).
        Deleting it prevents audit scripts from counting frozen-eval as
        training data (the 1500 -> 924 bug in Issue #1 P0-3)."""
        assert not (FE_DIR / "train.jsonl").exists(), (
            "frozen-eval-v2/train.jsonl must NOT exist. It duplicates "
            "test_raw.jsonl and was incorrectly counted as training data."
        )

    def test_no_validation_jsonl_in_frozen_eval(self):
        """validation.jsonl was an empty legacy file. Removed in P0-2."""
        assert not (FE_DIR / "validation.jsonl").exists(), (
            "frozen-eval-v2/validation.jsonl must NOT exist (legacy empty file)."
        )

    def test_test_raw_jsonl_exists(self):
        """The canonical evaluation file must be present."""
        assert (FE_DIR / "test_raw.jsonl").exists()

    def test_manifest_uses_test_sha256_not_train_sha256(self):
        """manifest.json must use test_sha256 / test_families, not the
        misleading train_sha256 / train_families (which imply training)."""
        m = json.load((FE_DIR / "manifest.json").open(encoding="utf-8"))
        assert "test_sha256" in m, "manifest must have test_sha256 (P0-2 fix)"
        assert m.get("sample_counts", {}).get("train", -1) == 0, (
            "frozen-eval-v2 manifest sample_counts.train must be 0 "
            "(it is an evaluation set, not a training set)"
        )
        assert "test_families" in m, (
            "manifest must record test_families (the families the eval uses)"
        )


# ---------------------------------------------------------------------------
# P0-3: dataset-audit.json totals must be 924 train, not 1500
# ---------------------------------------------------------------------------

class TestDatasetAuditTotals:
    """Audit must not count frozen-eval-v2 samples as training data."""

    def test_audit_file_exists(self):
        assert DATASET_AUDIT.exists(), (
            "dataset-audit.json not found. Run scripts/audit_p2_dataset.py"
        )

    def test_total_train_equals_924(self):
        """84 (stage1) + 280 (stage2) + 560 (stage3) = 924 train samples.
        frozen-eval-v2 must NOT be counted here."""
        a = json.load(DATASET_AUDIT.open(encoding="utf-8"))
        assert a["totals"]["train"] == 924, (
            f"Expected totals.train=924, got {a['totals']['train']}. "
            f"frozen-eval-v2 samples are leaking into the train count."
        )

    def test_frozen_eval_not_in_train_totals(self):
        """frozen-eval-v2 should be recorded under 'test', not 'train'."""
        a = json.load(DATASET_AUDIT.open(encoding="utf-8"))
        fe = a["stages"]["frozen-eval-v2"]
        assert fe["sample_counts"]["train"] == 0
        assert fe["sample_counts"]["test"] == 576

    def test_totals_have_test_field(self):
        """Audit totals must include a 'test' field (576 frozen-eval)."""
        a = json.load(DATASET_AUDIT.open(encoding="utf-8"))
        assert a["totals"].get("test") == 576


# ---------------------------------------------------------------------------
# P0-5: stratified-120 subset must exist with manifest
# ---------------------------------------------------------------------------

class TestStratified120Subset:
    """The 120-sample stratified evaluation must be fixed (Issue #1 P0-5)."""

    def test_subset_file_exists(self):
        assert (STRATIFIED_DIR / "test_raw.jsonl").exists()

    def test_subset_manifest_exists(self):
        assert (STRATIFIED_DIR / "manifest.json").exists()

    def test_subset_has_120_samples(self):
        m = json.load((STRATIFIED_DIR / "manifest.json").open(encoding="utf-8"))
        assert m["total_samples"] == 120
        assert len(m["sample_ids"]) == 120

    def test_subset_balanced_task_types(self):
        m = json.load((STRATIFIED_DIR / "manifest.json").open(encoding="utf-8"))
        counts = m["task_type_counts"]
        assert counts["code_generation"] == 40
        assert counts["static_repair"] == 40
        assert counts["execution_repair"] == 40

    def test_subset_manifest_records_sha_and_seed(self):
        m = json.load((STRATIFIED_DIR / "manifest.json").open(encoding="utf-8"))
        assert m.get("subset_sha256"), "manifest must record subset_sha256"
        assert m.get("seed") == 42, "manifest must record seed"
        assert m.get("family_ids"), "manifest must record family_ids"


# ---------------------------------------------------------------------------
# P0-6: adapter evidence weight + config SHA split
# ---------------------------------------------------------------------------

class TestAdapterEvidenceSplit:
    """parent_adapter_sha256 must be split into weight + config SHAs."""

    def test_evidence_file_exists(self):
        assert ADAPTER_EVIDENCE.exists()

    @pytest.fixture
    def evidence(self):
        return json.load(ADAPTER_EVIDENCE.open(encoding="utf-8"))

    def test_explicit_weight_and_config_sha_fields(self, evidence):
        """Each stage must record parent_adapter_weight_sha256 and
        parent_adapter_config_sha256 (P0-6 split)."""
        for stage in ("stage1-code", "stage2-boundary", "stage3-repair"):
            ev = evidence[stage]
            assert "parent_adapter_weight_sha256" in ev, (
                f"{stage} missing parent_adapter_weight_sha256"
            )
            assert "parent_adapter_config_sha256" in ev, (
                f"{stage} missing parent_adapter_config_sha256"
            )

    def test_stage1_has_no_parent(self, evidence):
        """Stage 1 trains from base, so parent_adapter_weight_sha256
        must be None."""
        assert evidence["stage1-code"]["parent_adapter_weight_sha256"] is None
        assert evidence["stage1-code"]["parent_adapter_config_sha256"] is None

    def test_stage2_parent_weight_matches_stage1_weight(self, evidence):
        """Stage2.parent_adapter_weight_sha256 must equal Stage1.weight_sha256
        (the actual adapter_model.safetensors SHA, not config)."""
        s1_w = evidence["stage1-code"]["weight_sha256"]
        s2_pw = evidence["stage2-boundary"]["parent_adapter_weight_sha256"]
        assert s2_pw == s1_w, (
            f"Stage2 parent weight SHA {s2_pw} != Stage1 weight SHA {s1_w}"
        )

    def test_stage3_parent_weight_matches_stage2_weight(self, evidence):
        s2_w = evidence["stage2-boundary"]["weight_sha256"]
        s3_pw = evidence["stage3-repair"]["parent_adapter_weight_sha256"]
        assert s3_pw == s2_w, (
            f"Stage3 parent weight SHA {s3_pw} != Stage2 weight SHA {s2_w}"
        )

    def test_all_adapter_weights_different(self, evidence):
        """Each stage's adapter weight SHA must differ (no overwrite)."""
        hashes = [
            evidence["stage1-code"]["weight_sha256"],
            evidence["stage2-boundary"]["weight_sha256"],
            evidence["stage3-repair"]["weight_sha256"],
        ]
        assert len(set(hashes)) == 3, f"Adapter weights not unique: {hashes}"

    def test_legacy_field_is_alias_of_config_sha(self, evidence):
        """parent_adapter_sha256 (legacy) must equal
        parent_adapter_config_sha256 for backward compatibility."""
        for stage in ("stage2-boundary", "stage3-repair"):
            ev = evidence[stage]
            assert ev["parent_adapter_sha256"] == ev["parent_adapter_config_sha256"]


# ---------------------------------------------------------------------------
# P0-2: training configs must not reference frozen-eval
# ---------------------------------------------------------------------------

class TestTrainingConfigsDontReadFrozenEval:
    """No P2 training config may point train_file or eval_file at the
    frozen-eval directory (hard guard in train_lora.py enforces this
    at runtime; this test enforces it at config level too)."""

    @pytest.fixture
    def p2_configs(self):
        return sorted(CONFIGS_DIR.glob("p2-*.yaml"))

    def test_p2_configs_exist(self, p2_configs):
        assert len(p2_configs) >= 6, f"Expected >=6 P2 configs, got {len(p2_configs)}"

    def test_no_train_file_references_frozen_eval(self, p2_configs):
        import yaml
        for cfg_path in p2_configs:
            cfg = yaml.safe_load(cfg_path.open(encoding="utf-8"))
            train_file = str(cfg.get("train_file", ""))
            eval_file = str(cfg.get("eval_file", ""))
            for label, p in (("train_file", train_file), ("eval_file", eval_file)):
                assert "frozen-eval" not in p.replace("\\", "/"), (
                    f"{cfg_path.name}: {label}={p} references frozen-eval. "
                    f"Training data must come from stage directories."
                )

    def test_no_train_file_is_named_test_raw(self, p2_configs):
        import yaml
        for cfg_path in p2_configs:
            cfg = yaml.safe_load(cfg_path.open(encoding="utf-8"))
            train_file = Path(str(cfg.get("train_file", "")))
            assert train_file.name != "test_raw.jsonl", (
                f"{cfg_path.name}: train_file must not be test_raw.jsonl"
            )


# ---------------------------------------------------------------------------
# Issue #1 fix: DAG parent-chain verification (branch graph, not linear)
# ---------------------------------------------------------------------------

class TestAdapterDAGVerification:
    """Verify adapter parent chain as a DAG (branch graph), not linear.

    DAG structure:
        stage1-code (root)
          └─ stage2-boundary (continual)
               ├─ stage3-repair (continual main chain)
               └─ stage3-v3-antiforget (continual branch from Stage2)
        independent-stage3 (root, no parent, independent from Base)

    The old linear verification assumed stages[i].parent == stages[i-1],
    which is wrong for branches (independent-stage3 and stage3-v3-antiforget).
    """

    @pytest.fixture
    def evidence(self):
        return json.load(ADAPTER_EVIDENCE.open(encoding="utf-8"))

    def test_verification_block_exists(self, evidence):
        """_verification block must exist with DAG fields."""
        assert "_verification" in evidence, "Missing _verification block"
        v = evidence["_verification"]
        assert "parent_chain_verified" in v
        assert "verification_mode" in v
        assert "dag_edges" in v

    def test_parent_chain_verified_is_true(self, evidence):
        """All DAG edges must pass verification."""
        assert evidence["_verification"]["parent_chain_verified"] is True

    def test_verification_mode_is_dag(self, evidence):
        """Verification mode must indicate DAG (not linear)."""
        mode = evidence["_verification"]["verification_mode"]
        assert "DAG" in mode, f"Expected DAG mode, got {mode}"

    def test_dag_edges_cover_all_5_adapters(self, evidence):
        """dag_edges must have one entry per adapter (5 total)."""
        edges = evidence["_verification"]["dag_edges"]
        assert len(edges) == 5, f"Expected 5 DAG edges, got {len(edges)}"

    def test_all_dag_edges_match(self, evidence):
        """Every DAG edge must have weight_match=True and config_match=True."""
        edges = evidence["_verification"]["dag_edges"]
        for e in edges:
            assert e["weight_match"] is True, (
                f"{e['child']} -> {e['parent']}: weight_match=False"
            )
            assert e["config_match"] is True, (
                f"{e['child']} -> {e['parent']}: config_match=False"
            )

    def test_independent_stage3_is_root(self, evidence):
        """Independent Stage3 has no parent (root node, independent from Base)."""
        ev = evidence["independent-stage3"]
        assert ev["parent_adapter_path"] is None
        assert ev["parent_adapter_weight_sha256"] is None
        assert ev["parent_adapter_config_sha256"] is None

    def test_stage1_is_root(self, evidence):
        """Stage1 is root (trains from base model, no parent adapter)."""
        ev = evidence["stage1-code"]
        assert ev["parent_adapter_path"] is None
        assert ev["parent_adapter_weight_sha256"] is None
        assert ev["parent_adapter_config_sha256"] is None

    def test_antiforget_parent_is_stage2(self, evidence):
        """stage3-v3-antiforget branches from stage2-boundary (NOT stage3-repair).

        This is the key DAG test: the old linear verification wrongly checked
        antiforget -> independent-stage3, but the real parent is stage2-boundary.
        """
        ev = evidence["stage3-v3-antiforget"]
        parent_path = ev["parent_adapter_path"]
        assert parent_path is not None
        assert "stage2-boundary" in parent_path.replace("\\", "/"), (
            f"antiforget parent should be stage2-boundary, got {parent_path}"
        )

    def test_antiforget_parent_weight_matches_stage2(self, evidence):
        """antiforget.parent_adapter_weight_sha256 == stage2.weight_sha256."""
        s2_w = evidence["stage2-boundary"]["weight_sha256"]
        af_pw = evidence["stage3-v3-antiforget"]["parent_adapter_weight_sha256"]
        assert af_pw == s2_w, (
            f"antiforget parent weight {af_pw} != Stage2 weight {s2_w}"
        )

    def test_antiforget_parent_config_matches_stage2(self, evidence):
        """antiforget.parent_adapter_config_sha256 == stage2.config_sha256."""
        s2_c = evidence["stage2-boundary"]["config_sha256"]
        af_pc = evidence["stage3-v3-antiforget"]["parent_adapter_config_sha256"]
        assert af_pc == s2_c, (
            f"antiforget parent config {af_pc} != Stage2 config {s2_c}"
        )

    def test_all_5_adapter_weights_different(self, evidence):
        """All 5 adapter weight SHA256 must differ (no overwrite)."""
        hashes = [
            evidence["stage1-code"]["weight_sha256"],
            evidence["stage2-boundary"]["weight_sha256"],
            evidence["stage3-repair"]["weight_sha256"],
            evidence["independent-stage3"]["weight_sha256"],
            evidence["stage3-v3-antiforget"]["weight_sha256"],
        ]
        assert len(set(hashes)) == 5, f"Adapter weights not unique: {hashes}"

    def test_dag_has_no_cycle(self, evidence):
        """DAG must be acyclic (topological sort succeeds).

        Walk parent edges; must reach a root (parent=None) without revisiting.
        """
        edges = evidence["_verification"]["dag_edges"]
        parent_of = {e["child"]: e["parent"] for e in edges}

        for child in parent_of:
            visited = set()
            node = child
            while node is not None:
                assert node not in visited, f"Cycle detected at {node}"
                visited.add(node)
                node = parent_of.get(node)
            # Must have reached a root (parent=None terminates the loop)

    def test_independent_stage3_weight_differs_from_stage3_repair(self, evidence):
        """Independent Stage3 and Continual Stage3 must have different weights.

        They use the same data but different training paths (independent vs
        continual), so their adapter weights must differ.
        """
        indep_w = evidence["independent-stage3"]["weight_sha256"]
        cont_w = evidence["stage3-repair"]["weight_sha256"]
        assert indep_w != cont_w, (
            "Independent Stage3 weight == Continual Stage3 weight "
            "(they should differ due to different parent paths)"
        )
