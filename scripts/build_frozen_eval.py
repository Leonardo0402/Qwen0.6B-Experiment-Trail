"""
scripts/build_frozen_eval.py -- Build the expanded P1 frozen evaluation set.

Loads the existing 6-family ``data/splits/test_raw.jsonl`` as the frozen base,
then expands it with *untrained* families drawn from the project's own data
pool (``data/verified/code_gen.jsonl`` + ``data/verified/repairs.jsonl``).

Anti-leakage contract
---------------------
* No external datasets are ever touched.
* No family_id may appear in BOTH training and the frozen eval.  Train
  families are detected from (in priority order):

    1. ``data/curriculum/{easy,boundary,repair}/families.json`` -- the
       canonical record of which families entered each curriculum stage.
    2. ``data/splits/train.jsonl`` -- a robust fallback that recovers
       family_id by matching each ChatML assistant message's target_code back
       to the verified pool.  This runs whenever train.jsonl is present,
       regardless of whether the curriculum files exist, so the no-overlap
       guarantee is enforced "within detectable range" even before the
       curriculum sub-task has produced its families.json files.

* The original ``data/splits/test_raw.jsonl`` is never modified; its 36 lines
  are preserved verbatim as the first 36 lines of the frozen output for full
  backward compatibility.

Usage
-----
    python scripts/build_frozen_eval.py

Exit codes
----------
    0   success (frozen set built; may include a shortfall report)
    1   fatal error (missing inputs, validation failure, etc.)
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.schemas import Sample  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED = 42
VERSION = "v1"

TEST_RAW_PATH = _ROOT / "data" / "splits" / "test_raw.jsonl"
TRAIN_PATH = _ROOT / "data" / "splits" / "train.jsonl"
CODE_GEN_PATH = _ROOT / "data" / "verified" / "code_gen.jsonl"
REPAIRS_PATH = _ROOT / "data" / "verified" / "repairs.jsonl"

CURRICULUM_DIRS = [
    _ROOT / "data" / "curriculum" / "easy",
    _ROOT / "data" / "curriculum" / "boundary",
    _ROOT / "data" / "curriculum" / "repair",
]

OUT_DIR = _ROOT / "data" / "frozen-eval" / VERSION

# Per-new-family selection target: 1 code_generation + up to 2 static_repair +
# up to 2 execution_repair (covers all three task_types, stays in 3-6 range).
NEW_FAMILY_PER_TASK_TYPE = {
    "code_generation": 1,
    "static_repair": 2,
    "execution_repair": 2,
}

TARGET_FAMILY_COUNT = 12

# ChatML assistant fence tokens used to reverse target_code matching.
_BT = chr(96)
_FENCE_OPEN = _BT * 3 + "python\n"
_FENCE_CLOSE = "\n" + _BT * 3


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------

def _load_jsonl_raw(path: Path) -> list[str]:
    """Return the non-blank stripped lines of a JSONL file."""
    lines: list[str] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                lines.append(line)
    return lines


def _load_samples(path: Path) -> list[Sample]:
    """Load and validate Sample objects from a JSONL file."""
    samples: list[Sample] = []
    for line in _load_jsonl_raw(path):
        samples.append(Sample.from_json_line(line))
    return samples


# ---------------------------------------------------------------------------
# Train-family detection
# ---------------------------------------------------------------------------

def _extract_family_ids_from_curriculum_file(path: Path) -> set[str]:
    """Best-effort extraction of family_id strings from a families.json file.

    Supports several plausible shapes:
      * ["fam_xxx", ...]
      * {"family_ids": ["fam_xxx", ...]}
      * {"families": ["fam_xxx", ...]}
      * {"fam_xxx": <anything>, ...}
    """
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)

    if isinstance(data, list):
        return {str(x) for x in data}
    if isinstance(data, dict):
        for key in ("family_ids", "families", "family_id", "ids"):
            if key in data and isinstance(data[key], list):
                return {str(x) for x in data[key]}
        # Fall back to treating top-level keys as family_ids.
        return {str(k) for k in data.keys()}
    return set()


def load_curriculum_train_families() -> tuple[set[str], list[str]]:
    """Collect family_ids listed in any curriculum ``families.json``.

    Returns
    -------
    (family_ids, sources)
        ``sources`` lists the relative paths that were actually read, for
        provenance reporting.  Missing files are silently skipped.
    """
    train_fams: set[str] = set()
    sources: list[str] = []
    for d in CURRICULUM_DIRS:
        fp = d / "families.json"
        if fp.exists():
            try:
                train_fams |= _extract_family_ids_from_curriculum_file(fp)
                sources.append(str(fp.relative_to(_ROOT)).replace("\\", "/"))
            except (json.JSONDecodeError, OSError):
                # Ignore malformed curriculum files rather than aborting.
                pass
    return train_fams, sources


def build_target_code_to_family(samples: list[Sample]) -> dict[str, str]:
    """Map stripped target_code -> family_id from verified samples.

    On collision (two families share identical target_code) the first family
    wins; such collisions are essentially impossible for these distinct
    function families but the behaviour is deterministic regardless.
    """
    tc2fam: dict[str, str] = {}
    for s in samples:
        key = s.target_code.strip()
        tc2fam.setdefault(key, s.family_id)
    return tc2fam


def detect_train_families_from_train_jsonl(
    tc2fam: dict[str, str],
) -> tuple[set[str], int]:
    """Recover train family_ids by matching train.jsonl assistant content.

    Each ChatML assistant message wraps the target_code in a ```python fence.
    We strip the fence and look the code up in *tc2fam*.

    Returns
    -------
    (family_ids, unmatched_line_count)
    """
    if not TRAIN_PATH.exists():
        return set(), 0
    train_fams: set[str] = set()
    unmatched = 0
    for line in _load_jsonl_raw(TRAIN_PATH):
        try:
            rec = json.loads(line)
            ac = rec["messages"][2]["content"]
        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
            unmatched += 1
            continue
        if isinstance(ac, str) and ac.startswith(_FENCE_OPEN) and ac.endswith(_FENCE_CLOSE):
            code = ac[len(_FENCE_OPEN):-len(_FENCE_CLOSE)].strip()
            fam = tc2fam.get(code)
            if fam is not None:
                train_fams.add(fam)
            else:
                unmatched += 1
        else:
            unmatched += 1
    return train_fams, unmatched


# ---------------------------------------------------------------------------
# Candidate selection
# ---------------------------------------------------------------------------

def select_samples_for_new_family(
    samples: list[Sample], family_id: str
) -> list[Sample]:
    """Pick a balanced, deterministic subset for one new family.

    Selects up to NEW_FAMILY_PER_TASK_TYPE samples per task_type, sorted by
    sample_id for reproducibility.  Always tries to cover all three
    task_types so the frozen set exercises generation + both repair modes.
    """
    by_tt: dict[str, list[Sample]] = {"code_generation": [],
                                      "static_repair": [],
                                      "execution_repair": []}
    for s in samples:
        if s.family_id == family_id and s.task_type in by_tt:
            by_tt[s.task_type].append(s)

    chosen: list[Sample] = []
    for tt in ("code_generation", "static_repair", "execution_repair"):
        bucket = sorted(by_tt[tt], key=lambda s: s.sample_id)
        limit = NEW_FAMILY_PER_TASK_TYPE[tt]
        chosen.extend(bucket[:limit])
    return chosen


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def _sha256_of_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _task_type_distribution(samples: list[Sample]) -> dict[str, int]:
    dist: dict[str, int] = {
        "code_generation": 0,
        "static_repair": 0,
        "execution_repair": 0,
    }
    for s in samples:
        dist[s.task_type] = dist.get(s.task_type, 0) + 1
    return dist


def _difficulty_distribution(samples: list[Sample]) -> dict[str, int]:
    dist: dict[str, int] = {}
    for s in samples:
        key = str(s.difficulty)
        dist[key] = dist.get(key, 0) + 1
    return dist


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------

def build_frozen_eval() -> int:
    """Build the frozen eval set; return process exit code."""
    # --- 1. Sanity: required inputs ---------------------------------------
    required = [TEST_RAW_PATH, CODE_GEN_PATH, REPAIRS_PATH]
    for p in required:
        if not p.exists():
            print(f"ERROR: required input missing: {p}", file=sys.stderr)
            return 1

    # --- 2. Load + validate test_raw (frozen base, backward compat) -------
    test_raw_lines = _load_jsonl_raw(TEST_RAW_PATH)
    test_raw_samples: list[Sample] = []
    for line in test_raw_lines:
        test_raw_samples.append(Sample.from_json_line(line))  # validates
    test_raw_families = {s.family_id for s in test_raw_samples}
    print(f"test_raw: {len(test_raw_samples)} samples / "
          f"{len(test_raw_families)} families")

    # --- 3. Load + validate verified pool ---------------------------------
    verified: list[Sample] = []
    for p in (CODE_GEN_PATH, REPAIRS_PATH):
        verified.extend(_load_samples(p))
    # Validate every sample explicitly (defence in depth).
    for s in verified:
        Sample.model_validate(s.model_dump(mode="json"))
    print(f"verified pool: {len(verified)} samples / "
          f"{len({s.family_id for s in verified})} families")

    verified_families = {s.family_id for s in verified}
    tc2fam = build_target_code_to_family(verified)

    # --- 4. Detect train families -----------------------------------------
    excluded_train_families: set[str] = set()
    curriculum_sources: list[str] = []

    curr_fams, curriculum_sources = load_curriculum_train_families()
    if curr_fams:
        excluded_train_families |= curr_fams
        print(f"curriculum families.json: excluded {len(curr_fams)} train "
              f"family_ids (sources: {curriculum_sources})")
    else:
        print("NOTE: 未检测到 curriculum families.json，"
              "无法通过 curriculum 排除训练 family；改用 train.jsonl 反查。")

    # Robust fallback / confirmation: recover train families from train.jsonl.
    train_fams_from_jsonl, unmatched = detect_train_families_from_train_jsonl(tc2fam)
    if train_fams_from_jsonl:
        print(f"train.jsonl 反查: 检测到 {len(train_fams_from_jsonl)} 个训练 "
              f"family_id (unmatched lines: {unmatched})")
        excluded_train_families |= train_fams_from_jsonl
    elif TRAIN_PATH.exists():
        print(f"WARNING: train.jsonl 存在但未匹配到任何 family "
              f"(unmatched lines: {unmatched})")

    # --- 5. Determine candidate (untrained, not already in test) families -
    candidates = (verified_families - test_raw_families) - excluded_train_families
    print(f"候选扩展 family (verified 且未训练 且未在 test_raw): "
          f"{len(candidates)}")
    if candidates:
        for f in sorted(candidates):
            print(f"  - {f}")

    # --- 6. Select balanced samples for each candidate family -------------
    new_samples: list[Sample] = []
    for fam in sorted(candidates):
        chosen = select_samples_for_new_family(verified, fam)
        new_samples.extend(chosen)
        print(f"  选取 {f}: {len(chosen)} 条 "
              f"({{{', '.join(sorted({s.task_type for s in chosen}))}}})")

    # --- 7. Merge: preserve test_raw verbatim, then append new samples ----
    # Dedup by sample_id defensively (test_raw and new families are disjoint
    # by construction, so this is a no-op safety net).
    output_lines: list[str] = list(test_raw_lines)
    seen_ids = {s.sample_id for s in test_raw_samples}
    for s in new_samples:
        if s.sample_id in seen_ids:
            continue
        seen_ids.add(s.sample_id)
        output_lines.append(s.to_json_line())

    # Final validation of the merged set.
    merged_samples: list[Sample] = []
    for line in output_lines:
        merged_samples.append(Sample.from_json_line(line))

    # --- 8. Write outputs -------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    out_test_raw = OUT_DIR / "test_raw.jsonl"
    with out_test_raw.open("w", encoding="utf-8", newline="\n") as fh:
        for line in output_lines:
            fh.write(line + "\n")

    sha256 = _sha256_of_file(out_test_raw)
    family_ids = sorted({s.family_id for s in merged_samples})
    task_dist = _task_type_distribution(merged_samples)
    diff_dist = _difficulty_distribution(merged_samples)

    # families.json
    families_info: list[dict] = []
    by_family: dict[str, int] = {}
    for s in merged_samples:
        by_family[s.family_id] = by_family.get(s.family_id, 0) + 1
    for fam in family_ids:
        families_info.append({"family_id": fam, "sample_count": by_family[fam]})

    families_json_path = OUT_DIR / "families.json"
    with families_json_path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(
            {"version": VERSION, "family_count": len(family_ids),
             "families": families_info},
            fh, indent=2, ensure_ascii=False,
        )

    # manifest.json
    now = _now_iso()
    manifest = {
        "version": VERSION,
        "created_at": now,
        "seed": SEED,
        "source_files": [
            "data/splits/test_raw.jsonl",
            "data/verified/code_gen.jsonl",
            "data/verified/repairs.jsonl",
        ],
        "test_raw_sha256": sha256,
        "sample_count": len(merged_samples),
        "family_count": len(family_ids),
        "task_type_distribution": task_dist,
        "difficulty_distribution": diff_dist,
        "families": family_ids,
        "excluded_train_families": sorted(excluded_train_families),
        "frozen_at": now,
        "notes": (
            "P1 expanded frozen eval set; only uses existing untrained "
            "families from data pool. Train families detected via "
            f"curriculum files ({curriculum_sources if curriculum_sources else 'none'}) "
            "and train.jsonl target_code reverse-matching."
        ),
    }
    manifest_path = OUT_DIR / "manifest.json"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    # README.md
    readme_path = OUT_DIR / "README.md"
    readme_lines = [
        "# Frozen Eval Set (v1)",
        "",
        "本目录为 P1 阶段冻结评测集，**一经冻结不得修改**。",
        "",
        "## 摘要",
        "",
        f"- 版本: `{VERSION}`",
        f"- 样本数: **{len(merged_samples)}**",
        f"- 独立 family 数: **{len(family_ids)}**",
        f"- SHA256 (`test_raw.jsonl`): `{sha256}`",
        f"- 冻结时间 (UTC): {now}",
        "",
        "## task_type 分布",
        "",
        "| task_type | 数量 |",
        "|---|---|",
    ]
    for tt in ("code_generation", "static_repair", "execution_repair"):
        readme_lines.append(f"| {tt} | {task_dist.get(tt, 0)} |")
    readme_lines += [
        "",
        "## difficulty 分布",
        "",
        "| difficulty | 数量 |",
        "|---|---|",
    ]
    for d in sorted(diff_dist.keys(), key=lambda x: int(x)):
        readme_lines.append(f"| {d} | {diff_dist[d]} |")
    readme_lines += [
        "",
        "## family 列表",
        "",
    ]
    for fam in family_ids:
        readme_lines.append(f"- `{fam}` ({by_family[fam]} 条)")
    readme_lines += [
        "",
        "## 数据来源",
        "",
        "- `data/splits/test_raw.jsonl` — 原 36 条冻结测试样本 (6 family)，"
        "逐行原样保留以保证向后兼容。",
        "- `data/verified/code_gen.jsonl` — code_generation 样本池。",
        "- `data/verified/repairs.jsonl` — static_repair / execution_repair 样本池。",
        "",
        "## 防泄漏保证",
        "",
        "- **严禁接入外部数据集**；仅使用项目现有数据池。",
        "- **严禁** family_id 同时出现在训练集与冻结评测集。训练 family 通过"
        "`data/curriculum/*/families.json`（如存在）与 `data/splits/train.jsonl` "
        "的 target_code 反查联合检测并排除。",
        f"- 已排除的训练 family 数: {len(excluded_train_families)}",
        "",
        "## 冻结约束",
        "",
        "- **禁止**后续训练使用本目录任何样本。",
        "- **禁止**根据模型在本集上的表现修改题目、target_code 或测试用例。",
        "- 如需扩展或修订，**必须创建新版本目录**（如 `v2`），不得原地修改 `v1`。",
        "",
        "## 目标达成情况",
        "",
        f"- P1 目标: >= {TARGET_FAMILY_COUNT} 个独立 family_id",
        f"- 实际: {len(family_ids)} 个 family_id",
    ]
    if len(family_ids) < TARGET_FAMILY_COUNT:
        readme_lines += [
            f"- **未达标**: 数据池仅含 {len(verified_families)} 个 family，"
            f"其中 {len(excluded_train_families)} 个已进入训练，"
            f"可用的未训练 family 不足 {TARGET_FAMILY_COUNT}。"
            "如实报告，未编造任何 family。",
            "",
            f"  - test_raw 原有 family: {len(test_raw_families)}",
            f"  - 新增未训练 family: {len(candidates)}",
            f"  - 合计: {len(family_ids)}",
        ]
    else:
        readme_lines.append(f"- 已达标 (>= {TARGET_FAMILY_COUNT})。")
    readme_lines.append("")

    with readme_path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(readme_lines))

    # --- 9. Final report --------------------------------------------------
    print("=" * 60)
    print("FROZEN EVAL v1 BUILD COMPLETE")
    print("=" * 60)
    print(f"output dir       : {OUT_DIR}")
    print(f"sample_count     : {len(merged_samples)}")
    print(f"family_count     : {len(family_ids)}")
    print(f"task_type_dist   : {task_dist}")
    print(f"difficulty_dist  : {diff_dist}")
    print(f"test_raw_sha256  : {sha256}")
    print(f"excluded_train   : {len(excluded_train_families)} families")
    if len(family_ids) < TARGET_FAMILY_COUNT:
        print(f"SHORTFALL: 目标 {TARGET_FAMILY_COUNT} family，"
              f"实际 {len(family_ids)} family "
              f"(数据池可用未训练 family 不足，已如实报告)")
    else:
        print(f"OK: 达到 {TARGET_FAMILY_COUNT} family 目标")
    return 0


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")  # type: ignore[attr-defined]
    except Exception:
        pass
    return build_frozen_eval()


if __name__ == "__main__":
    sys.exit(main())
