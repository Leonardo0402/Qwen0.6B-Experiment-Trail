"""Update all P2 manifest SHA256 after instruction augmentation."""
import hashlib
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    base = _ROOT / "data" / "p2-curriculum"

    for stage in ["stage1-code", "stage2-boundary", "stage3-repair"]:
        for split in ["train", "validation"]:
            f = base / stage / f"{split}.jsonl"
            if f.exists():
                sha = sha256_file(f)
                m_path = base / stage / "manifest.json"
                m = json.load(open(m_path))
                key = f"{split}_sha256"
                old = m.get(key, "")
                m[key] = sha
                with open(m_path, "w") as fh:
                    json.dump(m, fh, indent=2, ensure_ascii=False)
                old_short = old[:32] if old else "none"
                print(f"{stage}/{split}: {sha[:32]} (was: {old_short})")

    # frozen-eval
    for split in ["train", "test_raw"]:
        f = base / "frozen-eval-v2" / f"{split}.jsonl"
        if f.exists():
            sha = sha256_file(f)
            m_path = base / "frozen-eval-v2" / "manifest.json"
            m = json.load(open(m_path))
            old = m.get("train_sha256", "")
            m["train_sha256"] = sha
            with open(m_path, "w") as fh:
                json.dump(m, fh, indent=2, ensure_ascii=False)
            old_short = old[:32] if old else "none"
            print(f"frozen-eval/{split}: {sha[:32]} (was: {old_short})")


if __name__ == "__main__":
    main()
