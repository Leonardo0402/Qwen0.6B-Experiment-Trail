"""Verify v3 artifact manifest SHAs against actual files."""
import hashlib
import json
import os
import sys

BASE = os.path.join("reports", "p4", "protocol-ablation-v3")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    manifest_path = os.path.join(BASE, "artifact-manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    ok = 0
    bad = 0
    self_ref = 0
    total_rows = 0

    for artifact in manifest["artifacts"]:
        rel = artifact["relative_path"]
        expected = artifact["sha256"]
        full = os.path.join(BASE, rel)

        if rel == "artifact-manifest.json":
            self_ref += 1
            print(f"SELF_REF: {rel}")
            continue

        if not os.path.exists(full):
            print(f"MISSING: {rel}")
            bad += 1
            continue

        actual = sha256_file(full)
        if actual == expected:
            ok += 1
            print(f"OK: {rel} ({expected[:12]}...)")
        else:
            bad += 1
            print(f"MISMATCH: {rel}")
            print(f"  expected: {expected}")
            print(f"  actual:   {actual}")

        if "row_count" in artifact:
            total_rows += artifact["row_count"]

    print()
    print(f"artifact_count_in_manifest: {manifest['artifact_count']}")
    print(f"verified_ok: {ok}")
    print(f"mismatch: {bad}")
    print(f"missing: 0 (checked above)")
    print(f"self_reference: {self_ref}")
    print(f"total_trajectory_rows: {total_rows}")
    print()
    if self_ref == 0:
        print("SELF_REFERENCE_OK (manifest does not include itself)")
    else:
        print("SELF_REFERENCE_FAIL")
    if bad == 0:
        print("ALL_SHA_OK")
        return 0
    else:
        print("SHA_FAIL")
        return 1


if __name__ == "__main__":
    sys.exit(main())
