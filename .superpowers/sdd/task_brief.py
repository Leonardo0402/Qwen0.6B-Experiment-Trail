"""Python equivalent of the SDD task-brief bash script.

Extracts one task's full text from a plan into a uniquely named file.

Usage: py -3.11 task_brief.py PLAN_FILE TASK_NUMBER
"""
import re
import sys
from pathlib import Path


def extract_task(plan_path: Path, task_num: int) -> str:
    text = plan_path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)

    # Find task headers (## Task N) — not inside code fences
    in_fence = False
    task_starts = []  # (line_index, task_number)
    for i, line in enumerate(lines):
        if line.startswith("```"):
            in_fence = not in_fence
        if not in_fence and line.startswith("## "):
            m = re.match(r"^##\s+Task\s+(\d+)\b", line)
            if m:
                task_starts.append((i, int(m.group(1))))

    # Find the requested task
    target_start = None
    next_start = len(lines)
    for idx, (line_idx, num) in enumerate(task_starts):
        if num == task_num:
            target_start = line_idx
            if idx + 1 < len(task_starts):
                next_start = task_starts[idx + 1][0]
            break

    if target_start is None:
        print(f"Task {task_num} not found in {plan_path}", file=sys.stderr)
        sys.exit(1)

    return "".join(lines[target_start:next_start])


def main():
    if len(sys.argv) != 3:
        print("usage: task_brief.py PLAN_FILE TASK_NUMBER", file=sys.stderr)
        sys.exit(2)

    plan_path = Path(sys.argv[1]).resolve()
    task_num = int(sys.argv[2])

    if not plan_path.exists():
        print(f"no such plan file: {plan_path}", file=sys.stderr)
        sys.exit(2)

    content = extract_task(plan_path, task_num)

    # Write to .superpowers/sdd/task-N-brief.md
    root = Path(__file__).resolve().parent.parent
    # Actually, this script is IN .superpowers/sdd/, so parent.parent is repo root
    # But let's use git to find root
    import subprocess
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True, text=True, cwd=str(plan_path.parent)
    )
    if result.returncode == 0:
        root = Path(result.stdout.strip())
    out_dir = root / ".superpowers" / "sdd"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"task-{task_num}-brief.md"
    out_file.write_text(content, encoding="utf-8")
    print(str(out_file))


if __name__ == "__main__":
    main()
