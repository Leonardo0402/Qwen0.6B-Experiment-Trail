# Task 11 Report — Phase G: corrupted_recovered augmentation generator

**Status:** DONE

**Commit SHA:** `75a6f568b63a4d9cbb327c2179e930e54c6bf4de`

**BASE:** `69257e3` (T10 commit) → HEAD: `75a6f56`

## 1. What was implemented
Created `scripts/augment_corrupted_recovered.py` — a Phase G augmentation generator that, for each scripted trajectory, applies each of the 5 `CorruptionType` values at step indices `[1, 2, 3]` plus any patchable steps, runs the evaluator with `CorruptedActionProvider`, and records the original (uncorrupted) action sequence when the corrupted run still reaches success. Output goes to `data/p4-agent/trajectories-v1/corrupted-recovered.jsonl` at runtime (not committed). The script defines `_ListActionProvider` inline (same pattern as T10) and imports only `Action` from `src.agent_actions` (no SentinelAction).

## 2. Syntax verification
Command: `py -3.11 -c "import ast; ast.parse(open('scripts/augment_corrupted_recovered.py').read()); print('OK')"`

Output:
```
OK
```

## 3. Diffstat
Command: `git diff --stat 69257e3 HEAD`

```
 scripts/augment_corrupted_recovered.py | 121 +++++++++++++++++++++++++++++++++
 1 file changed, 121 insertions(+)
```

## 4. Self-review findings
- **Surgical changes:** Only one file created — `scripts/augment_corrupted_recovered.py` (121 lines). No `src/` files touched; `src/agent_trajectory.py` left frozen.
- **Staging discipline:** `git diff --cached --name-only` confirmed only `scripts/augment_corrupted_recovered.py` was staged. The modified `.superpowers/sdd/` files (briefs, reports, progress) and untracked review-package files were deliberately left unstaged.
- **Verbatim code:** Script content matches the brief byte-for-byte (module docstring, `_ROOT`/`sys.path`/`P4_ALLOW_NETWORK` setup, imports, `_ListActionProvider`, constants, `main()` body, and `__main__` guard). No additions, no extra error handling, no refactors.
- **Imports:** Only `Action` imported from `src.agent_actions` (no `SentinelAction`), as specified.
- **Commit message:** Matches brief exactly: `feat(p4-1): Phase G — corrupted_recovered augmentation generator`.
- **Git line-ending warning** (`LF will be replaced by CRLF`) is benign on Windows; does not affect file content in the repo.

## 5. Concerns
None. The script was not executed (per brief — only `ast.parse` verification in this environment); the actual run to produce `corrupted-recovered.jsonl` happens before PR merge.

## 6. One-line test summary
`ast.parse` on `scripts/augment_corrupted_recovered.py` returned `OK`; file staged alone and committed as `75a6f56` on top of T10.
