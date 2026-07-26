#!/usr/bin/env bash
# T3 trace batch: 25 varied tasks through trace proxy (:9101) -> gateway -> 201 model.
set -u
S="$(cd "$(dirname "$0")" && pwd)"
PROJ="$S/trace-project"
XDG="$S/vanilla-xdg"
LOG="$S/trace_batch.log"
cd "$PROJ"

TASKS=(
  "List the markdown files in this project."
  "Read README.md and summarize it in one sentence."
  "Read src/calc.py and explain what div does."
  "Add a subtract function to src/calc.py."
  "Find all TODO comments in this project."
  "Create a file docs/CHANGELOG.md with one entry dated today."
  "Read docs/TODO.md and implement the first TODO as a test file tests/test_calc.py."
  "Rename the function list_files in src/util.py to list_dir and update nothing else."
  "Count how many Python files exist in this project."
  "Read src/util.py and add a docstring to list_files."
  "Create a .gitignore appropriate for a Python project."
  "Fix the division-by-zero bug in src/calc.py."
  "Write a one-paragraph description of this project into docs/ABOUT.md."
  "List every file in the project and identify the largest one."
  "Add type hints to all functions in src/calc.py."
  "Search the project for the word 'calculator'."
  "Create a Makefile with a test target that runs pytest."
  "Read all files under docs/ and merge them into docs/INDEX.md."
  "Add an integer square-root function to src/util.py."
  "Check whether README.md mentions the util module and quote the line."
  "Create tests/test_util.py covering list_files."
  "Summarize the whole project structure as a bullet list."
  "Add error handling to list_files for a missing path."
  "Write a short usage example for calc.py into README.md."
  "Find any function without a docstring and list them."
)

i=0
for t in "${TASKS[@]}"; do
  i=$((i+1))
  echo "=== task $i/$(printf '%s' "${#TASKS[@]}"): $t ===" >> "$LOG"
  start=$(date +%s)
  XDG_CONFIG_HOME="$XDG" timeout 240 opencode run -m vx/qwen2.5:7b-instruct "$t" >> "$LOG" 2>&1
  rc=$?
  echo "=== task $i rc=$rc elapsed=$(( $(date +%s) - start ))s ===" >> "$LOG"
done
echo "BATCH_DONE tasks=$i" | tee -a "$LOG"
