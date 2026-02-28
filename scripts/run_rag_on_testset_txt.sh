#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

TEST_FILE="${1:-data/test_set_day_2.txt}"
OUT_FILE="${2:-data/answers/system_output_1.json}"
if [[ $# -gt 2 ]]; then
  EXTRA_ARGS=("${@:3}")
else
  EXTRA_ARGS=()
fi

if [[ ! -f "${TEST_FILE}" ]]; then
  echo "Error: test file not found: ${TEST_FILE}" >&2
  exit 1
fi

TMP_DIR="$(mktemp -d /tmp/rag_testset_txt.XXXXXX)"
TMP_QUERIES="${TMP_DIR}/queries.json"
trap 'rm -rf "${TMP_DIR}"' EXIT

python3 - "${TEST_FILE}" "${TMP_QUERIES}" <<'PY'
import json
import sys
from pathlib import Path

in_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])

questions = []
for raw in in_path.read_text(encoding="utf-8-sig").splitlines():
    q = raw.strip()
    if q:
        questions.append(q)

if not questions:
    raise SystemExit(f"No non-empty questions found in {in_path}")

payload = [{"id": str(i), "question": q} for i, q in enumerate(questions, start=1)]
out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[run_rag_on_testset_txt] prepared {len(payload)} queries -> {out_path}")
PY

uv run src/process/query_ppl.py \
  --queries "${TMP_QUERIES}" \
  --out "${OUT_FILE}" \
  "${EXTRA_ARGS[@]}"

echo "[run_rag_on_testset_txt] wrote answers to ${OUT_FILE}"
