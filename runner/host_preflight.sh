#!/bin/bash
set -euo pipefail

RUNNER_ROOT="${RUNNER_ROOT:-/mnt/nas/share/home/liuke/prjs/uter/model_with_mamba/mamba_runner}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/mnt/nas/share/home/liuke/prjs/uter/model_with_mamba/mamba_artifacts}"
TRAIN_JSON_REL="${TRAIN_JSON_REL:-all_entries_train.json}"
TEST_JSON_REL="${TEST_JSON_REL:-all_entries_test.json}"
CUDA_FREE_THRESHOLD_MIB="${CUDA_FREE_THRESHOLD_MIB:-1024}"
PYTHON_BIN="${PYTHON_BIN:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --runner-root)
      RUNNER_ROOT="$2"
      shift 2
      ;;
    --artifact-root)
      ARTIFACT_ROOT="$2"
      shift 2
      ;;
    --train-json)
      TRAIN_JSON_REL="$2"
      shift 2
      ;;
    --test-json)
      TEST_JSON_REL="$2"
      shift 2
      ;;
    --cuda-free-threshold-mib)
      CUDA_FREE_THRESHOLD_MIB="$2"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

detect_python_bin() {
  local candidate=""
  local candidates=()
  if [[ -n "${PYTHON_BIN}" ]]; then
    candidates+=("${PYTHON_BIN}")
  fi
  candidates+=(
    "/home/lk/.pyenv/versions/miniconda3-latest/envs/uter_mamba/bin/python"
    "/home/lk/.pyenv/versions/uter_mamba/bin/python"
    "python3"
    "python"
  )
  for candidate in "${candidates[@]}"; do
    if [[ -x "${candidate}" ]]; then
      echo "${candidate}"
      return 0
    fi
    if command -v "${candidate}" >/dev/null 2>&1; then
      command -v "${candidate}"
      return 0
    fi
  done
  return 1
}

for required_cmd in git nvidia-smi hostname; do
  if ! command -v "${required_cmd}" >/dev/null 2>&1; then
    echo "Missing required command: ${required_cmd}" >&2
    exit 1
  fi
done

PYTHON_BIN="$(detect_python_bin)"
"${PYTHON_BIN}" -m torch.distributed.run --help >/dev/null 2>&1

if [[ ! -d "${RUNNER_ROOT}/.git" ]]; then
  echo "Runner root is not a git checkout: ${RUNNER_ROOT}" >&2
  exit 1
fi

mkdir -p "${ARTIFACT_ROOT}"
PROBE_FILE="${ARTIFACT_ROOT}/.preflight_probe_$$"
touch "${PROBE_FILE}"
rm -f "${PROBE_FILE}"

TRAIN_JSON="${RUNNER_ROOT}/${TRAIN_JSON_REL}"
TEST_JSON="${RUNNER_ROOT}/${TEST_JSON_REL}"
if [[ ! -f "${TRAIN_JSON}" || ! -f "${TEST_JSON}" ]]; then
  echo "Train/test JSON missing under runner root" >&2
  exit 1
fi

export RUNNER_ROOT ARTIFACT_ROOT TRAIN_JSON TEST_JSON CUDA_FREE_THRESHOLD_MIB PYTHON_BIN
"${PYTHON_BIN}" - <<'PY'
import json
import os
import subprocess
import sys
from pathlib import Path


def first_image_path(train_json: Path) -> str | None:
    with train_json.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list) and payload:
        first = payload[0]
        if isinstance(first, dict):
            return first.get("image")
    return None


runner_root = Path(os.environ["RUNNER_ROOT"])
artifact_root = Path(os.environ["ARTIFACT_ROOT"])
train_json = Path(os.environ["TRAIN_JSON"])
test_json = Path(os.environ["TEST_JSON"])
python_bin = os.environ["PYTHON_BIN"]
threshold = int(os.environ["CUDA_FREE_THRESHOLD_MIB"])

gpu_query = subprocess.run(
    [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ],
    check=True,
    capture_output=True,
    text=True,
)

gpu_rows = []
free_gpu_ids = []
for line in gpu_query.stdout.strip().splitlines():
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != 3:
        continue
    index, used, total = int(parts[0]), int(parts[1]), int(parts[2])
    row = {
        "index": index,
        "memory_used_mib": used,
        "memory_total_mib": total,
        "free_under_threshold": used < threshold,
    }
    gpu_rows.append(row)
    if row["free_under_threshold"]:
        free_gpu_ids.append(index)

sample_image = first_image_path(train_json)
sample_image_exists = bool(sample_image and Path(sample_image).exists())

payload = {
    "ok": True,
    "host": subprocess.run(["hostname"], check=True, capture_output=True, text=True).stdout.strip(),
    "runner_root": str(runner_root),
    "artifact_root": str(artifact_root),
    "python_bin": python_bin,
    "train_json": str(train_json),
    "test_json": str(test_json),
    "sample_image": sample_image,
    "sample_image_exists": sample_image_exists,
    "gpu_rows": gpu_rows,
    "free_gpu_ids": free_gpu_ids,
    "free_gpu_count": len(free_gpu_ids),
}

if not sample_image_exists:
    payload["ok"] = False
    payload["error"] = f"Sample image path is not readable: {sample_image}"
    print(json.dumps(payload))
    sys.exit(1)

print(json.dumps(payload))
PY
