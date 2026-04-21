#!/bin/bash
set -euo pipefail

RUNNER_ROOT="${RUNNER_ROOT:-/mnt/nas/share/home/liuke/prjs/uter/model_with_mamba/mamba_runner}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/mnt/nas/share/home/liuke/prjs/uter/model_with_mamba/mamba_artifacts}"
PYTHON_BIN="${PYTHON_BIN:-}"
EXPERIMENT=""
PHASE=""
RUN_NAME=""
GPU_IDS=""
declare -a OVERRIDES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --experiment)
      EXPERIMENT="$2"
      shift 2
      ;;
    --phase)
      PHASE="$2"
      shift 2
      ;;
    --run-name)
      RUN_NAME="$2"
      shift 2
      ;;
    --gpu-ids)
      GPU_IDS="$2"
      shift 2
      ;;
    --runner-root)
      RUNNER_ROOT="$2"
      shift 2
      ;;
    --artifact-root)
      ARTIFACT_ROOT="$2"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --set)
      OVERRIDES+=("$2")
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${EXPERIMENT}" || -z "${PHASE}" || -z "${RUN_NAME}" || -z "${GPU_IDS}" ]]; then
  echo "experiment, phase, run-name, and gpu-ids are required" >&2
  exit 2
fi

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

case "${EXPERIMENT}" in
  fullsup_seg)
    BASE_CONFIG="${RUNNER_ROOT}/configs/fullsup_seg.yaml"
    TRAIN_SCRIPT="scripts/train_fullsup_seg_ddp.py"
    NEED_NO_RESUME=0
    ;;
  weak_cbfirst_raw)
    BASE_CONFIG="${RUNNER_ROOT}/configs/v20_retrain.yaml"
    TRAIN_SCRIPT="scripts/train_volumetric_ddp.py"
    NEED_NO_RESUME=1
    ;;
  weak_mambafirst_raw)
    BASE_CONFIG="${RUNNER_ROOT}/configs/v20_mamba_first_weak.yaml"
    TRAIN_SCRIPT="scripts/train_volumetric_ddp.py"
    NEED_NO_RESUME=1
    ;;
  *)
    echo "Unknown experiment: ${EXPERIMENT}" >&2
    exit 2
    ;;
esac

PYTHON_BIN="$(detect_python_bin)"
mkdir -p "${ARTIFACT_ROOT}/_resolved_configs"

RESOLVED_CONFIG="${ARTIFACT_ROOT}/_resolved_configs/${RUN_NAME}.yaml"
RENDER_ARGS=(
  "${PYTHON_BIN}" "${RUNNER_ROOT}/runner/render_resolved_config.py"
  "--base-config" "${BASE_CONFIG}"
  "--output" "${RESOLVED_CONFIG}"
)
for override in "${OVERRIDES[@]}"; do
  RENDER_ARGS+=("--set" "${override}")
done
"${RENDER_ARGS[@]}" >/dev/null

export GPU_IDS
GPU_COUNT="$("${PYTHON_BIN}" - <<'PY'
import os

gpu_ids = [item.strip() for item in os.environ["GPU_IDS"].split(",") if item.strip()]
print(len(gpu_ids))
PY
)"

MASTER_PORT="$("${PYTHON_BIN}" - <<'PY'
import socket

sock = socket.socket()
sock.bind(("", 0))
print(sock.getsockname()[1])
sock.close()
PY
)"

RUN_ROOT="${ARTIFACT_ROOT}/${RUN_NAME}"
if [[ -d "${RUN_ROOT}" ]] && [[ -n "$(ls -A "${RUN_ROOT}" 2>/dev/null)" ]]; then
  echo "Run root already exists and is not empty: ${RUN_ROOT}" >&2
  exit 1
fi
LOG_DIR="${RUN_ROOT}/logs"
mkdir -p "${LOG_DIR}"
LAUNCHER_LOG="${LOG_DIR}/launcher.log"
STATUS_FILE="${LOG_DIR}/runner_status.json"
META_FILE="${LOG_DIR}/runner_metadata.json"
WRAPPER_PATH="${LOG_DIR}/runner_wrapper.sh"
COMMIT_SHA="$(git -c safe.directory="${RUNNER_ROOT}" -C "${RUNNER_ROOT}" rev-parse HEAD)"
HOSTNAME_VALUE="$(hostname)"
ENV_PREFIX="$(cd "$(dirname "${PYTHON_BIN}")/.." && pwd)"
TORCH_LIB_PATH="${ENV_PREFIX}/lib/python3.11/site-packages/torch/lib"

export RUNNER_ROOT ARTIFACT_ROOT RUN_NAME RESOLVED_CONFIG TRAIN_SCRIPT GPU_IDS GPU_COUNT MASTER_PORT
export LAUNCHER_LOG STATUS_FILE META_FILE WRAPPER_PATH COMMIT_SHA HOSTNAME_VALUE EXPERIMENT PHASE
export PYTHON_BIN TORCH_LIB_PATH NEED_NO_RESUME

cat > "${WRAPPER_PATH}" <<'EOF'
#!/bin/bash
set -euo pipefail

cd "${RUNNER_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
if [[ -d "${TORCH_LIB_PATH}" ]]; then
  export LD_LIBRARY_PATH="${TORCH_LIB_PATH}:${LD_LIBRARY_PATH:-}"
fi
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

"${PYTHON_BIN}" - <<'PY'
import json
import os
import time
from pathlib import Path

payload = {
    "state": "running",
    "run_name": os.environ["RUN_NAME"],
    "experiment": os.environ["EXPERIMENT"],
    "phase": os.environ["PHASE"],
    "host": os.environ["HOSTNAME_VALUE"],
    "pid": os.getpid(),
    "gpu_ids": [item.strip() for item in os.environ["GPU_IDS"].split(",") if item.strip()],
    "gpu_count": int(os.environ["GPU_COUNT"]),
    "commit_sha": os.environ["COMMIT_SHA"],
    "config_path": os.environ["RESOLVED_CONFIG"],
    "launcher_log": os.environ["LAUNCHER_LOG"],
    "started_at_epoch": time.time(),
}
Path(os.environ["STATUS_FILE"]).write_text(json.dumps(payload), encoding="utf-8")
PY

TRAIN_ARGS=(
  "${TRAIN_SCRIPT}"
  "--config" "${RESOLVED_CONFIG}"
  "--output_root" "${ARTIFACT_ROOT}"
  "--run_name" "${RUN_NAME}"
)
if [[ "${NEED_NO_RESUME}" == "1" ]]; then
  TRAIN_ARGS+=("--no_resume")
fi

set +e
"${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node="${GPU_COUNT}" \
  --master_port="${MASTER_PORT}" \
  "${TRAIN_ARGS[@]}" >> "${LAUNCHER_LOG}" 2>&1
EXIT_CODE=$?
set -e

OOM_DETECTED=0
if grep -qiE 'out of memory|cuda error: out of memory' "${LAUNCHER_LOG}"; then
  OOM_DETECTED=1
fi

export EXIT_CODE OOM_DETECTED
"${PYTHON_BIN}" - <<'PY'
import json
import os
import time
from pathlib import Path

payload = {
    "state": "completed" if int(os.environ["EXIT_CODE"]) == 0 else "failed",
    "run_name": os.environ["RUN_NAME"],
    "experiment": os.environ["EXPERIMENT"],
    "phase": os.environ["PHASE"],
    "host": os.environ["HOSTNAME_VALUE"],
    "pid": os.getpid(),
    "gpu_ids": [item.strip() for item in os.environ["GPU_IDS"].split(",") if item.strip()],
    "gpu_count": int(os.environ["GPU_COUNT"]),
    "commit_sha": os.environ["COMMIT_SHA"],
    "config_path": os.environ["RESOLVED_CONFIG"],
    "launcher_log": os.environ["LAUNCHER_LOG"],
    "completed_at_epoch": time.time(),
    "exit_code": int(os.environ["EXIT_CODE"]),
    "oom_detected": bool(int(os.environ["OOM_DETECTED"])),
}
Path(os.environ["STATUS_FILE"]).write_text(json.dumps(payload), encoding="utf-8")
PY

exit "${EXIT_CODE}"
EOF

chmod +x "${WRAPPER_PATH}"

nohup "${WRAPPER_PATH}" >/dev/null 2>&1 &
PID="$!"

export PID
"${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

payload = {
    "run_name": os.environ["RUN_NAME"],
    "experiment": os.environ["EXPERIMENT"],
    "phase": os.environ["PHASE"],
    "host": os.environ["HOSTNAME_VALUE"],
    "pid": int(os.environ["PID"]),
    "gpu_ids": [item.strip() for item in os.environ["GPU_IDS"].split(",") if item.strip()],
    "gpu_count": int(os.environ["GPU_COUNT"]),
    "commit_sha": os.environ["COMMIT_SHA"],
    "config_path": os.environ["RESOLVED_CONFIG"],
    "launcher_log": os.environ["LAUNCHER_LOG"],
    "status_file": os.environ["STATUS_FILE"],
    "wrapper_path": os.environ["WRAPPER_PATH"],
    "runner_root": os.environ["RUNNER_ROOT"],
    "artifact_root": os.environ["ARTIFACT_ROOT"],
}
Path(os.environ["META_FILE"]).write_text(json.dumps(payload), encoding="utf-8")
print(json.dumps(payload))
PY
