#!/bin/bash
set -euo pipefail

GIT_REMOTE_URL="${GIT_REMOTE_URL:?Please set GIT_REMOTE_URL}"
RUNNER_ROOT="${RUNNER_ROOT:-/mnt/nas/share/home/liuke/prjs/uter/model_with_mamba/mamba_runner}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/mnt/nas/share/home/liuke/prjs/uter/model_with_mamba/mamba_artifacts}"
BRANCH="${BRANCH:-main}"

mkdir -p "${ARTIFACT_ROOT}"

git_runner() {
  git -c safe.directory="${RUNNER_ROOT}" -C "${RUNNER_ROOT}" "$@"
}

if [[ -d "${RUNNER_ROOT}/.git" ]]; then
  echo "[runner] Existing runner detected at ${RUNNER_ROOT}"
  git_runner fetch origin
else
  echo "[runner] Cloning ${GIT_REMOTE_URL} into ${RUNNER_ROOT}"
  git clone "${GIT_REMOTE_URL}" "${RUNNER_ROOT}"
fi

git_runner checkout "${BRANCH}"
git_runner pull --ff-only origin "${BRANCH}"

echo "[runner] Runner root: ${RUNNER_ROOT}"
echo "[runner] Artifact root: ${ARTIFACT_ROOT}"
echo "[runner] Active branch: ${BRANCH}"
