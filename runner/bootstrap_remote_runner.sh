#!/bin/bash
set -euo pipefail

GIT_REMOTE_URL="${GIT_REMOTE_URL:?Please set GIT_REMOTE_URL}"
RUNNER_ROOT="${RUNNER_ROOT:-/mnt/nas/share/home/liuke/prjs/uter/model_with_mamba/mamba_runner}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/mnt/nas/share/home/liuke/prjs/uter/model_with_mamba/mamba_artifacts}"
BRANCH="${BRANCH:-main}"

mkdir -p "${ARTIFACT_ROOT}"

if [[ -d "${RUNNER_ROOT}/.git" ]]; then
  echo "[runner] Existing runner detected at ${RUNNER_ROOT}"
  git -C "${RUNNER_ROOT}" fetch origin
else
  echo "[runner] Cloning ${GIT_REMOTE_URL} into ${RUNNER_ROOT}"
  git clone "${GIT_REMOTE_URL}" "${RUNNER_ROOT}"
fi

git -C "${RUNNER_ROOT}" checkout "${BRANCH}"
git -C "${RUNNER_ROOT}" pull --ff-only origin "${BRANCH}"

echo "[runner] Runner root: ${RUNNER_ROOT}"
echo "[runner] Artifact root: ${ARTIFACT_ROOT}"
echo "[runner] Active branch: ${BRANCH}"