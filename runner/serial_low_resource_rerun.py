from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


HOST_PRIORITY = ["8-228", "8-232", "8-238", "8-240"]


@dataclass
class ExperimentSpec:
    experiment: str
    phase: str
    run_name: str
    preferred_gpus: int
    fallback_gpus: int
    overrides: list[str]
    oom_retry_max_patches: int | None = None
    current_max_patches: int | None = None


def run_subprocess(command: list[str], *, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def run_ssh(host: str, script: str, *, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    return run_subprocess(["ssh", host, f"bash -lc {json.dumps(script)}"], timeout=timeout)


def local_git_output(repo_root: Path, *args: str) -> str:
    command = ["git", "-c", f"safe.directory={repo_root.as_posix()}", *args]
    result = run_subprocess(command)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def bootstrap_runner(
    host: str,
    *,
    remote_url: str,
    branch: str,
    runner_root: str,
    artifact_root: str,
) -> None:
    script = f"""
set -euo pipefail
RUNNER_ROOT={json.dumps(runner_root)}
ARTIFACT_ROOT={json.dumps(artifact_root)}
REMOTE_URL={json.dumps(remote_url)}
BRANCH={json.dumps(branch)}
mkdir -p "$(dirname "$RUNNER_ROOT")" "$ARTIFACT_ROOT"
if [[ ! -d "$RUNNER_ROOT/.git" ]]; then
  git clone "$REMOTE_URL" "$RUNNER_ROOT"
fi
export GIT_REMOTE_URL="$REMOTE_URL"
export RUNNER_ROOT="$RUNNER_ROOT"
export ARTIFACT_ROOT="$ARTIFACT_ROOT"
export BRANCH="$BRANCH"
bash "$RUNNER_ROOT/runner/bootstrap_remote_runner.sh"
git -c safe.directory="$RUNNER_ROOT" -C "$RUNNER_ROOT" rev-parse HEAD
"""
    result = run_ssh(host, script, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"{host} bootstrap failed: {result.stderr.strip() or result.stdout.strip()}")


def preflight_host(host: str, *, runner_root: str, artifact_root: str) -> dict[str, Any]:
    script = (
        f"set -euo pipefail; "
        f"bash {json.dumps(f'{runner_root}/runner/host_preflight.sh')} "
        f"--runner-root {json.dumps(runner_root)} "
        f"--artifact-root {json.dumps(artifact_root)}"
    )
    result = run_ssh(host, script, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"{host} preflight failed: {result.stderr.strip() or result.stdout.strip()}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def pick_host(host_infos: dict[str, dict[str, Any]], preferred_gpus: int, fallback_gpus: int) -> tuple[str, list[int]]:
    candidates = []
    for host, info in host_infos.items():
        gpu_ids = info.get("free_gpu_ids", [])
        if len(gpu_ids) >= preferred_gpus:
            candidates.append((host, gpu_ids, 0))
        elif len(gpu_ids) >= fallback_gpus:
            candidates.append((host, gpu_ids, 1))
    if not candidates:
        raise RuntimeError("No host currently satisfies the GPU requirement.")
    candidates.sort(
        key=lambda item: (
            item[2],
            -len(item[1]),
            HOST_PRIORITY.index(item[0]) if item[0] in HOST_PRIORITY else len(HOST_PRIORITY),
        )
    )
    host, gpu_ids, _ = candidates[0]
    requested = preferred_gpus if len(gpu_ids) >= preferred_gpus else fallback_gpus
    return host, gpu_ids[:requested]


def launch_experiment(
    host: str,
    *,
    runner_root: str,
    artifact_root: str,
    spec: ExperimentSpec,
    gpu_ids: list[int],
) -> dict[str, Any]:
    override_args = " ".join(f"--set {json.dumps(item)}" for item in spec.overrides)
    launch_cmd = (
        f"set -euo pipefail; "
        f"bash {json.dumps(f'{runner_root}/runner/launch_remote_experiment.sh')} "
        f"--runner-root {json.dumps(runner_root)} "
        f"--artifact-root {json.dumps(artifact_root)} "
        f"--experiment {json.dumps(spec.experiment)} "
        f"--phase {json.dumps(spec.phase)} "
        f"--run-name {json.dumps(spec.run_name)} "
        f"--gpu-ids {json.dumps(','.join(str(item) for item in gpu_ids))} "
        f"{override_args}"
    )
    result = run_ssh(host, launch_cmd, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"{host} launch failed: {result.stderr.strip() or result.stdout.strip()}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def wait_for_completion(host: str, status_file: str, poll_seconds: int) -> dict[str, Any]:
    while True:
        result = run_ssh(host, f"cat {json.dumps(status_file)}", timeout=120)
        if result.returncode == 0 and result.stdout.strip():
            payload = json.loads(result.stdout.strip())
            if payload.get("state") in {"completed", "failed"}:
                return payload
        time.sleep(poll_seconds)


def latest_remote_head(host: str, runner_root: str) -> str:
    result = run_ssh(
        host,
        f"git -c safe.directory={json.dumps(runner_root)} -C {json.dumps(runner_root)} rev-parse HEAD",
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"{host} head check failed: {result.stderr.strip() or result.stdout.strip()}")
    return result.stdout.strip().splitlines()[-1]


def refresh_hosts(hosts: list[str], *, remote_url: str, branch: str, runner_root: str, artifact_root: str) -> dict[str, dict[str, Any]]:
    host_infos: dict[str, dict[str, Any]] = {}
    for host in hosts:
        probe = run_subprocess(["ssh", host, "hostname"], timeout=30)
        if probe.returncode != 0:
            print(f"[skip] {host}: unreachable ({probe.stderr.strip() or probe.stdout.strip()})")
            continue
        try:
            bootstrap_runner(host, remote_url=remote_url, branch=branch, runner_root=runner_root, artifact_root=artifact_root)
            info = preflight_host(host, runner_root=runner_root, artifact_root=artifact_root)
            host_infos[host] = info
            print(f"[host] {host}: free_gpus={info.get('free_gpu_ids', [])} python={info.get('python_bin')}")
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {host}: {exc}")
    return host_infos


def run_serial_pipeline(
    *,
    repo_root: Path,
    remote_url: str,
    branch: str,
    runner_root: str,
    artifact_root: str,
    poll_seconds: int,
) -> None:
    local_head = local_git_output(repo_root, "rev-parse", "HEAD")
    host_infos = refresh_hosts(HOST_PRIORITY, remote_url=remote_url, branch=branch, runner_root=runner_root, artifact_root=artifact_root)
    if not host_infos:
        raise RuntimeError("No remote host passed bootstrap + preflight.")

    smoke_specs = [
        ExperimentSpec(
            experiment="fullsup_seg",
            phase="smoke",
            run_name="fullsup_seg_smoke",
            preferred_gpus=1,
            fallback_gpus=1,
            overrides=[
                "data.batch_size=1",
                "training.epochs=2",
                "training.val_freq=1",
                "training.val_start_epoch=1",
            ],
            oom_retry_max_patches=32,
        ),
        ExperimentSpec(
            experiment="weak_cbfirst_raw",
            phase="smoke",
            run_name="weak_cbfirst_raw_smoke",
            preferred_gpus=2,
            fallback_gpus=1,
            overrides=[
                "data.batch_size=1",
                "data.max_patches=16",
                "model.feature_extractor.mini_batch_size=4",
                "training.epochs=4",
                "training.val_freq=1",
                "training.seg_val_freq=1",
                "training.val_start_epoch=1",
                "training.pretrain_healthy_vqvae_epochs=1",
                "training.phase1_epochs=1",
                "checkpoint.resume=null",
            ],
        ),
        ExperimentSpec(
            experiment="weak_mambafirst_raw",
            phase="smoke",
            run_name="weak_mambafirst_raw_smoke",
            preferred_gpus=2,
            fallback_gpus=1,
            overrides=[
                "data.batch_size=1",
                "data.max_patches=16",
                "model.feature_extractor.mini_batch_size=4",
                "training.epochs=4",
                "training.val_freq=1",
                "training.seg_val_freq=1",
                "training.val_start_epoch=1",
                "training.pretrain_healthy_vqvae_epochs=1",
                "training.phase1_epochs=1",
                "checkpoint.resume=null",
            ],
        ),
    ]

    full_specs = [
        ExperimentSpec(
            experiment="fullsup_seg",
            phase="full",
            run_name="fullsup_seg_fixrerun_20260421",
            preferred_gpus=2,
            fallback_gpus=1,
            overrides=["data.batch_size=1"],
            oom_retry_max_patches=32,
            current_max_patches=64,
        ),
        ExperimentSpec(
            experiment="weak_cbfirst_raw",
            phase="full",
            run_name="weak_cbfirst_raw_fixrerun_20260421",
            preferred_gpus=2,
            fallback_gpus=1,
            overrides=[
                "data.batch_size=1",
                "data.max_patches=32",
                "model.feature_extractor.mini_batch_size=4",
                "checkpoint.resume=null",
            ],
            oom_retry_max_patches=16,
            current_max_patches=32,
        ),
        ExperimentSpec(
            experiment="weak_mambafirst_raw",
            phase="full",
            run_name="weak_mambafirst_raw_fixrerun_20260421",
            preferred_gpus=2,
            fallback_gpus=1,
            overrides=[
                "data.batch_size=1",
                "data.max_patches=32",
                "model.feature_extractor.mini_batch_size=4",
                "checkpoint.resume=null",
            ],
            oom_retry_max_patches=16,
            current_max_patches=32,
        ),
    ]

    full_by_experiment = {spec.experiment: spec for spec in full_specs}

    for spec in smoke_specs:
        host_infos = refresh_hosts(HOST_PRIORITY, remote_url=remote_url, branch=branch, runner_root=runner_root, artifact_root=artifact_root)
        host, gpu_ids = pick_host(host_infos, spec.preferred_gpus, spec.fallback_gpus)
        remote_head = latest_remote_head(host, runner_root)
        if remote_head != local_head:
            raise RuntimeError(f"{host} runner head {remote_head} does not match local head {local_head}")

        metadata = launch_experiment(host, runner_root=runner_root, artifact_root=artifact_root, spec=spec, gpu_ids=gpu_ids)
        print(f"[launch] {spec.run_name} on {host} gpus={gpu_ids} log={metadata['launcher_log']}")
        status = wait_for_completion(host, metadata["status_file"], poll_seconds)
        print(f"[done] {spec.run_name} state={status['state']} exit_code={status.get('exit_code')}")

        if status.get("state") == "failed" and status.get("oom_detected") and spec.oom_retry_max_patches:
            retry_spec = ExperimentSpec(
                experiment=spec.experiment,
                phase=spec.phase,
                run_name=f"{spec.run_name}_oomretry",
                preferred_gpus=spec.preferred_gpus,
                fallback_gpus=spec.fallback_gpus,
                overrides=[item for item in spec.overrides if not item.startswith("data.max_patches=")]
                + [f"data.max_patches={spec.oom_retry_max_patches}"],
            )
            metadata = launch_experiment(host, runner_root=runner_root, artifact_root=artifact_root, spec=retry_spec, gpu_ids=gpu_ids)
            print(f"[retry] {retry_spec.run_name} on {host} gpus={gpu_ids}")
            status = wait_for_completion(host, metadata["status_file"], poll_seconds)
            print(f"[done] {retry_spec.run_name} state={status['state']} exit_code={status.get('exit_code')}")
            if status.get("state") != "completed":
                raise RuntimeError(f"{retry_spec.run_name} failed even after OOM retry.")
            if spec.experiment == "fullsup_seg":
                full_by_experiment["fullsup_seg"].current_max_patches = spec.oom_retry_max_patches
        elif status.get("state") != "completed":
            raise RuntimeError(f"{spec.run_name} failed; see {status.get('launcher_log')}")

    for spec in full_specs:
        if spec.experiment == "fullsup_seg" and spec.current_max_patches is not None and spec.current_max_patches != 64:
            spec.overrides = [item for item in spec.overrides if not item.startswith("data.max_patches=")]
            spec.overrides.append(f"data.max_patches={spec.current_max_patches}")

        host_infos = refresh_hosts(HOST_PRIORITY, remote_url=remote_url, branch=branch, runner_root=runner_root, artifact_root=artifact_root)
        host, gpu_ids = pick_host(host_infos, spec.preferred_gpus, spec.fallback_gpus)
        remote_head = latest_remote_head(host, runner_root)
        if remote_head != local_head:
            raise RuntimeError(f"{host} runner head {remote_head} does not match local head {local_head}")

        metadata = launch_experiment(host, runner_root=runner_root, artifact_root=artifact_root, spec=spec, gpu_ids=gpu_ids)
        print(f"[launch] {spec.run_name} on {host} gpus={gpu_ids} log={metadata['launcher_log']}")
        status = wait_for_completion(host, metadata["status_file"], poll_seconds)
        print(f"[done] {spec.run_name} state={status['state']} exit_code={status.get('exit_code')}")

        if status.get("state") == "failed" and status.get("oom_detected") and spec.oom_retry_max_patches:
            if spec.current_max_patches == spec.oom_retry_max_patches:
                raise RuntimeError(f"{spec.run_name} hit OOM at the lowest planned max_patches.")
            retry_spec = ExperimentSpec(
                experiment=spec.experiment,
                phase=spec.phase,
                run_name=f"{spec.run_name}_oomretry",
                preferred_gpus=spec.preferred_gpus,
                fallback_gpus=spec.fallback_gpus,
                overrides=[item for item in spec.overrides if not item.startswith("data.max_patches=")]
                + [f"data.max_patches={spec.oom_retry_max_patches}"],
                current_max_patches=spec.oom_retry_max_patches,
            )
            metadata = launch_experiment(host, runner_root=runner_root, artifact_root=artifact_root, spec=retry_spec, gpu_ids=gpu_ids)
            print(f"[retry] {retry_spec.run_name} on {host} gpus={gpu_ids}")
            status = wait_for_completion(host, metadata["status_file"], poll_seconds)
            print(f"[done] {retry_spec.run_name} state={status['state']} exit_code={status.get('exit_code')}")
            if status.get("state") != "completed":
                raise RuntimeError(f"{retry_spec.run_name} failed even after OOM retry.")
        elif status.get("state") != "completed":
            raise RuntimeError(f"{spec.run_name} failed; see {status.get('launcher_log')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Serial low-resource rerun orchestrator for repaired mamba experiments.")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parent.parent), help="Local repository root.")
    parser.add_argument("--branch", default="main", help="Git branch to sync onto remote runners.")
    parser.add_argument(
        "--runner-root",
        default="/mnt/nas/share/home/liuke/prjs/uter/model_with_mamba/mamba_runner",
        help="Remote runner checkout directory.",
    )
    parser.add_argument(
        "--artifact-root",
        default="/mnt/nas/share/home/liuke/prjs/uter/model_with_mamba/mamba_artifacts",
        help="Remote artifact directory.",
    )
    parser.add_argument("--poll-seconds", type=int, default=60, help="Polling interval for remote run status.")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    remote_url = local_git_output(repo_root, "remote", "get-url", "origin")

    run_serial_pipeline(
        repo_root=repo_root,
        remote_url=remote_url,
        branch=args.branch,
        runner_root=args.runner_root,
        artifact_root=args.artifact_root,
        poll_seconds=args.poll_seconds,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise
