from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional


def _ensure_mapping(config: Any, attr_name: str) -> Dict[str, Any]:
    value = getattr(config, attr_name, None)
    if value is None:
        value = {}
        setattr(config, attr_name, value)
    return value


def resolve_repo_local_path(path_value: Optional[str], repo_root: Path | str) -> Optional[str]:
    if not path_value:
        return path_value

    repo_root = Path(repo_root).resolve()
    candidate = Path(str(path_value))
    if candidate.is_absolute():
        repo_local = repo_root / candidate.name
        if repo_local.exists():
            return str(repo_local)
        return str(candidate)

    repo_local = (repo_root / candidate).resolve()
    if repo_local.exists():
        return str(repo_local)
    return str(candidate)


def resolve_relative_path(path_value: Optional[str], base_dir: Path | str) -> Optional[str]:
    if not path_value:
        return path_value

    base_dir = Path(base_dir).resolve()
    candidate = Path(str(path_value)).expanduser()
    if candidate.is_absolute():
        return str(candidate)
    return str((base_dir / candidate).resolve())


def configure_runtime_paths(
    config: Any,
    *,
    script_path: str,
    config_path: str,
    output_root: Optional[str] = None,
    run_name: Optional[str] = None,
) -> Dict[str, str]:
    repo_root = Path(script_path).resolve().parent.parent

    data_cfg = _ensure_mapping(config, 'data')
    checkpoint_cfg = _ensure_mapping(config, 'checkpoint')
    logging_cfg = _ensure_mapping(config, 'logging')
    segmentation_cfg = _ensure_mapping(config, 'segmentation')

    for key in ('train_json', 'test_json'):
        if key in data_cfg:
            data_cfg[key] = resolve_repo_local_path(data_cfg.get(key), repo_root)

    if output_root:
        output_root_path = Path(output_root).expanduser()
        if not output_root_path.is_absolute():
            output_root_path = repo_root / output_root_path
    else:
        output_root_path = repo_root.parent / 'mamba_artifacts'
    output_root_path = output_root_path.resolve()

    active_run_name = run_name or Path(config_path).stem
    run_root = output_root_path / active_run_name
    checkpoint_dir = run_root / 'checkpoints'
    log_dir = run_root / 'logs'
    results_dir = run_root / 'results'

    checkpoint_cfg['save_dir'] = str(checkpoint_dir)
    logging_cfg['log_dir'] = str(log_dir)
    logging_cfg['validation_history_path'] = str(log_dir / 'validation_history.txt')
    logging_cfg['run_name'] = active_run_name
    logging_cfg['output_root'] = str(output_root_path)
    segmentation_cfg['output_dir'] = str(results_dir)

    resume_value = checkpoint_cfg.get('resume')
    if resume_value:
        checkpoint_cfg['resume'] = resolve_relative_path(str(resume_value), checkpoint_dir)

    return {
        'repo_root': str(repo_root),
        'output_root': str(output_root_path),
        'run_name': active_run_name,
        'run_root': str(run_root),
        'checkpoint_dir': str(checkpoint_dir),
        'log_dir': str(log_dir),
        'results_dir': str(results_dir),
    }


def attach_log_file_handler(log_dir: Path | str, filename: str = 'train.log') -> str:
    log_dir = Path(log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = (log_dir / filename).resolve()

    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename).resolve() == log_path:
            return str(log_path)

    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(root_logger.level or logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    root_logger.addHandler(file_handler)
    return str(log_path)
