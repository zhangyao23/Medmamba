# runner

## 2026-04 低资源重跑补充

当前远程 clean runner 的标准目录仍然是：

```text
/mnt/nas/share/home/liuke/prjs/uter/model_with_mamba/
├─ mamba_final/        # 历史快照，不再作为主执行目录
├─ mamba_runner/       # Git checkout，专门负责训练执行
└─ mamba_artifacts/    # logs / checkpoints / results
```

新增工具：

- `host_preflight.sh`
  - 在远程主机上检查 Git、Python、torch.distributed、`nvidia-smi`、JSON 索引和样本路径是否可用
  - 输出 JSON，包含空闲 GPU 列表和关键环境信息
- `render_resolved_config.py`
  - 根据 base config 生成派生配置副本
  - 支持 `--set dotted.path=value` 形式的覆盖
- `launch_remote_experiment.sh`
  - 在远程 `mamba_runner` 中启动单个实验
  - 自动把派生配置写到 `<ARTIFACT_ROOT>/_resolved_configs/`
  - 把运行元数据和状态写到对应 run 的 `logs/`
- `serial_low_resource_rerun.py`
  - 本地调度脚本
  - 会在 `8-228 / 8-232 / 8-238 / 8-240 / 8-243` 之间挑选可用机器
  - 按“先 smoke 再 full、实验严格串行”的顺序启动 `fullsup_seg`、`v20_retrain`、`v20_mamba_first_weak`
- `start_low_resource_serial_rerun.ps1`
  - Windows 侧后台包装脚本
  - 用本地 Python 启动 `serial_low_resource_rerun.py`，并把本地调度日志写到 `.runtime/low_resource_rerun/`

这个目录用于放置远程 clean runner 的初始化说明。

推荐的远程布局：

```text
/mnt/nas/share/home/liuke/prjs/uter/model_with_mamba/
├─ mamba_final/        # 冻结为历史快照
├─ mamba_runner/       # Git checkout，专门负责训练执行
└─ mamba_artifacts/    # logs / checkpoints / results
```

## 使用方式

1. 先准备一个真正可访问的 GitHub 仓库地址
2. 登录服务器
3. 设置环境变量：

```bash
export GIT_REMOTE_URL=git@github.com:你的账号/你的新仓库.git
export RUNNER_ROOT=/mnt/nas/share/home/liuke/prjs/uter/model_with_mamba/mamba_runner
export ARTIFACT_ROOT=/mnt/nas/share/home/liuke/prjs/uter/model_with_mamba/mamba_artifacts
export BRANCH=main
```

4. 执行：

```bash
bash runner/bootstrap_remote_runner.sh
```

完成后，`mamba_runner` 会成为新的执行目录，训练输出统一写到 `mamba_artifacts/`。
