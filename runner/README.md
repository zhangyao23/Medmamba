# runner

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