# mamba_source_repo

这是从远程历史工作区 `mamba_final` 中选择性回收出来的本地干净源码仓库。

当前仓库只保留了首批关键内容：

- `src/`、`scripts/`、`configs/`
- `run_*.sh`
- `requirements*.txt`、`setup_env.sh`
- `all_entries_*.json`
- `AGENTS.md`、`data_overview.md`

当前仓库明确不包含：

- 大体积 checkpoint
- 训练日志
- 评估输出
- 旧工作区中的临时结果和缓存

## 默认运行约定

训练脚本现在统一支持：

- `--output_root`
- `--run_name`

如果不显式传这两个参数，默认会把产物写到仓库同级目录下的 `mamba_artifacts/`：

```text
<repo_parent>/
├─ mamba_source_repo/
└─ mamba_artifacts/
   └─ <run_name>/
      ├─ checkpoints/
      ├─ logs/
      └─ results/
```

同时，配置里的 `train_json/test_json` 已从旧远程绝对路径中解耦，改为仓库内索引文件。

## 当前阻塞点

远程服务器已经能连通 GitHub SSH，但旧工作区配置的：

`git@github.com:zhangyao23/Mamba_final.git`

当前不可直接拉取。结合现场检查，问题更像是仓库地址/权限主体不一致，而不是网络不通。

在 GitHub 中枢真正可用前，推荐先把这里当作**本地主源码真源**，再通过新的 GitHub 仓库接上 `mamba_runner`。

## 建议下一步

1. 在 GitHub 上新建一个可访问的新仓库
2. 在本目录执行 `git init`
3. 首次提交本地干净仓库
4. 将远程 `mamba_runner` 指向新的 GitHub 仓库
5. 使用 `runner/bootstrap_remote_runner.sh` 初始化远程 runner