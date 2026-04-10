# Mamba Final - 实验记录与项目索引

> 最后更新: 2026-04-01

## 数据集

数据路径 (NAS, 新数据, 只读): `/mnt/nas/share/home/liuke/data2/TumorSegNew/TumorSegmentation/`
数据路径 (NAS, 旧数据, **已污染, 禁止使用**): `/mnt/nas/share/home/liuke/data2/TumorSegmentation/`
数据路径 (本地 SSD): `/home/lk/data/TumorSegmentation/`
数据概览文档: `data_overview.md`

共 11 个器官, 4206 个 volume (训练 3359 + 测试 847)。
数据索引文件: `all_entries_train.json`, `all_entries_test.json` (位于本地 SSD 数据根目录下)

### 本地 SSD 数据结构

本地 SSD 已从新 NAS 同步, 每个器官目录下包含:
- `images/` -- 原始影像文件
- `labels/` -- 原始多类别标签文件 (只读, 禁止修改)
- `labels_binary/` -- 二值化肿瘤标签 (由 `umamba_baseline/generate_binary_labels.py` 生成, 肿瘤=1, 其他=0)

### 标签处理要点

每个器官的肿瘤标签值不同, 加载 ground truth mask 时必须使用 `TUMOR_LABEL_MAP` 精确匹配:

| 器官 | 标签类型 | 肿瘤标签值 |
|------|---------|-----------|
| Bladder_Tumor_00 | 二值 | 1 |
| Breast_Tumor_00 | 二值 | 1 |
| Cervix_Tumor_00 | 多类别 | 2 |
| Colon_Tumor_00 | 多类别 | 14 |
| Kidney_Tumor_00 | 多类别 | 3 |
| Liver_Tumor_00 | 多类别 | 14 |
| Lung_Tumor_00 | 二值 | 1 |
| Lung_Tumor_01 | 多类别 | 5 |
| Pancreas_Tumor_00 | 多类别 | 14 |
| Prostate_Tumor_00 | 多类别 | 3 (注意浮点精度, 需 np.round()) |
| Uterus_Tumor_00 | 多类别 | 3 |

禁止使用 `(mask > 0)` 或 `(mask >= 2)` 这类启发式规则提取肿瘤, 必须用 `(mask == tumor_label)`。

### 安全规则

- 禁止使用 `os.symlink()` 或 `os.link()`, 必须使用 `shutil.copy2()` 创建独立副本
- 写入文件前必须用 `os.path.islink()` 检查目标是否为符号链接, 是则拒绝写入
- 脚本只允许写入自己的输出目录, 不得写入 NAS 原始数据路径
- 批量处理必须遵循"先拷贝再修改"模式

---

## 实验 A: 全监督分割 (Full Supervision Segmentation)

- 训练脚本: `scripts/train_fullsup_seg_ddp.py`
- 配置文件: `configs/fullsup_seg.yaml`
- 启动脚本: `run_mamba.sh`
- Checkpoint 目录: `volumetric_checkpoints_fullsup_seg/`
- 状态: **训练完成** (100 epoch, 使用修复后的 `TUMOR_LABEL_MAP` 代码)
- 最终结果: **Best PosDice=0.4307 @ thr=0.90** (epoch 84), Patch AUC=0.9446, Vol AUC=0.7851
- Best Checkpoint: `volumetric_checkpoints_fullsup_seg/best_seg.pth`
- 训练时间: 2026-03-27 18:30 ~ 2026-03-29 04:37 (~34 小时)
- 训练日志: `/tmp/fullsup_seg_train.log`
- 训练报告: `实验A_全监督分割训练报告.md`

## 实验 B: 弱监督 MIL -- v20_attc (True Weak Supervision)

- 训练脚本: `scripts/train_volumetric_ddp.py`
- 配置文件: `configs/v20_retrain.yaml` (lambda_gt_code=0, lambda_patch_seg=0, lambda_voxel_seg=0)
- Checkpoint 目录: `/mnt/nas/share/home/liuke/prjs/uter/model_with_mamba/volumetric_checkpoints_v20_attc/`
- Best Checkpoint: `best_seg_model.pth` (epoch 35)
- 状态: **评估完成** (使用修正 TUMOR_LABEL_MAP, 2026-03-29)
- 模型架构: ResNet-34 3D Encoder -> Codebook VQ (256 codes, 192H+64C) -> Z-Order -> Bidirectional Mamba -> Attention MIL Head
- 分类结果: **AUC=0.8460**, PR-AUC=0.9669, thr>=0.30 时 Specificity=1.0 (零假阳性)
- 分割结果 (Recon Error): **Best PosDice=0.1173** @ thr=0.78 (旧 `mask>0` 逻辑下为 ~0.22)
- 评估结果目录: `eval_results/v20_attc_recon_seg/`, `eval_results/v20_attc_codebook_seg/`
- 详细实验报告: `paper/experiments/completed/v20_attc_弱监督分割优化实验报告.md`

### 实验 B-ablation: Mamba-first 弱监督消融 (v20_mamba_first_weak)

- 训练脚本: `scripts/train_volumetric_ddp.py`
- 配置文件: `configs/v20_mamba_first_weak.yaml` (基于 v20_retrain.yaml, 仅改 `codebook_after_mamba: true`)
- 启动脚本: `run_mamba_first_weak.sh`
- Checkpoint 目录: `volumetric_checkpoints_mamba_first_weak/`
- 训练日志: `mamba_first_weak_train.log`
- 状态: **训练发散, 已终止** (Epoch ~115 开始 NaN, 在 Epoch 126 崩溃)
- Best Checkpoint: `best_model.pth` (AUC=0.8534), `best_seg_model.pth` (PosDice=0.0599)
- 模型架构: ResNet-34 3D Encoder -> Z-Order -> Bidirectional Mamba -> Codebook VQ (256 codes, 192H+64C) -> Attention MIL Head
- 消融目的: 与实验 B (Codebook-first) 公平对比, 唯一变量为 Codebook/Mamba 位置
- 弱监督: lambda_gt_code=0, lambda_patch_seg=0, lambda_voxel_seg=0 (与 v20_attc 完全一致)
- 发散原因: Mamba-first 架构下 Codebook 量化 Mamba 输出 (不断变化的目标), 形成正反馈环路导致 VQ loss 持续上升最终 NaN。Loss 轨迹: 3.0 (Epoch ~55) -> 4.3 (Epoch ~100) -> 5.1 -> NaN (Epoch ~115)。已保存的 best checkpoint 在发散前保存, 可用于评估。

### 实验 B-ablation2/3: 分区冻结消融 (已取消)

- 状态: **已取消**
- 配置文件: `configs/v20_cb_first_frozen_partition.yaml`, `configs/v20_mamba_first_frozen_partition.yaml`
- 取消原因: 代码分析发现 `enable_dynamic_partition()` 控制的是 Phase 2 的量化路由模式 (全码本 vs 受限路由), 而非分区定义的动态更新。分区定义 (`healthy_code_mask`) 在 `complete_phase1()` 后只设置一次, 从不更新, 已是冻结状态。跳过 `enable_dynamic_partition()` 导致 Phase 2 健康样本路由被限制到前 192 codes, 与 Phase 1 的全码本路由不一致, 造成 Epoch 31 loss 瞬间从 1.05 跳至 15.9 后 NaN 崩溃。此消融无有效对比意义。
- 相关代码修改已还原: `train_volumetric_ddp.py` 恢复为无条件调用 `enable_dynamic_partition()`

### 实验 B-ablation4: Mamba-first 稳定训练 (v20_mamba_first_stable)

- 训练脚本: `scripts/train_mamba_first_ddp.py` (独立副本, 不影响原始 train_volumetric_ddp.py)
- 配置文件: `configs/v20_mamba_first_stable.yaml`
- 启动脚本: `run_mamba_first_stable.sh`
- Checkpoint 目录: `volumetric_checkpoints_mamba_first_stable/`
- 训练日志: `mamba_first_stable_train.log`
- 状态: **已暂停** (2026-04-01, Epoch 63 暂停, 释放 GPU 给 cb_first_mamba_pretrain 实验)
- 模型架构: ResNet-34 3D -> Z-Order -> Bidirectional Mamba -> LayerNorm -> Codebook VQ (256 codes, 192H+64C) -> Attention MIL Head
- 与 v20_mamba_first_weak 的区别:
  - Phase 0 (30 epoch): Mamba 参与 healthy-only reconstruction pretrain (原来 Mamba 不参与)
  - Mamba 输出后加 LayerNorm 稳定 Codebook 输入分布
  - Phase 0 从 3 epoch 延长到 30 epoch
  - lr 从 3e-5 降到 2e-5
- 设计理由: 学长认为 Mamba 应放在 Codebook 前面加强特征表示, 帮助 Codebook 学到更好的离散化。之前发散的根本原因是 Codebook 从未在 Mamba 输出上做过 pretrain, Phase 2 突然切换导致分布不匹配。本方案通过充分的 Phase A pretrain 解决此问题。

### 实验 B-ablation5: Codebook-first + Mamba 参与 Phase 0 预训练 (v20_cb_first_mamba_pretrain)

- 训练脚本: `scripts/train_cb_first_mamba_pretrain_ddp.py` (独立副本, 基于 train_volumetric_ddp.py)
- 配置文件: `configs/v20_cb_first_mamba_pretrain.yaml`
- 启动脚本: `run_cb_first_mamba_pretrain.sh`
- Checkpoint 目录: `volumetric_checkpoints_cb_first_mamba_pretrain/`
- 训练日志: `cb_first_mamba_pretrain_train.log`
- 状态: **训练中** (2026-04-01 启动, GPU 0,1, 2 卡 DDP)
- 模型架构: ResNet-34 3D Encoder -> Codebook VQ (256 codes, 192H+64C) -> Z-Order -> Bidirectional Mamba -> Attention MIL Head
- 与 V2 baseline (v20_attc) 的区别:
  - Phase 0 (10 epoch): Mamba 解冻并参与 reconstruction pretrain
  - Phase 0 重建路径: ResNet -> Codebook -> Mamba -> Decoder (原来是 ResNet -> Codebook -> Decoder)
  - Phase 1/2 重建路径不变: ResNet -> Codebook -> Decoder
  - pretrain_healthy_vqvae_epochs 从 3 增加到 10
  - 使用本地 SSD 数据 (all_entries_train_local.json)
  - world_size: 2 (适配 GPU 可用性)
- 设计理由: 学长认为 Codebook 本身参数少学习能力弱, 如果一开始离散化就分错了后面 Mamba 起不了作用。让 Mamba 在 Phase 0 就参与重建, 重建 loss 梯度通过 Mamba 回传到 Codebook, 迫使 Codebook 产生对 Mamba 友好的离散表示, 提升离散化质量。
- 代码修改:
  - `forward()` codebook-first 分支: 通过 `_pretrain_mode` flag 控制 Decoder 输入 (Phase 0: context_orig, Phase 1+: quantized)
  - `set_trainable_for_pretrain()`: 新增 `include_mamba_in_pretrain` 参数, Phase 0 时解冻 Mamba + SpatialScanner

### 实验 B (旧): volumetric_v2 (非真正弱监督, 参考用)

- 配置文件: `configs/volumetric_v2_config.yaml` (lambda_patch_seg=1.0, 使用了 GT mask)
- Checkpoint 目录: `volumetric_checkpoints_v2/`
- 状态: **已评估, 但非真正弱监督** (lambda_patch_seg=1.0 导致训练中使用了 GT mask)
- 分类 AUC=0.7248, Patch Dice ~0 (因训练时用旧标签逻辑生成 GT)

## 实验 C: 全监督 MIL + Codebook

- 训练脚本: `scripts/train_fullsup_mil_codebook_ddp.py`
- 配置文件: `configs/fullsup_mil_codebook.yaml`
- 启动脚本: `run_fullsup_mil_codebook.sh`
- 状态: 待启动, 代码已修复标签逻辑

## 实验 D: Baseline (无 Mamba)

- 训练脚本: `scripts/train_baseline_ddp.py`
- 配置文件: `configs/baseline_common.yaml`
- 状态: 待定

## 实验 E: U-Mamba Baseline

- 目录: `umamba_baseline/`
- 数据转换: `umamba_baseline/convert_to_nnunet.py` (已重写, 使用 `shutil.copy2`, 从 `labels_binary/` 读取)
- nnUNet 数据: `umamba_baseline/nnUNet_raw/Dataset501_TumorSeg/`
- 状态: **数据转换已完成, nnUNet 预处理正在另一台服务器运行**
- 工具脚本:
  - `umamba_baseline/generate_binary_labels.py` -- 使用 `TUMOR_LABEL_MAP` 在本地数据集生成 `labels_binary/`
  - `umamba_baseline/convert_to_nnunet.py` -- 将数据转为 nnUNet 格式 (安全版, 无 symlink)
  - `umamba_baseline/fix_geometry_safe.py` -- 修复 image/label 间的 affine 矩阵不匹配
  - `umamba_baseline/run_full_pipeline.sh` -- 完整流水线包装脚本
  - `umamba_baseline/run_umamba_baseline.sh` -- nnUNet 预处理+训练+评估脚本
- 预处理命令 (在其他服务器运行):

```bash
cd /path/to/mamba_final/umamba_baseline
export nnUNet_raw="$(pwd)/nnUNet_raw"
export nnUNet_preprocessed="$(pwd)/nnUNet_preprocessed"
export nnUNet_results="$(pwd)/nnUNet_results"
conda activate uter_mamba
nnUNetv2_plan_and_preprocess -d 501 -c 3d_fullres --verify_dataset_integrity -np 8 -npfp 8
```

---

## 评估脚本

| 脚本 | 用途 |
|------|------|
| `scripts/evaluate_segmentation.py` | 全量分割评估 (Dice/IoU) |
| `scripts/eval_checkpoint.py` | Checkpoint 快速评估 |
| `scripts/eval_patch_level.py` | Patch 级别评估 |

---

## 代码结构

```
mamba_final/
  scripts/           # 训练和评估脚本
  configs/           # YAML 配置文件
  src/
    data/            # 数据加载 (volumetric_dataset.py, volumetric_loader.py)
    models/          # 模型定义
    training/        # 训练工具
    utils/           # 通用工具
  umamba_baseline/   # U-Mamba baseline 实验
    generate_binary_labels.py   # 生成二值化标签
    convert_to_nnunet.py        # 数据格式转换 (安全版)
    fix_geometry_safe.py        # 修复 affine 不匹配
    nnUNet_raw/                 # nnUNet 原始数据
    nnUNet_preprocessed/        # nnUNet 预处理输出
    U-Mamba/                    # U-Mamba 代码库
  paper/             # 论文相关文档
    outline/         # 论文大纲
    experiments/     # 实验报告 (completed/ + planned/)
    resources/       # 资源索引
  archive/           # 已归档的过时文档 (README.md, USAGE_GUIDE.md, 研究方案.md)
```

---

## 文档索引

| 文档 | 说明 | 状态 |
|------|------|------|
| `实验AB_Mamba模型结果总结.md` | 实验 A+B 完整对比总结 (飞书文档版, 含详细数据分析) | 有效 |
| `实验A_全监督分割训练报告.md` | 实验 A 完整训练结果 (100 epoch, Best PosDice=0.4307) | 有效 |
| `data_overview.md` | 数据集各器官的详细标签定义 | 有效 |
| `弱监督论文故事线.md` | 弱监督 MIL 叙事框架 | 有效 (PosDice 已重新验证: 0.1173) |
| `v20_性能报告.md` | v20 完整技术报告 | 有效 (PosDice 已重新验证) |
| `消融实验_Codebook位置对比.md` | CB-first vs Mamba-first | 有效 (PosDice 数值需按新标签理解) |
| `消融实验_Patch尺寸对比.md` | 16x16x16 vs 32x32x32 | 有效 (PosDice 数值需按新标签理解) |
| `模型架构说明.md` | 3D Volumetric MIL 架构详解 | 有效 |
| `paper/outline/论文大纲_StoryB.md` | Story B 论文大纲 | 有效 |
| `paper/experiments/planned/待完成实验计划.md` | 待完成实验清单 | 有效 |
| `paper/experiments/completed/Exp1_Baseline对比结果.md` | Baseline 对比 (GAP/ABMIL/TransMIL/DSMIL) | 有效 |
| `paper/experiments/completed/v20_attc_弱监督分割优化实验报告.md` | v20_attc 纯弱监督优化 (256 codes) | 有效 (PosDice 已重新验证: 0.1173, 旧值 0.22 基于错误标签) |
| `paper/resources/资源索引.md` | 代码/数据/checkpoint 位置索引 | 已更新路径 |

---

## 已知问题与修复记录

### 2026-03-30: phase2_freeze_healthy_dynamic 修改已还原

**操作**: 之前为分区冻结消融添加了条件判断 (跳过 `enable_dynamic_partition()`), 但实际效果是改变了 Phase 2 路由规则 (健康样本被限制到前 192 codes), 导致 Phase 2 第一个 epoch loss 暴涨至 15.9 后 NaN。
**原因分析**: `enable_dynamic_partition()` 的语义是"启用全码本路由", 而非"启用动态分区更新"。分区定义本身已在 `complete_phase1()` 后固定, 无需额外冻结。
**还原**: 恢复为无条件调用 `enable_dynamic_partition()`, 代码回到修改前状态。

### 2026-03-28: U-Mamba baseline 数据重建

**操作**: 为实验 E 重建了完整的数据处理流水线:
1. `generate_binary_labels.py` -- 使用 `TUMOR_LABEL_MAP` 在本地 SSD 每个器官目录下生成 `labels_binary/` 子目录, 肿瘤=1, 其他=0
2. `convert_to_nnunet.py` -- 完全重写, 使用 `shutil.copy2()` 替代 symlink, 从 `labels_binary/` 读取标签, 阴性样本生成全零 mask
3. `fix_geometry_safe.py` -- 修复 34 个样本的 image/label affine 矩阵不匹配问题
4. 清除并重新生成 `nnUNet_raw/Dataset501_TumorSeg/` (1282 训练 + 325 测试)
**结果**: 数据转换完成, 预处理正在另一台服务器运行

### 2026-03-28: 过时文档归档

**操作**: 将 3 个严重过时的文档移入 `archive/` 目录:
- `README.md` -- 描述旧的 2D 架构, 与当前 3D patch-based 实现不符
- `USAGE_GUIDE.md` -- 同上, 引用错误的环境名和模型参数
- `研究方案.md` -- 最初期提案, 基于 2D 切片设想

**操作**: 为 5 个包含旧评估指标的文档添加了 `[2026-03-28 警告]` 标注

### 2026-03-26: 标签加载 bug 修复

**问题**: 所有脚本中的 `load_gt_mask_hwz` 使用 `(mask > 0)` 提取肿瘤, 对多类别器官会把器官区域也算作肿瘤。
**修复**: 添加 `TUMOR_LABEL_MAP`, 按器官精确匹配肿瘤标签值 `(mask == tumor_label)`。
**影响文件**: `train_fullsup_seg_ddp.py`, `train_fullsup_mil_codebook_ddp.py`, `evaluate_segmentation.py`, `eval_checkpoint.py`, `eval_patch_level.py`, `train_baseline_ddp.py`, `train_volumetric_ddp.py`

### 2026-03-26: 验证阶段内存泄漏修复

**问题**: `train_fullsup_seg_ddp.py` 验证循环中大量 3D mask 累积在内存中, 导致 swap 耗尽和后续 epoch 严重变慢。
**修复**: 验证结束后显式 `del` 大列表并调用 `gc.collect()`。

### 2026-03-22: NAS 数据污染事故

**问题**: `umamba_baseline/fix_labels.py` 通过符号链接将二值化标签写回了 NAS data2 原始文件, 导致多类别器官阳性样本标签不可逆损坏。
**影响范围**: Cervix (67), Colon (126), Kidney (246/485), Liver (15/201), Lung_01 (418/421), Pancreas (226/500), Prostate (106/316), Uterus (266/300) 的阳性标签文件。
**修复状态**: 新的正确数据已更新到 `/mnt/nas/share/home/liuke/data2/TumorSegNew/TumorSegmentation/`。本地 SSD 已同步。

---

## 环境

- Conda 环境: `uter_mamba`
- Python: `/home/lk/.pyenv/versions/miniconda3-latest/envs/uter_mamba/bin/python`
- GPU: 7 卡 DDP (`--nproc_per_node=7`)
