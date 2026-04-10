# TumorSegmentation 数据集概览

数据路径 (NAS, 正确数据): `/mnt/nas/share/home/liuke/data2/TumorSegNew/TumorSegmentation/`
数据路径 (本地 SSD): `/home/lk/data/TumorSegmentation/`
索引文件 (项目内): `all_entries_train.json` (3359), `all_entries_test.json` (847)

> 注: 新数据目录为只读 (`dr-xr-xr-x`), 索引 JSON 文件存放在项目根目录
> `/mnt/nas/share/home/liuke/prjs/uter/model_with_mamba/mamba_final/` 下。

## 整体结构

```
TumorSegmentation/
  datainfo.xlsx               # 数据集元信息
  <Organ_Tumor_XX>/
    dataset.json              # 该器官的标签定义 (含 DOI)
    images/                   # 影像文件 (.nii.gz)
    labels/                   # 标签文件 (.nii.gz)
```

每条 entry 包含 `image`(影像路径) 和 `label`(volume-level 标签, 1=阳性/有肿瘤, 0=阴性/无肿瘤)。

## 各器官数据

### Bladder_Tumor_00 - 膀胱肿瘤

- 文件命名: `Center1_001.nii.gz`
- DOI: `10.5281/zenodo.13622759`
- 训练集: 176 (阳性 176, 阴性 0) | 测试集: 45 (阳性 45, 阴性 0)
- 标签类型: **二值**
- 标签定义:

| 标签值 | 含义 |
|--------|------|
| 0 | background |
| 1 | tumor |

### Breast_Tumor_00 - 乳腺肿瘤

- 文件命名: `DUKE_001_0001.nii.gz`, `ISPY1_1130_0001.nii.gz`, `NACT_01_0001.nii.gz`
- 来源: Duke (291) + I-SPY (1151) + NACT (64) = 1506
- DOI: `10.1038/s41597-025-04707-4`
- 训练集: 1204 (阳性 1204, 阴性 0) | 测试集: 302 (阳性 302, 阴性 0)
- 标签类型: **二值**
- 标签定义:

| 标签值 | 含义 |
|--------|------|
| 0 | background |
| 1 | tumor |

### Cervix_Tumor_00 - 宫颈肿瘤

- 文件命名: `CCTH-A01_MR1_SAG_T2.nii.gz` (MRI, 矢状面 T2)
- DOI: `10.7937/ERZ5-QZ59`
- 训练集: 53 (阳性 53, 阴性 0) | 测试集: 14 (阳性 14, 阴性 0)
- 标签类型: **多类别**
- 标签定义:

| 标签值 | 含义 |
|--------|------|
| 0 | background |
| 1 | uterus (子宫) |
| 2 | **tumor (肿瘤)** |

### Colon_Tumor_00 - 结直肠肿瘤

- 文件命名: `FLARE23_0012.nii.gz` (来源: FLARE 2023 Challenge)
- DOI: `10.48550/arXiv.1902.09063`
- 训练集: 100 (阳性 100, 阴性 0) | 测试集: 26 (阳性 26, 阴性 0)
- 标签类型: **多类别** (标签值不连续)
- 标签定义:

| 标签值 | 含义 |
|--------|------|
| 0 | background |
| 14 | **tumor (肿瘤)** |

> 注: 标签 1-13 为其他腹部器官, 但该子集只保留了 background 和 tumor。

### Kidney_Tumor_00 - 肾脏肿瘤

- 文件命名: `case_00000.nii.gz` (来源: KiTS Challenge)
- DOI: `10.1016/j.media.2020.101821`
- 训练集: 388 (阳性 195, 阴性 193) | 测试集: 97 (阳性 51, 阴性 46)
- 标签类型: **多类别**
- 标签定义:

| 标签值 | 含义 |
|--------|------|
| 0 | background |
| 1 | kidney (肾脏) |
| 2 | cyst (囊肿) |
| 3 | **tumor (肿瘤)** |

### Liver_Tumor_00 - 肝脏肿瘤

- 文件命名: `FLARE23_0008.nii.gz` (来源: FLARE 2023 Challenge)
- DOI: `10.48550/arXiv.1902.09063`
- 训练集: 160 (阳性 91, 阴性 69) | 测试集: 41 (阳性 27, 阴性 14)
- 标签类型: **多类别** (14 类腹部器官 + 肿瘤)
- 标签定义:

| 标签值 | 含义 |
|--------|------|
| 0 | background |
| 1 | liver (肝脏) |
| 2 | right kidney (右肾) |
| 3 | spleen (脾脏) |
| 4 | pancreas (胰腺) |
| 5 | Aorta (主动脉) |
| 6 | inferior vena cava (下腔静脉) |
| 7 | right adrenal gland (右肾上腺) |
| 8 | left adrenal gland (左肾上腺) |
| 9 | gallbladder (胆囊) |
| 10 | esophagus (食管) |
| 11 | stomach (胃) |
| 12 | duodenum (十二指肠) |
| 13 | left kidney (左肾) |
| 14 | **tumor (肿瘤)** |

### Lung_Tumor_00 - 肺部肿瘤 (数据集 A)

- 文件命名: `lung_001.nii.gz` (来源: Medical Segmentation Decathlon)
- DOI: `10.48550/arXiv.1902.09063`
- 训练集: 50 (阳性 50, 阴性 0) | 测试集: 13 (阳性 13, 阴性 0)
- 标签类型: **二值**
- 标签定义:

| 标签值 | 含义 |
|--------|------|
| 0 | background |
| 1 | tumor |

### Lung_Tumor_01 - 肺部肿瘤 (数据集 B)

- 文件命名: `LUNG1-001_CT.nii.gz` (来源: NSCLC-Radiomics / StructSeg)
- DOI: `10.7937/K9/TCIA.2015.PF0M9REI`
- 训练集: 336 (阳性 333, 阴性 3) | 测试集: 85 (阳性 85, 阴性 0)
- 标签类型: **多类别**
- 标签定义:

| 标签值 | 含义 |
|--------|------|
| 0 | background |
| 1 | lung (肺) |
| 2 | heart (心脏) |
| 3 | esophagus (食管) |
| 4 | spinal cord (脊髓) |
| 5 | **tumor (肿瘤)** |

### Pancreas_Tumor_00 - 胰腺肿瘤

- 文件命名: `FLARE23_0007.nii.gz` (来源: FLARE 2023 Challenge)
- DOI: `10.48550/arXiv.1902.09063`
- 训练集: 400 (阳性 226, 阴性 174) | 测试集: 100 (阳性 55, 阴性 45)
- 标签类型: **多类别** (14 类腹部器官 + 肿瘤, 与 Liver_Tumor_00 相同标签体系)
- 标签定义:

| 标签值 | 含义 |
|--------|------|
| 0 | background |
| 1 | liver (肝脏) |
| 2 | right kidney (右肾) |
| 3 | spleen (脾脏) |
| 4 | pancreas (胰腺) |
| 5 | Aorta (主动脉) |
| 6 | inferior vena cava (下腔静脉) |
| 7 | right adrenal gland (右肾上腺) |
| 8 | left adrenal gland (左肾上腺) |
| 9 | gallbladder (胆囊) |
| 10 | esophagus (食管) |
| 11 | stomach (胃) |
| 12 | duodenum (十二指肠) |
| 13 | left kidney (左肾) |
| 14 | **tumor (肿瘤)** |

### Prostate_Tumor_00 - 前列腺肿瘤

- 文件命名: `001_adc.nii.gz`, `001_t2.nii.gz` (MRI, 含 ADC 和 T2 序列)
- DOI: `10.5281/zenodo.6481141`
- 训练集: 252 (阳性 84, 阴性 168) | 测试集: 64 (阳性 22, 阴性 42)
- 标签类型: **多类别**
- 标签定义:

> 注: 该子集标签存在浮点精度问题 (如 0.9999999997671694 而非 1.0), 使用时需做 `np.round()` 处理。

| 标签值 | 含义 |
|--------|------|
| 0 | background |
| 1 | transitional zone (移行区) |
| 2 | peripheral zone (外周区) |
| 3 | **tumor (肿瘤)** |

### Uterus_Tumor_00 - 子宫肿瘤

- 文件命名: `UMD_221129_001_t2.nii.gz` (MRI, T2 序列)
- DOI: `10.6084/m9.figshare.23541312.v3`
- 训练集: 240 (阳性 240, 阴性 0) | 测试集: 60 (阳性 60, 阴性 0)
- 标签类型: **多类别**
- 标签定义:

| 标签值 | 含义 |
|--------|------|
| 0 | background |
| 1 | uterine wall (子宫壁) |
| 2 | uterine cavity (子宫腔) |
| 3 | **tumor (肿瘤)** |
| 4 | nabothian cyst (纳博特囊肿) |

## 汇总

| 器官 | 标签类型 | 肿瘤标签值 | 训练集 | 测试集 | 总计 |
|------|---------|-----------|--------|--------|------|
| Bladder_Tumor_00 | 二值 | 1 | 176 | 45 | 221 |
| Breast_Tumor_00 | 二值 | 1 | 1204 | 302 | 1506 |
| Cervix_Tumor_00 | 多类别 | 2 | 53 | 14 | 67 |
| Colon_Tumor_00 | 多类别 | 14 | 100 | 26 | 126 |
| Kidney_Tumor_00 | 多类别 | 3 | 388 | 97 | 485 |
| Liver_Tumor_00 | 多类别 | 14 | 160 | 41 | 201 |
| Lung_Tumor_00 | 二值 | 1 | 50 | 13 | 63 |
| Lung_Tumor_01 | 多类别 | 5 | 336 | 85 | 421 |
| Pancreas_Tumor_00 | 多类别 | 14 | 400 | 100 | 500 |
| Prostate_Tumor_00 | 多类别 | 3 | 252 | 64 | 316 |
| Uterus_Tumor_00 | 多类别 | 3 | 240 | 60 | 300 |
| **Total** | | | **3359** | **847** | **4206** |
