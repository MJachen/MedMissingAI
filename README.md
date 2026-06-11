# MedMissingAI

MedMissingAI 是一个用于科研训练、求职展示和理解医学影像 AI 流程的 Python/PyTorch 原型项目。当前第一阶段只实现最小闭环：manifest 数据设计、NIfTI 读取、预处理接口、Dataset/Dataloader、简单缺失模态分类 baseline、训练/验证/测试脚本，以及一个基础梯度热图函数。

本项目不作为临床诊断软件。

## 目录结构

```text
configs/                    # YAML 配置
data/                       # manifest 示例；真实影像建议放 data/raw/，不提交
scripts/                    # 命令行入口
src/medmissingai/
  data/                     # manifest、NIfTI 读取、预处理、Dataset
  models/                   # 模型
  training/                 # loss、metrics、训练/评估 loop
  inference/                # 概率预测
  visualization/            # 热图/可视化
  utils/                    # config、seed、IO
tests/                      # 轻量测试
```

## 数据 Manifest

示例文件：`data/manifest_example.csv`

每一行表示一个病例或一次扫描：

| column | 含义 |
| --- | --- |
| `sample_id` | 样本唯一 ID |
| `label` | 分类标签，当前 baseline 使用整数类别，如 `0/1` |
| `split` | `train`、`val` 或 `test` |
| `t1,t1ce,t2,flair` | 每个 MRI 模态的 NIfTI 路径；空单元格表示该模态缺失 |

第一阶段的缺失模态策略很简单：缺失模态图像填 0，同时 `modality_mask` 中对应位置为 0。

## Tensor Shape 约定

单个 NIfTI 读入后：

```text
NiBabel raw: usually [X, Y, Z]
read_nifti output: [D, H, W]
preprocessor output per modality: [D, H, W]
dataset image output: [M, D, H, W]
dataset modality_mask output: [M]
dataloader image batch: [B, M, D, H, W]
dataloader modality_mask batch: [B, M]
model logits output: [B, num_classes]
predict probability output: [B, num_classes]
heatmap output: [D, H, W]
```

其中：

- `B` 是 batch size。
- `M` 是模态数量，例如 `[t1, t1ce, t2, flair]` 时 `M=4`。
- `D,H,W` 是预处理后的 3D 体数据尺寸，默认 `[64,128,128]`。

## 模块职责

- `src/medmissingai/data/manifest.py`：检查 CSV 字段、划分 train/val/test、判断模态是否缺失。
- `src/medmissingai/data/dataset.py`：读取 NIfTI，按模态堆叠为 `[M,D,H,W]`，同时输出缺失模态 mask。
- `src/medmissingai/data/transforms.py`：单个 3D volume 的预处理。当前包含 z-score 和 resize，后续可加入重采样、裁剪、配准后的空间检查。
- `src/medmissingai/models/baseline.py`：小型 3D CNN。输入图像通道和 broadcast 后的 modality mask，输出分类 logits。
- `src/medmissingai/training/engine.py`：训练一个 epoch、评估、保存 checkpoint。
- `src/medmissingai/inference/predict.py`：把 logits 转成类别概率。
- `src/medmissingai/visualization/heatmap.py`：简单 gradient saliency，中间切片叠加保存为 PNG。

## 环境安装

建议使用虚拟环境：

```bash
pip install -e ".[dev]"
```

如果暂时不安装包，也可以在项目根目录设置 `PYTHONPATH=src` 后运行脚本。

## Local development and remote server training workflow

本项目推荐把 Windows 本地作为 Codex 开发和小样本调试环境，把 Linux GPU 服务器作为正式训练环境。模型逻辑保持在 `src/medmissingai/` 中，环境差异只放在 `configs/` 里：

- `configs/data_local.yaml`：本地 manifest、模态和预处理尺寸。
- `configs/data_server.yaml`：服务器 manifest、模态和预处理尺寸。
- `configs/train_local.yaml`：本地调试用 `save_dir`、`batch_size`、`num_workers`、`device` 等训练参数。
- `configs/train_server.yaml`：服务器训练用 `save_dir`、`batch_size`、`num_workers`、`device` 等训练参数。
- `configs/model.yaml`：模型结构参数。

本地 Windows 调试：

```powershell
$env:PYTHONPATH = "src"
python scripts/train.py `
  --data-config configs/data_local.yaml `
  --train-config configs/train_local.yaml `
  --model-config configs/model.yaml
```

本地只建议使用小 manifest、小 `target_shape`、小 `batch_size` 和少量 epoch，确认数据读取、shape、import 路径和训练闭环能跑通。正式数据路径、输出目录、`batch_size`、`num_workers`、`device`、`save_dir` 都应只改 YAML，不要写死到 Python 代码里。

本地开发完成后提交并推送：

```powershell
git status
git add .gitignore README.md configs scripts src tests
git commit -m "Organize local and server training configs"
git push origin <branch-name>
```

服务器拉取最新代码：

```bash
cd /path/to/MedMissingAI
git pull origin <branch-name>
pip install -e ".[dev]"
```

服务器启动训练：

```bash
export PYTHONPATH=src
python scripts/train.py \
  --data-config configs/data_server.yaml \
  --train-config configs/train_server.yaml \
  --model-config configs/model.yaml
```

训练启动时会自动在 `training.save_dir` 中保存：

- `git_commit.txt`：当前 git commit hash。
- `command.txt`：当前启动命令。
- `config_snapshot/`：本次使用的配置文件快照。

从服务器下载结果到 Windows：

```powershell
scp -r user@server:/path/to/MedMissingAI/outputs/server_train .\outputs\
```

如果服务器使用不同 `save_dir`，把上面路径替换成 `configs/train_server.yaml` 里的 `training.save_dir`。

不应该提交到 GitHub 的文件包括：真实医学影像、预处理数据、训练输出、checkpoint、数组缓存和 pickle 缓存。具体规则已写入 `.gitignore`，包括 `data/raw/`、`data/processed/`、`results/`、`outputs/`、`checkpoints/`、`*.nii`、`*.nii.gz`、`*.pt`、`*.pth`、`*.ckpt`、`*.npy`、`*.npz`、`*.pkl`。

## 运行

先准备真实 NIfTI 文件并更新 manifest 路径。示例 manifest 只是字段模板，不包含真实影像。

训练：

```bash
python scripts/train.py \
  --data-config configs/data_local.yaml \
  --train-config configs/train_local.yaml \
  --model-config configs/model.yaml
```

测试：

```bash
python scripts/evaluate.py \
  --data-config configs/data_local.yaml \
  --train-config configs/train_local.yaml \
  --model-config configs/model.yaml \
  --split test
```

单样本预测：

```bash
python scripts/predict.py \
  --data-config configs/data_local.yaml \
  --train-config configs/train_local.yaml \
  --model-config configs/model.yaml \
  --sample-id case_0003
```

保存基础热图：

```bash
python scripts/predict.py \
  --data-config configs/data_local.yaml \
  --train-config configs/train_local.yaml \
  --model-config configs/model.yaml \
  --sample-id case_0003 \
  --save-heatmap
```

导出逐样本预测概率：

```bash
python scripts/export_predictions.py \
  --data-config configs/data_local.yaml \
  --train-config configs/train_local.yaml \
  --model-config configs/model.yaml \
  --split test
```

用验证集校准阈值，并分析测试集指标：

```bash
python scripts/export_predictions.py \
  --data-config configs/data_local.yaml \
  --train-config configs/train_local.yaml \
  --model-config configs/model.yaml \
  --split val

python scripts/analyze_predictions.py \
  --predictions outputs/local_debug/predictions_test.csv \
  --calibration-predictions outputs/local_debug/predictions_val.csv \
  --threshold-metric balanced_accuracy
```

## BraTS2021 Smoke Run

当前项目已支持从本机 BraTS2021 数据和标签 Excel 生成缺失模态 manifest：

- 影像目录：`E:\brats2021`
- 标签文件：`E:\EXPS\tensorexps\brats2021_label2020.xlsx`
- 标签约定：`label=1` 表示 HGG，`label=0` 表示 LGG
- 模态：`t1,t1ce,t2,flair`
- setting：四个模态的 15 种非空可用组合，缺失模态在 manifest 中留空

生成 smoke manifest：

```bash
$env:PYTHONPATH = "src"
conda run -n tensor python scripts/prepare_brats2021_manifest.py ^
  --data-root E:\brats2021 ^
  --label-file E:\EXPS\tensorexps\brats2021_label2020.xlsx ^
  --output data\brats2021_manifest_smoke.csv ^
  --settings all ^
  --max-cases 4 ^
  --train-frac 0.5 ^
  --val-frac 0.25 ^
  --seed 42
```

这个 smoke manifest 会抽取 4 个病例，并展开成 60 行：`4 cases * 15 settings`。

训练：

```bash
$env:PYTHONPATH = "src"
conda run -n tensor python scripts/train.py --config configs/brats2021_smoke.yaml
```

测试：

```bash
$env:PYTHONPATH = "src"
conda run -n tensor python scripts/evaluate.py --config configs/brats2021_smoke.yaml --split test
```

预测并保存热图：

```bash
$env:PYTHONPATH = "src"
conda run -n tensor python scripts/predict.py ^
  --config configs/brats2021_smoke.yaml ^
  --sample-id BraTS2021_01517__t1 ^
  --save-heatmap
```

服务器完整评估推荐流程：

```bash
export PYTHONPATH=src

python scripts/evaluate.py \
  --data-config configs/data_server.yaml \
  --train-config configs/train_server.yaml \
  --model-config configs/model.yaml \
  --checkpoint outputs/server_train/best.pt \
  --split test

python scripts/export_predictions.py \
  --data-config configs/data_server.yaml \
  --train-config configs/train_server.yaml \
  --model-config configs/model.yaml \
  --checkpoint outputs/server_train/best.pt \
  --split val

python scripts/export_predictions.py \
  --data-config configs/data_server.yaml \
  --train-config configs/train_server.yaml \
  --model-config configs/model.yaml \
  --checkpoint outputs/server_train/best.pt \
  --split test

python scripts/analyze_predictions.py \
  --predictions outputs/server_train/predictions_test.csv \
  --calibration-predictions outputs/server_train/predictions_val.csv \
  --threshold-metric balanced_accuracy
```

`analyze_predictions.py` 会保存整体指标、校准阈值指标、confusion matrix、sensitivity、specificity、balanced accuracy、precision、recall，以及按 `availability` 分组的缺失模态组合指标。

主表文件会写到 `*_main_table.csv`，固定包含：

- `auc`
- `balanced_accuracy`
- `sensitivity`
- `specificity`
- `macro_f1`

其中 `default_0_5` 使用默认阈值 `0.5`，`validation_calibrated` 使用验证集按 `--threshold-metric` 选出的阈值。若预测文件包含 `availability`，脚本还会在 `*_metrics_by_availability_calibrated.csv` 中按每种可用模态组合单独校准阈值。

下一轮训练默认按 `training.checkpoint_metric: balanced_accuracy` 选择 `best.pt`，并用 `training.class_weights: balanced` 启用 class-weighted CE。低成本缺失模态 baseline 可单独打开，避免多个改动混在一次实验里：

```yaml
training:
  modality_dropout_prob: 0.1

model:
  use_learnable_missing_token: true
```

## 第 1 周任务清单

1. 用真实数据复制并修改 `data/manifest_example.csv`，确认每个 NIfTI 路径能被读取。
2. 用 3 到 5 个样本跑通 Dataset，检查输出 shape：`image [M,D,H,W]`，`modality_mask [M]`。
3. 确认标签定义和 `model.num_classes` 一致。
4. 用很小的 `target_shape` 和少量 epoch 跑通训练闭环。
5. 检查 `outputs/baseline/history.json`、`best.pt`、测试指标 JSON 是否生成。
6. 对 1 个测试样本生成预测概率和热图 PNG。
7. 记录下一阶段要改进的点：更稳健预处理、真实医学指标、交叉验证、Grad-CAM、更强缺失模态模型。
