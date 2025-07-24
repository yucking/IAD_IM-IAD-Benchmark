# IM-IAD: 制造业工业图像异常检测基准


致力于为研究者提供一个统一的图像异常检测验证环境，配有标准配置和方法。

[[Main Page]](https://github.com/M-3LAB) [[Survey]](https://github.com/M-3LAB/awesome-industrial-anomaly-detection) [[Benchmark]](https://github.com/M-3LAB/open-iad) [[Result]](https://github.com/M-3LAB/IM-IAD)
## Envs

```bash
pytorch 1.10.1 or 1.8.1
conda activate open-iad
pip3 install -r requirements.txt

# example
pip install scikit-image
pip instll scikit-learn
pip install opencv-python
```

## Project Instruction
```bash
IM-IAD
├── arch                 # 模型基础类
├── augmentation         # 数据增强
├── checkpoints          # 预训练模型或依赖
├── configuration        # 配置文件目录
│   ├── 1_model_base     # 模型配置（优先级最高）
│   ├── 2_train_base     # 训练配置（中等优先级）
│   ├── 3_dataset_base   # 数据集配置（优先级最低）
│   ├── config.py        # 主配置入口
│   ├── device.py        # 设备管理
│   └── registeration.py # 注册新模型、数据集等
├── data_io              # 数据加载接口
├── dataset              # 数据集处理
├── loss_function        # 损失函数
├── metric               # 评估指标
├── models               # 模型模块的基础层
├── optimizer            # 优化器
├── paradigms            # 学习范式（集中式、联邦式）
│   ├── centralized
│   │   ├── c2d.py       # 2D集中式训练
│   │   ├── c3d.py       # 3D集中式训练
│   └── federated
│       └── f2d.py       # 2D联邦式训练
├── tools                # 工具函数
├── work_dir             # 保存结果目录
├── main.py              # 项目主入口
└── requirements.txt     # 依赖库列表
```
## 教程

### 如何添加新方法或新数据集？

> Please refer to the following steps:

+ Register your NEW METHOD (e.g., MODEL, NET, DATASET, SETTING, SERVER) in [configuration/registration.py](configuration/registration.py)
+ Add names of MODEL, NET, DATASET, SETTING into [configuration/config.py](configuration/config.py)
+ Implement MODEL in [arch/_example.py](arch/_example.py) and [models/_example/net_example.py](models/_example/net_example.py)
+ Put MODEL configuration in [configuration/1_model_base/_example.yaml](configuration/1_model_base/_example.yaml)
+ Implement DATASET in [dataset/_example.py](dataset/_example.py)
+ Put DATASET configuration in [configuration/3_dataset_base/_example.yaml](configuration/3_dataset_base/_example.yaml)
+ Implement NEW SETTING in [data_io/data_holder.py](data_io/data_holder.py)
+ If provide NEW SETTING, update OUTPUT path of results in [tools/record_helper.py](tools/record_helper.py)
+ Add NEW METHOD description in [README.md](README.md)
+ Shell command, "python3 main.py -v -m _example -n net_example -d _example -tid 0 -vid 0 -g 1"


## 支持的数据集（通过 --dataset 或 -d 指定）

> 2D: mvtec2d, mpdd, mvtecloco, mtd, btad, mvtec2df3d, coad

> 3D: mvtec3d

示例目录结构（以 mvtec2d 为例）：
```bash
.
├── bottle
│   ├── ground_truth
│   │   ├── broken_large
│   │   │   ├── 000_mask.png
│   │   │   ├── 001_mask.png
│   │   │   ├── ...
│   ├── test
│   │   ├── broken_large
│   │   │   ├── 000.png
│   │   │   ├── 001.png
│   │   │   ├── ...
│   │   └── good
│   │       ├── 000.png
│   │       ├── 001.png
│   │       ├── ...
│   └── train
│       └── good
│           ├── 000.png
│           ├── 001.png
│           ├── ...
├── cable
├── screw
└── ...
```
## 学习范式（Learning Paradigm）

|| Paradigm 名称   | Marker | 训练数据                        | 测试数据                           | 解释            |
|----|------------- | ------ | ------------------------------------------------- | ------------------------------ | ---------------------- |
| $\bigstar$ | *centralized 2d* | -p c2d | |  |
| | **vanilla**   | `-v`   | 所有 `id=0` 图像（通常只取正常图像）                            | 所有 `id=0` 图像（含正常和异常）           | 无监督训练，只学习正常图，测试中用于检测异常 |
| | **semi**      | `-s`   | 所有正常图像 + 少量标注异常图（都为 `id=0`）                       | 所有 `id=0` 测试图像，但剔除训练中已出现的异常图   | 半监督训练，适合研究异常标签利用效率     |
| | **fewshot**   | `-f`   | `id=0` 类中抽取少量正常图进行训练                              | 使用完整 `id=0` 类测试集               | 小样本无监督设定，用少量图像学习正常模式   |
| | **continual** | `-c`   | 同时使用 `id=0` 和 `id=1` 类的完整数据                       | 使用 `id=0` 和 `id=1` 的测试集        | 持续学习设定：学习新类不遗忘旧类       |
| | **noisy**     | `-z`   | `id=0` 中的正常图 + 伪异常图（即 noisy 数据）                   | `id=0` 中的测试图，但不含训练中用过的 noisy 图 | 模拟异常检测中的伪标签干扰或缺陷混淆情形   |
| | **transfer**  | `-t`   | Step 1：完整训练 `id=0` 类<br>Step 2：对 `id=1` 类少量数据进行微调 | 分别在 `id=0` 和 `id=1` 上评估        | 跨类迁移学习，适合研究模型的泛化能力     |

| $\bigstar$ | *centralized 3d* | -p c3d | To be updated! | |  |
| $\bigstar$ | *federated 2d* | -p f2d |  To be updated! | | |


## 2D Model
| No. | Method / -m | Net / -n | Paper Title|
| ------ | ------ | ------ | ------ |
| 1 | cfa | net_cfa | CFA: Coupled-hypersphere-based feature adaptation for target-oriented anomaly localization |
| 2 | csflow | net_csflow | Fully convolutional cross-scale-flows for image-based defect detection |
| 3 | cutpaste | vit_b_16 | Cutpaste: self-supervised learning for anomaly detection and localization |
| 4 | devnet | net_devnet | Explainable deep few-shot anomaly detection with deviation networks |
| 5 | dne | vit_b_16 | Towards continual adaptation in industrial anomaly detection |
| 6 | dra | net_dra | Catching both gray and black swans: open-set supervised anomaly detection |
| 7 | fastflow | net_fastflow | Fastflow: unsupervised anomaly detection and localization via 2d normalizing flows |
| 8 | favae | net_favae | Anomaly localization by modeling perceptual features |
| 9 | igd | net_igd | Deep one-class classification via interpolated gaussian descriptor |
| 10 | padim  | resnet18, wide_resnet50 | Padim: a patch distribution modeling framework for anomaly detection and localization |
| 11 | patchcore  | resnet18, wide_resnet50 | Towards total recall in industrial anomaly detection |
| 12 | reverse | net_reverse | Anomaly detection via reverse distillation from one-class embedding |
| 13 | simplenet  | wide_resnet50 | SimpleNet: a simple network for image anomaly detection and localization |
| 14 | softpatch  | resnet18, wide_resnet50 | SoftPatch: unsupervised anomaly detection with noisy data |
| 15 | spade  | resnet18, wide_resnet50 | Sub-image anomaly detection with deep pyramid correspondences |
| 16 | stpm  | resnet18, wide_resnet50 | Student-teacher feature pyramid matching for anomaly detection |

<!-- ## 3D Model
| No. | Method / -m | Net / -n | Paper Title |
| ------ | ------ | ------ | ------ |
| 1 | 3d_btf | -- | Back to the feature: classical 3D features are (almost) all you need for 3D anomaly detection | 
| 2 | 3d_ast | | AST: Asymmetric student-teacher networks for industrial anomaly detection | -->


## Run Example

> Vanilla / -v
```bash
python3 main.py -v -m cfa -n net_cfa -d mvtec2d -tid 0 -vid 0 -g 1
python3 main.py -v -m csflow -n net_csflow -d mvtec2d -tid 0 -vid 0 -g 1
python3 main.py -v -m cutpaste -n vit_b_16  -d mvtec2d -tid 0 -vid 0  -g 1
python3 main.py -v -m fastflow -n net_fastflow -d mvtec2d -tid 0 -vid 0 -g 1
python3 main.py -v -m favae -n net_favae -d mvtec2d -tid 0 -vid 0  -g 1
# python3 main.py -v -m graphcore -n vig_ti_224_gelu -d mvtec2d -tid 0 -vid 0 -sp 0.001 -g 1
python3 main.py -v -m igd -n net_igd -d mvtec2d -tid 0 -vid 0 -g 1
python3 main.py -v -m padim -n resnet18 -d mvtec2d -tid 0 -vid 0 -g 1
python3 main.py -v -m cutpaste -n vit_b_16 -d mvtec2d -tid 0 -vid 0 -g 1
python3 main.py -v -m patchcore -n wide_resnet50 -d mvtec2d -tid 0 -vid 0 -sp 0.001 -g 1
python3 main.py -v -m reverse -n net_reverse -d mvtec2d -tid 0 -vid 0 -g 1
python3 main.py -v -m simplenet -n wide_resnet50 -d mvtec2d -tid 0 -vid 0 -g 1
python3 main.py -v -m spade -n resnet18 -d mvtec2d -tid 0 -vid 0 -g 1
python3 main.py -v -m stpm -n resnet18 -d mvtec2d -tid 0 -vid 0 -g 1
```

> Semi / -s
```bash
python3 main.py -s -m devnet -n net_devnet -d mvtec2d -tid 0 -vid 0 -g 1
python3 main.py -s -m dra -n net_dra -d mvtec2d -tid 0 -vid 0 -g 1
```

> Fewshot / -f
```bash
python3 main.py -f -fe 1 -m patchcore -n wide_resnet50 -d mvtec2d -tid 0 -vid 0 -sp 0.1 -g 1 -fda -fnd 4 -fat rotation
python3 main.py -f -fe 1 -m _patchcore -n wide_resnet50 -d mvtec2d -tid 0 -vid 0 -sp 1 -fda -fnd 4 -g 1
python3 main.py -f -fe 1 -m csflow -n net_csflow -d mvtec2d -tid 0 -vid 0 -g 1
python3 main.py -f -fe 1 -m cfa -n net_cfa -d mvtec2d -tid 0 -vid 0 -g 1
python3 main.py -f -fe 1 -m fastflow -n net_fastflow -d mvtec2d -tid 0 -vid 0 -g 1
python3 main.py -f -fe 1 -m cutpaste -n vit_b_16  -d mvtec2d -tid 0 -vid 0  -g 1
python3 main.py -f -fe 1 -m padim -n resnet18 -d mvtec2d -tid 0 -vid 0 -g 1
python3 main.py -f -fe 1 -m favae -n net_favae -d mvtec2d -tid 0 -vid 0  -g 1
python3 main.py -f -fe 1 -m cutpaste -n vit_b_16 -d mvtec2d -tid 0 -vid 0 -g 1
python3 main.py -f -fe 1 -m igd -n net_igd -d mvtec2d -tid 0 -vid 0 -g 1
python3 main.py -f -fe 1 -m reverse -n net_reverse -d mvtec2d -tid 0 -vid 0 -g 1
python3 main.py -f -fe 1 -m spade -n resnet18 -d mvtec2d -tid 0 -vid 0 -g 1
python3 main.py -f -fe 1 -m stpm -n resnet18 -d mvtec2d -tid 0 -vid 0 -g 1
python3 main.py -f -fe 1 -m simplenet -n wide_resnet50 -d mvtec2d -tid 0 -vid 0 -g 1
```

> Continual / -c
```bash
python3 main.py -c -m patchcore -n wide_resnet50 -d mvtec2d -tid 0 1 -vid 0 1 -sp 0.001 -g 1
python3 main.py -c -m csflow -n net_csflow -d mvtec2d -tid 0 1 -vid 0 1 -g 1
python3 main.py -c -m cfa -n net_cfa -d mvtec2d -tid 0 1 -vid 0 1 -g 1
python3 main.py -c -m fastflow -n net_fastflow -d mvtec2d -tid 0 1 -vid 0 1 -g 1
python3 main.py -c -m cutpaste -n vit_b_16  -d mvtec2d -tid 0 1 -vid 0 1  -g 1
python3 main.py -c -m padim -n resnet18 -d mvtec2d -tid 0 1 -vid 0 1 -g 1
python3 main.py -c -m favae -n net_favae -d mvtec2d -tid 0 1 -vid 0 1  -g 1
python3 main.py -c -m cutpaste -n vit_b_16 -d mvtec2d -tid 0 1 -vid 0 1 -g 1
python3 main.py -c -m igd -n net_igd -d mvtec2d -tid 0 1 -vid 0 1 -g 1
python3 main.py -c -m reverse -n net_reverse -d mvtec2d -tid 0 1 -vid 0 1 -g 1
python3 main.py -c -m spade -n resnet18 -d mvtec2d -tid 0 1 -vid 0 1 -g 1
python3 main.py -c -m stpm -n resnet18 -d mvtec2d -tid 0 1 -vid 0 1 -g 1
python3 main.py -c -m dne -n vit_b_16 -d mvtec2d -tid 0 1 -vid 0 1 -g 1
python3 main.py -c -m simplenet -n wide_resnet50 -d mvtec2d -tid 0 1 -vid 0 1 -g 1
```

> Noisy / -z
```bash
python3 main.py -z -nr 0.1 -no -m softpatch -n wide_resnet50  -d mvtec2d -tid 0 -vid 0 -sp 0.001 -g 1
python3 main.py -z -nr 0.1 -no -m patchcore -n wide_resnet50  -d mvtec2d  -tid 0 -vid 0 -sp 0.001 -g 1
python3 main.py -z -nr 0.1 -no -m csflow -n net_csflow -d mvtec2d -tid 0 -vid 0 -g 1
python3 main.py -z -nr 0.1 -no -m cfa -n net_cfa -d mvtec2d -tid 0 -vid 0 -g 1
python3 main.py -z -nr 0.1 -no -m fastflow -n net_fastflow -d mvtec2d -tid 0 -vid 0 -g 1
python3 main.py -z -nr 0.1 -no -m cutpaste -n vit_b_16  -d mvtec2d -tid 0 -vid 0  -g 1
python3 main.py -z -nr 0.1 -no -m padim -n resnet18 -d mvtec2d -tid 0 -vid 0 -g 1
python3 main.py -z -nr 0.1 -no -m favae -n net_favae -d mvtec2d -tid 0 -vid 0  -g 1
python3 main.py -z -nr 0.1 -no -m cutpaste -n vit_b_16 -d mvtec2d -tid 0 -vid 0 -g 1
python3 main.py -z -nr 0.1 -no -m reverse -n net_reverse -d mvtec2d -tid 0 -vid 0 -g 1
python3 main.py -z -nr 0.1 -no -m spade -n resnet18 -d mvtec2d -tid 0 -vid 0 -g 1
python3 main.py -z -nr 0.1 -no -m stpm -n resnet18 -d mvtec2d -tid 0 -vid 0 -g 1
python3 main.py -z -nr 0.1 -no -m igd -n net_igd -d mvtec2d -tid 0 -vid 0 -g 1
python3 main.py -z -nr 0.1 -no -m simplenet -n wide_resnet50  -d mvtec2d -tid 0 -vid 0 -g 1
```

> Transfer / -t
```bash
python3 main.py -t -ttn 8 -m reverse -n net_reverse -d coad -tid 0 -vid 1 -g 1 -ne 10
python3 main.py -t -ttn 8 -m cfa -n net_cfa -d coad -tid 0 -vid 1 -g 1 -ne 10
python3 main.py -t -ttn 8 -m csflow -n net_csflow -d coad -tid 0 -vid 1 -g 1 -ne 10
python3 main.py -t -ttn 8 -m fastflow -n net_fastflow -d coad -tid 0 -vid 1 -g 1 -ne 10
python3 main.py -t -ttn 8 -m favae -n net_favae -d coad -tid 0 -vid 1 -g 1 -ne 10
python3 main.py -t -ttn 8 -m padim -n resnet18 -d coad -tid 0 -vid 1 -g 1
python3 main.py -t -ttn 8 -m patchcore -n wide_resnet50 -d coad -tid 0 -vid 1 -g 1
python3 main.py -t -ttn 8 -m stpm -n resnet18 -d coad -tid 0 -vid 1 -g 1
python3 main.py -t -ttn 8 -m simplenet -n wide_resnet50 -d coad -tid 0 -vid 1 -g 1
```
