# Beezy

Beezy 是一种面向链顺序恢复的混合 DNA 存储架构。它将基于序列特征的树序排名、短链头和独立保存在计算机端的树种子（tree seed）结合起来。解码器利用树种子恢复链顺序，再重建原始文件。纠错部分结合了受约束 trellis 编码、Log-MAP BCJR 解码和纵向 Reed–Solomon 编码。

本仓库提供编解码算法、体外实验数据处理、文件恢复示例和评估函数。[English](README.md)

## 安装

已在 Linux、Python 3.11.15 环境中测试。从仓库根目录执行以下命令，在独立环境中安装依赖：

```bash
conda create -n beezy python=3.11 pip
conda activate beezy
BEEZY_PYTHON="$CONDA_PREFIX/bin/python"
"$BEEZY_PYTHON" -m pip install -r requirements-dev.txt
```

如果已有独立环境，将 `BEEZY_PYTHON` 设置为该环境的 Python 解释器路径即可。[requirements.txt](requirements.txt) 列出运行依赖，[requirements-dev.txt](requirements-dev.txt) 还包含构建与测试依赖。

NumPy 后端无需编译扩展即可运行。如需加快解码，可在具备 C 编译器的环境中编译随包提供的 Cython 源码：

```bash
"$BEEZY_PYTHON" setup.py build_ext --inplace
```

解码器在编译扩展可用时自动使用扩展。可设置 `BEEZY_BCJR_BACKEND=numpy` 或 `BEEZY_BCJR_BACKEND=cython`，明确选择后端。

## 一个完整示例

从仓库根目录执行以下命令。示例将 [examples/message.txt](examples/message.txt) 编码为 DNA，在平均深度 30 下模拟无噪声读段，再使用一个工作进程恢复文件。

```bash
mkdir -p .example-output

"$BEEZY_PYTHON" main.py encode \
  --input examples/message.txt \
  --name .example-output/demo

"$BEEZY_PYTHON" main.py simulate \
  --name .example-output/demo \
  --depth 30 --random-seed 20260801 \
  --substitution 0 --deletion 0 --insertion 0 --dropout 0

"$BEEZY_PYTHON" main.py decode \
  --reads .example-output/demo_reads \
  --tree-seed .example-output/demo_seed \
  --output .example-output/restored.txt \
  --workers 1 \
  --substitution 0 --deletion 0 --insertion 0

"$BEEZY_PYTHON" main.py evaluate \
  --original examples/message.txt \
  --recovered .example-output/restored.txt
```

恢复成功时输出 `File hashes match.`。以下文件保存在 `.example-output/` 中：

| 文件 | 含义 |
| --- | --- |
| `demo_encoded_strands.txt` | 编码后的 DNA 链池，每行一条链 |
| `demo_seed` | JSON 格式的计算机端树种子和文件长度元数据 |
| `demo_reads` | 模拟读段，保留重复读段 |
| `restored.txt` | 恢复后的文件 |

使用不同的输出前缀，可以分别保留各次运行的结果。

## 默认链结构与解码输入

| 设置 | 默认值 |
| --- | --- |
| 数据区（payload） | 每条数据链 44 nt，对应 11 字节 |
| 用于树序排名的特征区间 | 起点为 0，长度为 44 nt |
| 链头（header） | 4 nt |
| RS 校验链数 | `ceil(数据链数 × 0.10)` |
| 编码后的插入片段 | 100 nt，包含 trellis 终止部分 |
| 文件白化 | 默认启用，文件重建时执行逆变换 |
| BCJR 漂移窗口 | 3 |
| 解码工作进程数 | 24；示例中显式设为 1 |

解码输入为每行一条预处理读段的文本文件，以及与其匹配的计算机端树种子。请保留重复读段，解码器会利用重复次数恢复链顺序。排序后，流程使用硬判决 Reed–Solomon 解码恢复数据。

通过 `--substitution`、`--deletion`、`--insertion` 设置适用的 BCJR 信道概率。修改链结构时，编码和解码应提供相互匹配的 `--payload-length`、`--tree-start`、`--tree-length` 和 `--parity-strands`。非默认参数会记录到 `*_encoding_parameters.json` 供查阅，解码时仍需显式传入。全部选项可通过 `main.py encode --help` 和 `main.py decode --help` 查看。

## 体外实验

原始双端测序数据可在 [figshare](https://doi.org/10.6084/m9.figshare.34700607.v1) 获取。

[in_vitro/](in_vitro/README.zh-CN.md) 包含两个处理步骤：

1. [merge_pairs.py](in_vitro/merge_pairs.py)：调用 fastp 合并 R1/R2 双端 FASTQ 读段。
2. [trim_cutadapt.py](in_vitro/trim_cutadapt.py)：去除两端固定序列、统一方向，导出每行一条序列的文本，供 `main.py decode` 使用。

输出统一保存在 `runs/in_vitro/`，裁剪结果位于 `runs/in_vitro/trimmed/`。

## 评估与评分规则

| 指标 | 定义 | 函数 |
| --- | --- | --- |
| 文件完整恢复 | 原文件与恢复文件的 SHA-256 一致 | [`compare_file_hashes`](evaluation/metrics.py) |
| 字节恢复率 | 固定位置上正确恢复的字节数 / 原始字节数 | [`score_bytes`](evaluation/recovery_metrics.py) |
| 覆盖率 | 已分配的位置数 / 参考位置总数 | [`score_indices`](evaluation/recovery_metrics.py) |
| 排序准确率 | 已分配候选中来源位置与目标位置一致的读段数 / 已分配候选的全部支持读段数 | [`score_indices`](evaluation/recovery_metrics.py) |

计算文件数据恢复率时，传入数据链的数据区及 `original_byte_count`，排除 RS 校验链和末尾填充。每 4 个碱基对应 1 字节，缺失或无效的碱基组计为未恢复；同时说明评估的是 RS 解码前还是解码后的序列。

`main.py evaluate` 用于比较文件哈希，其余指标通过 Python API 计算。排序准确率需要候选序列的读段来源计数，并应与覆盖率同时报告。没有任何分配时，覆盖率为 0，排序准确率为 `None`。评分真值与解码输入分开提供。

## 测试

```bash
"$BEEZY_PYTHON" -m pytest -q
```

测试覆盖编解码、排序、读段频次、树种子打包、缓存和评分规则。安装 [in_vitro/requirements.txt](in_vitro/requirements.txt) 后，也会运行 Cutadapt 测试。

## 目录结构

```text
Beezy/
├── main.py                  # 命令行入口
├── pipeline/                # 文件编码与恢复
├── indexing/                # 树序排名、链头和链顺序恢复
├── ecc/                     # Trellis、BCJR 和 Reed–Solomon 实现
├── simulation/              # 读段采样与测序错误信道
├── in_vitro/                # 体外实验：双端合并、裁剪与导出
├── evaluation/              # 文件与池级评分
├── examples/message.txt     # 一个小型输入文件
├── tests/                   # 算法与指标测试
├── requirements.txt         # 运行依赖
├── requirements-dev.txt     # 构建与测试依赖
└── setup.py                 # 可选的 Cython 扩展构建
```

## 许可证

Beezy 代码采用 [MIT 许可证](LICENSE)。链接中的测序数据集采用 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 许可。
