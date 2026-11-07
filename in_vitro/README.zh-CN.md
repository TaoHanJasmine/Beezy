# 体外实验

流程使用 fastp 0.24.0 合并双端读段，再用 Cutadapt 5.2 去除文库边界并统一存储片段方向。生成的文本可作为 `main.py decode` 的输入。

| 步骤 | 入口 | 输出 |
| --- | --- | --- |
| 1. 合并 R1/R2 双端读段 | [merge_pairs.py](merge_pairs.py) | `runs/in_vitro/merged.fastq` |
| 2. 裁剪并统一方向 | [trim_cutadapt.py](trim_cutadapt.py) | `runs/in_vitro/trimmed/processed_reads.txt` |

## 输入数据

从 [Beezy in vitro sequencing data](https://doi.org/10.6084/m9.figshare.34700607.v1) 下载成对的 FASTQ 文件。两个文件包含相互配对的读段。

| 端 | 文件 | MD5 |
| --- | --- | --- |
| R1 | `JZ26226409-OPS-HDM-26090102-QC-O-2458_combined_R1.fastq.gz` | `6f41777436a39917a78ff95c173e2ec4` |
| R2 | `JZ26226409-OPS-HDM-26090102-QC-O-2458_combined_R2.fastq.gz` | `2e71f9863113ce5b7f519cf509a19561` |

## 1. 安装工具

使用主 README 中的 Beezy 独立环境，安装 [fastp 0.24.0](https://github.com/OpenGene/fastp/releases/tag/v0.24.0) 和 Python 依赖：

```bash
conda install -n beezy -c conda-forge -c bioconda fastp=0.24.0
"$BEEZY_PYTHON" -m pip install -r in_vitro/requirements.txt
```

## 2. 合并双端读段

从仓库根目录执行，将 `R1` 和 `R2` 设置为下载文件的路径。脚本沿用原始合并步骤中的 fastp 默认参数，并记录版本和运行日志。输出目录应为尚未创建的新目录。

```bash
R1=/path/to/JZ26226409-OPS-HDM-26090102-QC-O-2458_combined_R1.fastq.gz
R2=/path/to/JZ26226409-OPS-HDM-26090102-QC-O-2458_combined_R2.fastq.gz
"$BEEZY_PYTHON" -m in_vitro.merge_pairs "$R1" "$R2" runs/in_vitro 8
```

该步骤生成 `runs/in_vitro/merged.fastq`，以及 `fastp.html`、`fastp.json`、`fastp.log` 和 `fastp_version.txt`。如果 fastp 不在 `PATH` 中，可通过 `--fastp /path/to/fastp` 指定可执行文件路径。

## 3. 裁剪并导出

```bash
"$BEEZY_PYTHON" -m in_vitro.trim_cutadapt in_vitro/cutadapt_config.json
```

[配置文件](cutadapt_config.json) 指定了文库边界：

| 参数 | 设置 |
| --- | --- |
| 左边界 | `TTCTGAGCGAATAGGAGTCCTCGCACCAGCGTGT`（34 nt） |
| 右边界 | `CACTGCGGCTCCTCA`（15 nt） |
| 边界匹配最大错误率 | 0.1 |
| 边界最小重叠 | 完整边界长度，分别为 34 / 15 nt |
| 方向 | 正向及反向互补 |

Cutadapt 要求同时识别两侧关联边界，允许替换及插删，保留中间的存储片段。流程保留读段重复次数，包括相同 UMI 的读段；过滤空片段和含 `N` 的片段，保留不同长度的存储片段。

配置中的路径相对于配置文件所在目录解析。处理其他输入时，修改 `merged_input` 和 `output_directory`，每次使用新的裁剪输出目录。可选字段 `expected_merged_reads` 用于将实际输入条数与独立已知的条数进行核对。

## 输出

结果保存在 `runs/in_vitro/trimmed/`：

| 文件 | 内容 |
| --- | --- |
| `processed_reads.txt` | 解码输入，每行一条序列 |
| `inserts.fastq.gz` | 裁剪后的片段，保留读段 ID 和质量值 |
| `untrimmed.fastq.gz` | 未识别到所需边界的读段 |
| `empty.fastq.gz` | 裁剪后为空的片段 |
| `cutadapt.json`、`cutadapt.log` | Cutadapt 计数和运行日志 |
| `preprocessing_summary.json` | 命令、版本、读段计数和片段长度分布 |

将 `processed_reads.txt` 作为 `--reads`，配合该编码文库对应的 tree seed 进行解码。脚本调用 Cutadapt 完成裁剪，再将通过过滤的读段导出为解码器所需的文本格式。匹配规则详见 [Cutadapt 关联接头文档](https://cutadapt.readthedocs.io/en/v5.2/guide.html#linked-adapters-combined-5-and-3-adapter)。
