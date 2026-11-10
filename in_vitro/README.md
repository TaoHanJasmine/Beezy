# In vitro sequencing

The workflow uses fastp 0.24.0 to merge paired reads and Cutadapt 5.2 to remove library boundaries and orient the inserts. The resulting text file can be passed to `main.py decode`.

| Step | Entry point | Output |
| --- | --- | --- |
| 1. Merge R1/R2 | [merge_pairs.py](merge_pairs.py) | `runs/in_vitro/merged.fastq` |
| 2. Trim and orient | [trim_cutadapt.py](trim_cutadapt.py) | `runs/in_vitro/trimmed/processed_reads.txt` |

## Input data

The original image encoded for the experiment is [tju.bmp](tju.bmp), a 435 × 552 pixel BMP file of 960,534 bytes. Its SHA-256 checksum matches the source image recorded in the experiment's encoding manifest:

```text
8e1a8cab9e33cb2d1b5a3ff369bdc12ee807ae065149321483d34117d57887da
```

Use this file as the reference when evaluating recovery. From the repository root, replace `RECOVERED_BMP` with the path to the decoded BMP file:

```bash
"$BEEZY_PYTHON" main.py evaluate --original in_vitro/tju.bmp --recovered "$RECOVERED_BMP"
```

Download the paired FASTQ files from [Beezy in vitro sequencing data](https://doi.org/10.6084/m9.figshare.34700607.v1). Each mate file contains the same set of read pairs.

| Mate | File | MD5 |
| --- | --- | --- |
| R1 | `JZ26226409-OPS-HDM-26090102-QC-O-2458_combined_R1.fastq.gz` | `6f41777436a39917a78ff95c173e2ec4` |
| R2 | `JZ26226409-OPS-HDM-26090102-QC-O-2458_combined_R2.fastq.gz` | `2e71f9863113ce5b7f519cf509a19561` |

## 1. Install the tools

Use the dedicated Beezy environment described in the main README. Install [fastp 0.24.0](https://github.com/OpenGene/fastp/releases/tag/v0.24.0) and the Python dependencies:

```bash
conda install -n beezy -c conda-forge -c bioconda fastp=0.24.0
"$BEEZY_PYTHON" -m pip install -r in_vitro/requirements.txt
```

## 2. Merge the paired reads

Run from the repository root and set `R1` and `R2` to the downloaded files. The script calls fastp with the defaults from the original merge and records its version and log. The output directory must be new.

```bash
R1=/path/to/JZ26226409-OPS-HDM-26090102-QC-O-2458_combined_R1.fastq.gz
R2=/path/to/JZ26226409-OPS-HDM-26090102-QC-O-2458_combined_R2.fastq.gz
"$BEEZY_PYTHON" -m in_vitro.merge_pairs "$R1" "$R2" runs/in_vitro 8
```

This creates `runs/in_vitro/merged.fastq`, `fastp.html`, `fastp.json`, `fastp.log`, and `fastp_version.txt`. Use `--fastp /path/to/fastp` to specify the executable when it is not on `PATH`.

## 3. Trim and export

```bash
"$BEEZY_PYTHON" -m in_vitro.trim_cutadapt in_vitro/cutadapt_config.json
```

The [configuration](cutadapt_config.json) specifies the library boundaries:

| Parameter | Value |
| --- | --- |
| Left boundary | `TTCTGAGCGAATAGGAGTCCTCGCACCAGCGTGT` (34 nt) |
| Right boundary | `CACTGCGGCTCCTCA` (15 nt) |
| Maximum adapter error rate | 0.1 |
| Minimum adapter overlap | Full boundary length, 34 / 15 nt |
| Orientation | Forward and reverse complement |

Cutadapt requires both linked boundaries, allows mismatches and indels, and keeps the intervening insert. Read multiplicity is preserved, including repeated UMIs. Empty inserts and inserts containing `N` are filtered out; variable insert lengths are retained.

Paths in the configuration are relative to the configuration file. Change `merged_input` and `output_directory` for another input. Use a new trimming output directory for each run. An optional `expected_merged_reads` field checks the input count against an independently known value.

## Output

Files are written to `runs/in_vitro/trimmed/`:

| File | Content |
| --- | --- |
| `processed_reads.txt` | Decoder input, one sequence per line |
| `inserts.fastq.gz` | Trimmed inserts with read IDs and quality scores |
| `untrimmed.fastq.gz` | Reads without the required boundaries |
| `empty.fastq.gz` | Empty trimmed inserts |
| `cutadapt.json`, `cutadapt.log` | Cutadapt counts and execution log |
| `preprocessing_summary.json` | Command, version, counts, and insert length distribution |

Use `processed_reads.txt` as `--reads` with the tree seed corresponding to that encoded library. This wrapper reuses Cutadapt for trimming and exports its accepted reads in the decoder's text format. See the [Cutadapt linked-adapter documentation](https://cutadapt.readthedocs.io/en/v5.2/guide.html#linked-adapters-combined-5-and-3-adapter) for matching details.
