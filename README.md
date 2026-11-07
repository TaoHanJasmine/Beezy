# Beezy

Beezy is a hybrid DNA storage architecture for strand order recovery. It combines sequence-derived tree ranks with short strand headers and a separately stored digital tree seed. The decoder uses this seed to recover strand order before reconstructing the source file. Error correction combines constrained trellis coding, Log-MAP BCJR decoding, and vertical Reed–Solomon coding.

The repository includes the codec, in vitro sequencing tools, a file recovery example, and evaluation functions.

## Installation

Tested with Python 3.11.15 on Linux. From the repository root, install the dependencies in a dedicated environment:

```bash
conda create -n beezy python=3.11 pip
conda activate beezy
BEEZY_PYTHON="$CONDA_PREFIX/bin/python"
"$BEEZY_PYTHON" -m pip install -r requirements-dev.txt
```

For an existing environment, set `BEEZY_PYTHON` to its Python executable. Runtime dependencies are listed in [requirements.txt](requirements.txt); [requirements-dev.txt](requirements-dev.txt) also includes build and test dependencies.

The NumPy backend works without compiling an extension. For faster decoding, build the bundled Cython source with a C compiler available:

```bash
"$BEEZY_PYTHON" setup.py build_ext --inplace
```

The decoder automatically selects the compiled extension when available. Set `BEEZY_BCJR_BACKEND=numpy` or `BEEZY_BCJR_BACKEND=cython` to select a backend explicitly.

## One complete example

Run the following commands from the repository root. The example encodes [examples/message.txt](examples/message.txt), simulates noiseless reads at mean depth 30, and recovers the file using one worker.

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

Successful recovery prints `File hashes match.`. Outputs are written to `.example-output/`:

| File | Meaning |
| --- | --- |
| `demo_encoded_strands.txt` | Encoded DNA pool, one strand per line |
| `demo_seed` | Digital tree seed and file length metadata in JSON format |
| `demo_reads` | Simulated reads, with repeated reads retained |
| `restored.txt` | Reconstructed file |

Use a different output prefix to keep results from separate runs.

## Default layout and decoding inputs

| Setting | Default |
| --- | --- |
| Payload | 44 nt (11 bytes) per data strand |
| Feature interval for tree ranking | Start 0, length 44 nt |
| Header | 4 nt |
| RS parity strands | `ceil(number of data strands × 0.10)` |
| Encoded insert | 100 nt, including trellis termination |
| File whitening | Enabled; reversed during file reconstruction |
| BCJR drift window | 3 |
| Decoder workers | 24; the example explicitly uses 1 |

Decoding takes a text file containing one processed read per line and the matching digital tree seed. Keep duplicate reads: their counts are used for order recovery. The pipeline applies hard-decision Reed–Solomon decoding after ordering.

Set `--substitution`, `--deletion`, and `--insertion` to the applicable BCJR channel probabilities. When changing the strand layout, provide matching `--payload-length`, `--tree-start`, `--tree-length`, and `--parity-strands` to encoding and decoding. Nondefault settings are recorded in `*_encoding_parameters.json` for reference and must be passed explicitly to decoding. Full options are available through `main.py encode --help` and `main.py decode --help`.

## In vitro sequencing

Raw paired-end sequencing data are available on [figshare](https://doi.org/10.6084/m9.figshare.34700607.v1).

[in_vitro/](in_vitro/README.md) contains two processing steps:

1. [merge_pairs.py](in_vitro/merge_pairs.py) merges R1/R2 FASTQ reads with fastp.
2. [trim_cutadapt.py](in_vitro/trim_cutadapt.py) removes flanking sequences, normalizes orientation, and exports one sequence per line for `main.py decode`.

Outputs are written to `runs/in_vitro/`, with trimmed reads in `runs/in_vitro/trimmed/`.

## Evaluation and scoring rules

| Metric | Definition | Function |
| --- | --- | --- |
| Exact file recovery | SHA-256 digests of the original and recovered files match | [`compare_file_hashes`](evaluation/metrics.py) |
| Byte recovery rate | Correct bytes at fixed positions / original byte count | [`score_bytes`](evaluation/recovery_metrics.py) |
| Coverage | Assigned positions / reference positions | [`score_indices`](evaluation/recovery_metrics.py) |
| Ordering accuracy | Reads with matching source and assigned positions / all supporting reads in assigned candidates | [`score_indices`](evaluation/recovery_metrics.py) |

For file data recovery, supply the data payloads and `original_byte_count` to exclude RS parity and trailing padding. Each four-base group represents one byte; missing or invalid groups count as unrecovered. State whether the scored payloads are before or after RS decoding.

`main.py evaluate` compares file hashes; the other metrics are Python APIs. Ordering accuracy requires candidate source-read counts and should be reported with coverage. With no assignments, coverage is zero and ordering accuracy is `None`. Evaluation truth is supplied separately from decoding inputs.

## Tests

```bash
"$BEEZY_PYTHON" -m pytest -q
```

Tests cover the codec, ordering, read frequencies, tree seed packing, caching, and scoring. Install [in_vitro/requirements.txt](in_vitro/requirements.txt) to include the Cutadapt tests.

## Directory layout

```text
Beezy/
├── main.py                  # Command-line entry
├── pipeline/                # File encoding and recovery
├── indexing/                # Tree ranks, headers, and order recovery
├── ecc/                     # Trellis, BCJR, and Reed–Solomon implementations
├── simulation/              # Read sampling and sequencing error channel
├── in_vitro/                # In vitro sequencing: merge, trim, and export
├── evaluation/              # File and pool-level scoring
├── examples/message.txt     # One small input file
├── tests/                   # Algorithm and metric tests
├── requirements.txt         # Runtime dependencies
├── requirements-dev.txt     # Build and test dependencies
└── setup.py                 # Optional Cython extension build
```

## License

Beezy is released under the [MIT License](LICENSE). The linked sequencing dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
