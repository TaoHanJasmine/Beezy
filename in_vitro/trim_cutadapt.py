"""Run Cutadapt linked-adapter trimming and export reads for Beezy decoding."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import dnaio


def command(config, source, output):
    left, right = config["left_boundary"], config["right_boundary"]
    linked = f"{left};min_overlap={len(left)}...{right};min_overlap={len(right)}"
    return [sys.executable, "-m", "cutadapt", "-j", str(config["threads"]),
            "-g", linked, "-e", str(config["max_error_rate"]), "--revcomp",
            "--minimum-length", "1", "--max-n", "0",
            "--untrimmed-output", str(output / "untrimmed.fastq.gz"),
            "--too-short-output", str(output / "empty.fastq.gz"),
            "--json", str(output / "cutadapt.json"),
            "-o", str(output / "inserts.fastq.gz"), str(source)]


def run(config_path):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    source = (config_path.parent / config["merged_input"]).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    output = (config_path.parent / config["output_directory"]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    cmd = command(config, source, output)
    metadata = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
                "command": cmd, "config": config, "source": str(source)}

    def save():
        (output / "preprocessing_summary.json").write_text(json.dumps(metadata, indent=2) + "\n")

    save()
    try:
        metadata["cutadapt_version"] = subprocess.check_output(
            [sys.executable, "-m", "cutadapt", "--version"], text=True).strip()
        with (output / "cutadapt.log").open("w") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
        report = json.loads((output / "cutadapt.json").read_text())
        input_count = report["read_counts"]["input"]
        expected = config.get("expected_merged_reads")
        if expected is not None and input_count != expected:
            raise ValueError(f"Expected {expected} merged reads, found {input_count}")
        lengths = Counter()
        with dnaio.open(output / "inserts.fastq.gz") as reads, (output / "processed_reads.txt").open("w") as text:
            for read in reads:
                if not read.sequence or set(read.sequence) - set("ACGT"):
                    raise ValueError(f"Unexpected decoder alphabet in {read.name}")
                text.write(read.sequence + "\n")
                lengths[len(read.sequence)] += 1
        count = sum(lengths.values())
        filtered = report["read_counts"]["filtered"]
        if count != report["read_counts"]["output"]:
            raise ValueError("FASTQ and decoder-text output counts disagree")
        if count + sum(v for v in filtered.values() if v is not None) != input_count:
            raise ValueError("Accepted and filtered counts do not match the input")
        metadata.update(status="complete", input_reads=input_count, accepted_reads=count,
                        length_histogram=dict(sorted(lengths.items())),
                        cutadapt_counts=report["read_counts"],
                        finished_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        metadata.update(status="failed", error=str(error))
        raise
    finally:
        save()
    print(f"Processed reads written to: {output / 'processed_reads.txt'}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="JSON configuration; paths are relative to this file")
    run(parser.parse_args().config)
