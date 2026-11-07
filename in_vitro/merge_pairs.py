"""Merge paired FASTQ reads with the fastp settings used for Beezy."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess


def run(r1, r2, output_directory, threads=8, fastp="fastp"):
    r1, r2 = Path(r1), Path(r2)
    if not r1.is_file() or not r2.is_file():
        raise ValueError("Both input FASTQ files must exist.")
    if threads < 1:
        raise ValueError("THREADS must be a positive integer.")
    executable = shutil.which(fastp)
    if executable is None:
        raise FileNotFoundError(
            "fastp was not found. Install fastp 0.24.0 or use --fastp to specify its executable.")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "fastp_version.txt").open("w") as log:
        subprocess.run([executable, "--version"], stdout=log, stderr=subprocess.STDOUT, check=True)
    command = [executable, "-i", str(r1), "-I", str(r2),
               "--merge", "--merged_out", str(output / "merged.fastq"),
               "--html", str(output / "fastp.html"), "--json", str(output / "fastp.json"),
               "-w", str(threads)]
    with (output / "fastp.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(f"Merged reads written to: {output / 'merged.fastq'}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("r1", help="R1 FASTQ file")
    parser.add_argument("r2", help="R2 FASTQ file")
    parser.add_argument("output_directory", help="New output directory, for example runs/in_vitro")
    parser.add_argument("threads", nargs="?", type=int, default=8)
    parser.add_argument("--fastp", default=os.environ.get("FASTP_BIN", "fastp"),
                        help="fastp executable (default: FASTP_BIN or fastp on PATH)")
    args = parser.parse_args()
    try:
        run(args.r1, args.r2, args.output_directory, args.threads, args.fastp)
    except FileNotFoundError as error:
        parser.exit(127, f"{error}\n")
    except (OSError, ValueError) as error:
        parser.exit(2, f"{error}\n")
    except subprocess.CalledProcessError as error:
        parser.exit(error.returncode, f"fastp exited with code {error.returncode}. See the output log.\n")


if __name__ == "__main__":
    main()
