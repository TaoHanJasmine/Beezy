import json

import pytest

pytest.importorskip("dnaio", reason="Install in_vitro/requirements.txt to test preprocessing")
pytest.importorskip("cutadapt", reason="Install in_vitro/requirements.txt to test preprocessing")

from ..in_vitro.trim_cutadapt import run


def make_input(tmp_path, expected=None):
    left = "TTCTGAGCGAATAGGAGTCCTCGCACCAGCGTGT"
    right = "CACTGCGGCTCCTCA"
    insert = "AGTC" * 25
    full = "ACGTACGTACGT" + left + insert + right + "C" * 11
    reverse = full.translate(str.maketrans("ACGT", "TGCA"))[::-1]
    sequences = [full, full, reverse,
                 "ACGTACGTACGT" + left[:5] + "C" + left[6:] + insert + right + "CCCCC",
                 "ACGTACGTACGT" + left + insert[:-1] + right + "CCCCC",
                 "ACGTACGTACGT" + left + insert,
                 "ACGTACGTACGT" + left + "AN" + right + "CCCCC",
                 "ACGTACGTACGT" + left + right + "CCCCC"]
    (tmp_path / "input.fastq").write_text("".join(
        f"@r{i}\n{s}\n+\n{'I' * len(s)}\n" for i, s in enumerate(sequences)))
    config = {"merged_input": "input.fastq", "output_directory": "out",
              "left_boundary": left, "right_boundary": right,
              "threads": 1, "max_error_rate": 0.1}
    if expected is not None:
        config["expected_merged_reads"] = expected
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path, insert


@pytest.mark.parametrize("expected", [None, 8])
def test_linked_trimming_and_decoder_export(tmp_path, expected):
    config, insert = make_input(tmp_path, expected)
    run(config)
    output = (tmp_path / "out/processed_reads.txt").read_text().splitlines()
    assert output == [insert, insert, insert, insert, insert[:-1]]
    summary = json.loads((tmp_path / "out/preprocessing_summary.json").read_text())
    assert summary["status"] == "complete"
    assert summary["input_reads"] == 8
    assert summary["accepted_reads"] == 5
    assert summary["cutadapt_counts"]["filtered"]["discard_untrimmed"] == 1


def test_count_mismatch_records_failure(tmp_path):
    config, _ = make_input(tmp_path, 9)
    with pytest.raises(ValueError, match="Expected 9 merged reads, found 8"):
        run(config)
    summary = json.loads((tmp_path / "out/preprocessing_summary.json").read_text())
    assert summary["status"] == "failed"
    assert not (tmp_path / "out/processed_reads.txt").exists()


def test_existing_output_is_preserved(tmp_path):
    config, _ = make_input(tmp_path)
    (tmp_path / "out").mkdir()
    marker = tmp_path / "out/keep.txt"
    marker.write_text("keep")
    with pytest.raises(FileExistsError):
        run(config)
    assert marker.read_text() == "keep"
