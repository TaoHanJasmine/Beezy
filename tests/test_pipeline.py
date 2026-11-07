"""Regression tests for the production encoding and recovery pipeline."""

from contextlib import redirect_stderr
import io
import itertools
import json
from pathlib import Path
import random
import tempfile
import unittest

from ..ecc.bcjr import decode_hard, decode_hard_batch
from ..ecc.trellis_encoding import (
    BASE_TO_BITS,
    CODEBOOK_SHA256,
    GC_OUTPUT_TABLE,
    INPUT_PAIRS,
    LOWER_GENERATOR_INPUTS,
    LOWER_GENERATOR_MATRIX,
    NEXT_STATE_TABLE,
    OUTPUT_TABLE,
    PAIR_PLANE_PATTERNS,
    TRELLIS,
    build_trellis,
    codec_info,
    encode_constrained,
)
from ..ecc.rs import (
    vertical_rs_decode_hard,
    vertical_rs_encode,
)
from ..evaluation import compare_file_hashes
from ..main import build_parser, main as cli_main
from ..pipeline.decoder import decode
from ..pipeline.encoder import encode, resolve_parity_strand_count
from ..pipeline.config import (
    DEFAULT_BCJR_CHANNEL,
    DEFAULT_SIMULATION_CHANNEL,
    derive_default_data_strand_count,
)
from ..pipeline.data_io import (
    WHITENING_ALGORITHM,
    WHITENING_SEED,
    WHITENING_VERSION,
    bytes_to_dna_sequences,
    dna_sequences_to_bytes,
)
from ..simulation import simulate


ZERO_NOISE = {"ps": 0.0, "pd": 0.0, "pi": 0.0, "column": 0.0}


def _max_run(sequence):
    longest = 0
    current = 0
    previous = None
    for base in sequence:
        current = current + 1 if base == previous else 1
        previous = base
        longest = max(longest, current)
    return longest


class ProductionPipelineTests(unittest.TestCase):
    def test_simulation_and_bcjr_channel_defaults_are_independent(self):
        self.assertIsNot(DEFAULT_SIMULATION_CHANNEL, DEFAULT_BCJR_CHANNEL)
        self.assertEqual(
            dict(DEFAULT_SIMULATION_CHANNEL),
            {"ps": 0.0, "pd": 0.0, "pi": 0.0, "column": 0.0},
        )
        self.assertEqual(
            dict(DEFAULT_BCJR_CHANNEL),
            {"ps": 0.0, "pd": 0.0, "pi": 0.0},
        )

    def test_unwhitened_memory_mapping_matches_the_frozen_mapping(self):
        data = bytes((0x00, 0x1B, 0xE4, 0xFF, 0x40))

        sequences = bytes_to_dna_sequences(data, payload_length=16)

        self.assertEqual(
            sequences,
            ["AAAAACGTTGCATTTT", "CAAAAAAAAAAAAAAA"],
        )
        self.assertEqual(
            dna_sequences_to_bytes(sequences, len(data)),
            data,
        )

    def test_whitening_round_trip_and_position_separates_equal_blocks(self):
        repeated_block = b"\x00" * 11
        data = repeated_block * 3

        sequences = bytes_to_dna_sequences(
            data,
            payload_length=44,
            whiten=True,
        )

        self.assertEqual(len(sequences), 3)
        self.assertEqual(len(set(sequences)), 3)
        self.assertEqual(
            dna_sequences_to_bytes(
                sequences,
                len(data),
                whiten=True,
            ),
            data,
        )

    def test_whitening_does_not_spread_an_error_between_bytes_or_blocks(self):
        data = bytes(range(22))
        sequences = bytes_to_dna_sequences(
            data,
            payload_length=44,
            whiten=True,
        )
        damaged = list(sequences)
        damaged_block = list(damaged[1])
        damaged_block[5] = "A" if damaged_block[5] != "A" else "C"
        damaged[1] = "".join(damaged_block)

        recovered = dna_sequences_to_bytes(
            damaged,
            len(data),
            whiten=True,
        )

        self.assertEqual(
            [
                index
                for index, pair in enumerate(zip(data, recovered))
                if pair[0] != pair[1]
            ],
            [12],
        )

    def test_whitening_protocol_is_frozen_in_code(self):
        self.assertEqual(WHITENING_ALGORITHM, "shake256-block-xor")
        self.assertEqual(WHITENING_VERSION, 1)
        self.assertEqual(WHITENING_SEED, 20260801)
        with self.assertRaisesRegex(ValueError, "boolean"):
            bytes_to_dna_sequences(b"strict", 44, whiten="yes")

    def test_codec_parameters_are_frozen(self):
        self.assertEqual(BASE_TO_BITS["A"], (0, 0))
        self.assertEqual(BASE_TO_BITS["C"], (1, 1))
        self.assertEqual(len(INPUT_PAIRS), 16)
        self.assertEqual(len(set(INPUT_PAIRS)), 16)
        self.assertEqual(INPUT_PAIRS[0], "AA")
        self.assertEqual(INPUT_PAIRS[-1], "CC")
        self.assertEqual(len(PAIR_PLANE_PATTERNS), 4)
        self.assertEqual(len(GC_OUTPUT_TABLE), 4)
        self.assertTrue(all(len(row) == 4 for row in GC_OUTPUT_TABLE))
        self.assertEqual(len(LOWER_GENERATOR_INPUTS), 8)
        self.assertEqual(len(LOWER_GENERATOR_MATRIX), 4)
        self.assertTrue(
            all(len(row) == 8 for row in LOWER_GENERATOR_MATRIX)
        )
        self.assertEqual(build_trellis(), TRELLIS)
        self.assertEqual(TRELLIS.initial_state, 0)
        self.assertEqual(TRELLIS.termination_symbol, 0)
        self.assertEqual(TRELLIS.number_states, 16)
        self.assertEqual(TRELLIS.number_inputs, 16)
        self.assertEqual(len(OUTPUT_TABLE), 16)
        self.assertTrue(all(len(row) == 16 for row in OUTPUT_TABLE))
        self.assertTrue(
            all(len(output) == 4 for row in OUTPUT_TABLE for output in row)
        )
        self.assertEqual(len(NEXT_STATE_TABLE), 16)
        self.assertTrue(
            all(row == tuple(range(16)) for row in NEXT_STATE_TABLE)
        )
        info = codec_info()
        self.assertEqual(info["codebook_sha256"], CODEBOOK_SHA256)
        self.assertEqual(info["states"], 16)

    def test_all_local_paths_meet_biological_constraints(self):
        checked_windows = 0
        for upper_states in itertools.product(range(4), repeat=5):
            sequence = "".join(
                OUTPUT_TABLE[4 * upper_states[index]][
                    4 * upper_states[index + 1]
                ]
                for index in range(4)
            )
            self.assertLessEqual(_max_run(sequence), 4)
            for offset in range(4):
                window = sequence[offset : offset + 12]
                gc_count = window.count("G") + window.count("C")
                self.assertGreaterEqual(gc_count, 4)
                self.assertLessEqual(gc_count, 8)
                checked_windows += 1
        self.assertEqual(checked_windows, 4096)

    def test_hard_bcjr_round_trip(self):
        for source in ("", "AC", "ACGTACGT", "TTGGCCAATTGG"):
            transmitted, payload_steps = encode_constrained(source)
            bcjr_decoded_strand, diagnostics = decode_hard(
                transmitted,
                payload_steps,
                {"ps": 0.01, "pd": 0.01, "pi": 0.01},
                drift_window=3,
                return_diagnostics=True,
            )
            self.assertEqual(bcjr_decoded_strand, source)
            self.assertEqual(len(transmitted), 2 * len(source) + 4)
            self.assertEqual(
                diagnostics["algorithm"],
                "log_map",
            )
            self.assertNotIn("selected_base_confidences", diagnostics)

    def test_hard_bcjr_recovers_joint_ids_events(self):
        source = "ATGCAGCGTA"
        transmitted, payload_steps = encode_constrained(source)
        received = list(transmitted)
        received[3] = "A" if received[3] != "A" else "C"
        del received[7]
        received.insert(12, "C")

        bcjr_decoded_strand = decode_hard(
            "".join(received),
            payload_steps,
            {"ps": 0.05, "pd": 0.05, "pi": 0.05},
            drift_window=3,
        )

        self.assertEqual(bcjr_decoded_strand, source)

    def test_hard_bcjr_batch_matches_serial_order(self):
        sources = ("ACGTACGT", "TTGGCCAA", "GATACAGT", "CCAATTGG")
        encoded = [encode_constrained(source) for source in sources]
        channel = {"ps": 0.01, "pd": 0.01, "pi": 0.01}
        serial = [
            decode_hard(dna, steps, channel, drift_window=3)
            for dna, steps in encoded
        ]
        parallel = decode_hard_batch(
            [dna for dna, _ in encoded],
            encoded[0][1],
            channel,
            drift_window=3,
            workers=2,
        )
        self.assertEqual(parallel, serial)

    def test_rs_hard_decoder_recovers_errors_and_missing_strands(self):
        rng = random.Random(17)
        data = [
            "".join(rng.choice("ACGT") for _ in range(20))
            for _ in range(24)
        ]
        encoded, padding_bases = vertical_rs_encode(
            data,
            parity_strand_count=8,
        )
        received = {index: dna for index, dna in enumerate(encoded)}
        del received[3]
        del received[25]
        wrong = received[7]
        received[7] = ("T" if wrong[0] != "T" else "A") + wrong[1:]

        decoded, diagnostics = vertical_rs_decode_hard(
            received,
            data_strand_count=len(data),
            parity_strand_count=8,
            padding_bases=padding_bases,
            return_diagnostics=True,
        )

        self.assertEqual(decoded, data)
        self.assertEqual(diagnostics["algorithm"], "hard")
        self.assertEqual(diagnostics["erasures"], 2)
        self.assertEqual(diagnostics["failed_columns"], 0)

    def test_rs_hard_decoder_rejects_non_strand_observations(self):
        data = ["ACGTACGT"] * 4
        encoded, padding_bases = vertical_rs_encode(
            data,
            parity_strand_count=2,
        )
        received = {index: dna for index, dna in enumerate(encoded)}
        received[0] = [(encoded[0], 1.0), (encoded[0], 0.0)]
        with self.assertRaisesRegex(ValueError, "must contain one DNA strand"):
            vertical_rs_decode_hard(
                received,
                data_strand_count=4,
                parity_strand_count=2,
                padding_bases=padding_bases,
            )

    def test_default_parity_layout_is_reversible_from_seed_length(self):
        self.assertEqual(resolve_parity_strand_count(8150), 815)
        self.assertEqual(
            derive_default_data_strand_count(8965),
            8150,
        )
        self.assertEqual(
            resolve_parity_strand_count(8150, parity_strand_count=80),
            80,
        )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            resolve_parity_strand_count(8150, parity_strand_count=-1)
        with self.assertRaisesRegex(ValueError, "incompatible"):
            derive_default_data_strand_count(1)

    def test_default_encoder_writes_only_strands_and_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.bin"
            source.write_bytes(bytes(range(21)))
            name = str(root / "case")

            encode(
                artifact_prefix=name,
                source_path=str(source),
            )

            seed_payload = json.loads(
                Path(name + "_seed").read_text(encoding="utf-8")
            )
            self.assertEqual(seed_payload["source_size_bytes"], 21)
            self.assertEqual(len(seed_payload["tree_seed"]), 3)
            self.assertEqual(
                set(seed_payload),
                {"source_size_bytes", "tree_seed"},
            )
            encoded_strands = Path(
                name + "_encoded_strands.txt"
            ).read_text(encoding="utf-8").splitlines()
            self.assertTrue(encoded_strands)
            self.assertTrue(
                all(len(strand) == 100 for strand in encoded_strands)
            )
            self.assertEqual(
                {path.name for path in root.glob("case*")},
                {"case_encoded_strands.txt", "case_seed"},
            )

    def test_default_whitening_needs_no_seed_marker_and_round_trips(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.bin"
            source.write_bytes((b"repetition!" * 4) + b"tail")
            name = str(root / "case")
            encode(
                artifact_prefix=name,
                source_path=source,
            )

            parameters_path = Path(name + "_encoding_parameters.json")
            self.assertFalse(parameters_path.exists())
            seed_payload = json.loads(
                Path(name + "_seed").read_text(encoding="utf-8")
            )
            self.assertEqual(
                set(seed_payload),
                {"source_size_bytes", "tree_seed"},
            )
            encoded_strands = Path(name + "_encoded_strands.txt").read_text(
                encoding="utf-8"
            )
            Path(name + "_reads").write_text(
                encoded_strands + encoded_strands,
                encoding="utf-8",
            )
            parameters_path.write_text("not JSON", encoding="utf-8")
            Path(name + "_ecc_meta.json").write_text(
                "not JSON",
                encoding="utf-8",
            )
            restored = root / "restored.bin"

            decode(
                reads_path=Path(name + "_reads"),
                tree_seed_path=Path(name + "_seed"),
                channel=ZERO_NOISE,
                drift_window=0,
                worker_count=2,
                output_path=restored,
            )

            self.assertEqual(restored.read_bytes(), source.read_bytes())

    def test_marked_seed_remains_an_unwhitened_legacy_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.bin"
            source.write_bytes(b"legacy unwhitened source")
            name = str(root / "legacy")
            encode(
                artifact_prefix=name,
                source_path=source,
                whiten=False,
            )

            seed_payload = json.loads(
                Path(name + "_seed").read_text(encoding="utf-8")
            )
            self.assertEqual(
                seed_payload["file_whitening"],
                False,
            )
            self.assertFalse(
                Path(name + "_encoding_parameters.json").exists()
            )
            encoded_strands = Path(name + "_encoded_strands.txt").read_text(
                encoding="utf-8"
            )
            Path(name + "_reads").write_text(
                encoded_strands + encoded_strands,
                encoding="utf-8",
            )
            restored = root / "restored.bin"

            decode(
                reads_path=Path(name + "_reads"),
                tree_seed_path=Path(name + "_seed"),
                channel=ZERO_NOISE,
                drift_window=0,
                worker_count=2,
                output_path=restored,
            )

            self.assertEqual(restored.read_bytes(), source.read_bytes())

    def test_fixed_whitening_is_independent_of_cli_random_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.bin"
            source.write_bytes(b"same input across CLI runs")
            first_name = str(root / "first")
            second_name = str(root / "second")
            cli_main(
                [
                    "encode", "--input", str(source), "--name", first_name,
                    "--random-seed", "12345",
                ]
            )
            cli_main(
                [
                    "encode", "--input", str(source), "--name", second_name,
                    "--random-seed", "54321",
                ]
            )

            self.assertFalse(
                Path(first_name + "_encoding_parameters.json").exists()
            )
            self.assertFalse(
                Path(second_name + "_encoding_parameters.json").exists()
            )
            self.assertEqual(
                Path(first_name + "_encoded_strands.txt").read_bytes(),
                Path(second_name + "_encoded_strands.txt").read_bytes(),
            )

    def test_decoder_rejects_unknown_seed_whitening_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.bin"
            source.write_bytes(b"strict metadata")
            name = str(root / "case")
            encode(
                artifact_prefix=name,
                source_path=source,
            )
            seed_path = Path(name + "_seed")
            seed_payload = json.loads(seed_path.read_text(encoding="utf-8"))
            seed_payload["file_whitening"] = True
            seed_path.write_text(json.dumps(seed_payload), encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError,
                "file_whitening must be false",
            ):
                decode(
                    reads_path=Path(name + "_reads"),
                    tree_seed_path=seed_path,
                    output_path=root / "restored.bin",
                )

    def test_default_artifacts_preserve_source_length_and_trailing_zeros(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.bin"
            rng = random.Random(20260810)
            source.write_bytes(
                bytes(rng.randrange(256) for _ in range(398)) + b"\x00" * 4
            )
            name = str(root / "case")

            encode(artifact_prefix=name, source_path=source)
            encoded_strands = Path(
                name + "_encoded_strands.txt"
            ).read_text(encoding="utf-8")
            Path(name + "_reads").write_text(
                encoded_strands + encoded_strands,
                encoding="utf-8",
            )
            restored = root / "restored.bin"
            decode(
                reads_path=Path(name + "_reads"),
                tree_seed_path=Path(name + "_seed"),
                channel=ZERO_NOISE,
                drift_window=0,
                worker_count=2,
                output_path=restored,
            )

            self.assertEqual(restored.read_bytes(), source.read_bytes())
            self.assertFalse(
                Path(name + "_encoding_parameters.json").exists()
            )
            self.assertFalse(Path(name + "_ecc_meta.json").exists())

    def test_legacy_list_seed_is_rejected_without_sidecar_lookup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            name = str(root / "legacy")
            seed_path = Path(name + "_seed")
            seed_path.write_text("[0]", encoding="utf-8")
            Path(name + "_encoding_parameters.json").write_text(
                "not JSON",
                encoding="utf-8",
            )
            Path(name + "_ecc_meta.json").write_text(
                "not JSON",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "legacy list tree seeds"):
                decode(
                    reads_path=root / "missing_reads",
                    tree_seed_path=seed_path,
                    output_path=root / "restored.bin",
                )

    def test_non_default_encoding_keeps_seed_schema_and_writes_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.bin"
            source.write_bytes(bytes(range(21)))
            name = str(root / "case")

            encode(
                artifact_prefix=name,
                payload_length=20,
                tree_start=2,
                tree_length=16,
                source_path=source,
                parity_strand_count=3,
            )

            seed_payload = json.loads(
                Path(name + "_seed").read_text(encoding="utf-8")
            )
            self.assertEqual(
                set(seed_payload),
                {"source_size_bytes", "tree_seed"},
            )
            parameters = json.loads(
                Path(name + "_encoding_parameters.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                parameters,
                {
                    "parity_strand_count": 3,
                    "payload_length": 20,
                    "tree_start": 2,
                    "tree_length": 16,
                },
            )

    def test_payload_length_must_align_to_rs_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.bin"
            source.write_bytes(b"payload")
            with self.assertRaisesRegex(ValueError, "divisible by 4"):
                encode(
                    artifact_prefix=Path(directory) / "case",
                    payload_length=81,
                    source_path=source,
                )

    def test_pipeline_records_algorithms_and_uses_hard_outputs(self):
        with tempfile.TemporaryDirectory(
            prefix="beezy-pipeline-test-"
        ) as directory:
            name = str(Path(directory) / "case")
            source = Path(directory) / "source.bin"
            source.write_bytes(bytes(range(8)))
            random.seed(91)
            encode(
                artifact_prefix=name,
                payload_length=8,
                tree_length=8,
                source_path=str(source),
                parity_strand_count=4,
            )
            parameters = json.loads(
                Path(name + "_encoding_parameters.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                parameters,
                {
                    "parity_strand_count": 4,
                    "payload_length": 8,
                    "tree_length": 8,
                },
            )
            Path(name + "_encoding_parameters.json").write_text(
                "not JSON",
                encoding="utf-8",
            )
            self.assertFalse(Path(name + "_ecc_meta.json").exists())
            self.assertFalse(Path(name + "_reads").exists())
            self.assertFalse(Path(name + "_ref_msg").exists())
            self.assertFalse(Path(name + "_encode_msg").exists())
            encoded_strands_text = Path(
                name + "_encoded_strands.txt"
            ).read_text(encoding="utf-8")
            self.assertFalse(Path(name + "_tx_msg").exists())
            Path(name + "_reads").write_text(
                encoded_strands_text + encoded_strands_text,
                encoding="utf-8",
            )
            restored = Path(directory) / "restored.bin"
            rs_decoded_strands, diagnostics = decode(
                reads_path=Path(name + "_reads"),
                tree_seed_path=Path(name + "_seed"),
                channel=ZERO_NOISE,
                drift_window=0,
                worker_count=2,
                output_path=str(restored),
                return_diagnostics=True,
                parity_strand_count=4,
                payload_length=8,
                tree_length=8,
            )

            self.assertTrue(rs_decoded_strands)
            self.assertTrue(
                compare_file_hashes(source, restored)["match"]
            )
            self.assertEqual(
                diagnostics["decoder"]["codec_fingerprint"],
                CODEBOOK_SHA256,
            )
            self.assertEqual(diagnostics["bcjr"]["algorithm"], "log_map")
            self.assertEqual(diagnostics["bcjr"]["worker_count"], 2)
            self.assertEqual(
                diagnostics["decoder"]["route"],
                "payload_frequency_index_recovery",
            )
            self.assertEqual(
                diagnostics["payload_aggregation"][
                    "minimum_payload_frequency"
                ],
                2,
            )
            self.assertNotIn("legal_tree_maximum_depth", diagnostics)
            self.assertNotIn("strand_errors_after_rs", diagnostics)
            self.assertNotIn("base_errors_after_rs", diagnostics)
            self.assertNotIn("file_hash_match", diagnostics)
            self.assertEqual(
                diagnostics["reed_solomon"]["algorithm"],
                "hard",
            )
            self.assertFalse(
                diagnostics["reed_solomon"]["output_is_partial"]
            )

    def test_missing_strands_are_recovered_and_file_hash_matches(self):
        with tempfile.TemporaryDirectory(
            prefix="beezy-file-test-"
        ) as directory:
            root = Path(directory)
            source = root / "source.bin"
            source.write_bytes(bytes(range(31)))
            name = str(root / "case")
            random.seed(23)
            encode(
                artifact_prefix=name,
                payload_length=16,
                tree_length=16,
                source_path=str(source),
                parity_strand_count=4,
            )
            encoded_strands = Path(
                name + "_encoded_strands.txt"
            ).read_text(encoding="utf-8").splitlines()
            retained = [
                strand
                for index, strand in enumerate(encoded_strands)
                if index not in {1, 6}
            ]
            Path(name + "_reads").write_text(
                "\n".join(retained * 2) + "\n", encoding="utf-8"
            )
            restored = root / "restored.bin"
            _, diagnostics = decode(
                reads_path=Path(name + "_reads"),
                tree_seed_path=Path(name + "_seed"),
                channel=ZERO_NOISE,
                drift_window=2,
                worker_count=2,
                output_path=str(restored),
                return_diagnostics=True,
                parity_strand_count=4,
                payload_length=16,
                tree_length=16,
            )

            self.assertEqual(restored.read_bytes(), source.read_bytes())
            self.assertEqual(
                diagnostics["index_recovery"]["summary"][
                    "unresolved_index_count"
                ],
                2,
            )
            self.assertTrue(
                compare_file_hashes(source, restored)["match"]
            )

    def test_simulation_is_an_explicit_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            name = str(Path(directory) / "case")
            source = Path(directory) / "source.bin"
            source.write_bytes(bytes(range(8)))
            encode(
                artifact_prefix=name,
                payload_length=8,
                tree_length=8,
                source_path=str(source),
                parity_strand_count=4,
            )

            random.seed(17)
            reads_path = simulate(
                name,
                sequencing_depth=2,
                channel=ZERO_NOISE,
            )

            self.assertEqual(reads_path, Path(name + "_reads"))
            self.assertTrue(reads_path.exists())
            self.assertGreater(len(reads_path.read_text().splitlines()), 0)

    def test_decoder_rejects_unknown_seed_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seed_path = root / "seed"
            seed_path.write_text(
                json.dumps(
                    {
                        "source_size_bytes": 0,
                        "tree_seed": [0],
                        "artifact_prefix": "forbidden",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "only the optional"):
                decode(
                    reads_path=root / "reads",
                    tree_seed_path=seed_path,
                )

    def test_decoder_rejects_layout_parameters_embedded_in_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seed_path = root / "seed"
            seed_path.write_text(
                json.dumps(
                    {
                        "source_size_bytes": 0,
                        "tree_seed": [0],
                        "encoding_parameters": {},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "only the optional"):
                decode(
                    reads_path=root / "reads",
                    tree_seed_path=seed_path,
                )

    def test_cli_exposes_only_production_controls(self):
        parser = build_parser()
        encode_args = parser.parse_args(["encode", "--input", "source.bin"])
        simulate_args = parser.parse_args(["simulate"])
        decode_args = parser.parse_args(
            [
                "decode",
                "--reads",
                "reads.txt",
                "--tree-seed",
                "seed.json",
                "--output",
                "out.bin",
            ]
        )
        non_default_decode_args = parser.parse_args(
            [
                "decode",
                "--reads",
                "reads.txt",
                "--tree-seed",
                "seed.json",
                "--output",
                "out.bin",
                "--parity-strands",
                "4",
                "--payload-length",
                "20",
                "--tree-start",
                "2",
                "--tree-length",
                "16",
            ]
        )
        evaluate_args = parser.parse_args(
            ["evaluate", "--original", "source.bin", "--recovered", "out.bin"]
        )

        self.assertEqual(encode_args.command, "encode")
        self.assertIsNone(encode_args.parity_strands)
        self.assertEqual(encode_args.payload_length, 44)
        self.assertEqual(encode_args.tree_start, 0)
        self.assertEqual(encode_args.tree_length, 44)
        self.assertTrue(encode_args.whiten)
        whiten_args = parser.parse_args(
            ["encode", "--input", "source.bmp", "--whiten"]
        )
        self.assertTrue(whiten_args.whiten)
        no_whiten_args = parser.parse_args(
            ["encode", "--input", "source.bmp", "--no-whiten"]
        )
        self.assertFalse(no_whiten_args.whiten)
        count_args = parser.parse_args(
            ["encode", "--input", "source.bin", "--parity-strands", "80"]
        )
        length_args = parser.parse_args(
            [
                "encode",
                "--input",
                "source.bin",
                "--payload-length",
                "100",
            ]
        )
        self.assertEqual(count_args.parity_strands, 80)
        self.assertEqual(length_args.payload_length, 100)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(
                ["encode", "--input", "source.bin", "--parity-rate", "0.1"]
            )
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(
                ["encode", "--input", "source.bin", "--strand-length", "100"]
            )
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "encode",
                    "--input",
                    "source.bin",
                    "--data-strand-length",
                    "100",
                ]
            )
        self.assertEqual(simulate_args.command, "simulate")
        self.assertEqual(simulate_args.depth, 100)
        self.assertEqual(simulate_args.substitution, 0.0)
        self.assertEqual(simulate_args.deletion, 0.0)
        self.assertEqual(simulate_args.insertion, 0.0)
        self.assertEqual(simulate_args.dropout, 0.0)
        self.assertEqual(decode_args.command, "decode")
        self.assertEqual(decode_args.substitution, 0.0)
        self.assertEqual(decode_args.deletion, 0.0)
        self.assertEqual(decode_args.insertion, 0.0)
        self.assertIsNone(decode_args.parity_strands)
        self.assertEqual(decode_args.payload_length, 44)
        self.assertEqual(decode_args.tree_start, 0)
        self.assertEqual(decode_args.tree_length, 44)
        self.assertEqual(non_default_decode_args.parity_strands, 4)
        self.assertEqual(non_default_decode_args.payload_length, 20)
        self.assertEqual(non_default_decode_args.tree_start, 2)
        self.assertEqual(non_default_decode_args.tree_length, 16)
        self.assertEqual(evaluate_args.command, "evaluate")
        self.assertEqual(decode_args.drift, 3)
        self.assertEqual(decode_args.workers, 24)
        self.assertFalse(hasattr(decode_args, "name"))
        self.assertFalse(hasattr(decode_args, "threshold"))
        self.assertFalse(hasattr(decode_args, "depth"))
        self.assertFalse(hasattr(encode_args, "substitution"))
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["decode"])
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "decode",
                    "--reads",
                    "reads.txt",
                    "--tree-seed",
                    "seed.json",
                    "--output",
                    "out.bin",
                    "--name",
                    "forbidden",
                ]
            )


if __name__ == "__main__":
    unittest.main()
