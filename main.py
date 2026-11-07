"""Command-line entry for Beezy encoding, simulation, decoding, and evaluation."""

import argparse
import random

try:
    from .evaluation import compare_file_hashes
    from .pipeline.decoder import decode
    from .pipeline.encoder import encode
    from .pipeline import config
    from .simulation import simulate
except ImportError:
    from evaluation import compare_file_hashes
    from pipeline.decoder import decode
    from pipeline.encoder import encode
    from pipeline import config
    from simulation import simulate


def _add_artifact_prefix(parser):
    parser.add_argument(
        "--name",
        default=config.DEFAULT_ARTIFACT_PREFIX,
        help="artifact prefix",
    )


def _add_channel_parameters(parser, defaults, include_dropout=False):
    parser.add_argument(
        "--substitution",
        type=float,
        default=defaults["ps"],
    )
    parser.add_argument(
        "--deletion", type=float, default=defaults["pd"]
    )
    parser.add_argument(
        "--insertion", type=float, default=defaults["pi"]
    )
    if include_dropout:
        parser.add_argument(
            "--dropout", type=float, default=defaults["column"]
        )


def _channel_from_args(args):
    channel = {
        "ps": args.substitution,
        "pd": args.deletion,
        "pi": args.insertion,
    }
    if hasattr(args, "dropout"):
        channel["column"] = args.dropout
    return channel


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Beezy constrained DNA codec with hard-decision Reed-Solomon decoding"
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    encode_parser = commands.add_parser("encode", help="encode a DNA artifact")
    _add_artifact_prefix(encode_parser)
    encode_parser.add_argument("--input", help="source file; omit for random strands")
    encode_parser.add_argument(
        "--random-strands",
        type=int,
        default=config.DEFAULT_RANDOM_STRAND_COUNT,
    )
    encode_parser.add_argument(
        "--payload-length",
        type=int,
        default=config.DEFAULT_PAYLOAD_LENGTH,
    )
    encode_parser.add_argument(
        "--tree-start", type=int, default=config.DEFAULT_TREE_START
    )
    encode_parser.add_argument(
        "--tree-length", type=int, default=config.DEFAULT_TREE_LENGTH
    )
    encode_parser.add_argument(
        "--parity-strands",
        type=int,
        help=(
            "absolute number of Reed-Solomon parity strands; "
            "defaults to ceil(data strands * 0.10)"
        ),
    )
    encode_parser.add_argument(
        "--tree-seed",
        help="computer-side tree seed output; defaults to <name>_seed",
    )
    encode_parser.add_argument(
        "--random-seed", type=int, default=config.DEFAULT_RANDOM_SEED
    )
    encode_parser.add_argument(
        "--whiten",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "apply frozen reversible position-dependent file whitening "
            "before DNA mapping and Reed-Solomon encoding (default: enabled)"
        ),
    )

    simulate_parser = commands.add_parser(
        "simulate",
        help="generate sequencing reads from an encoded artifact",
    )
    _add_artifact_prefix(simulate_parser)
    simulate_parser.add_argument("--output", help="simulated reads path")
    simulate_parser.add_argument(
        "--depth", type=int, default=config.DEFAULT_SIMULATION_DEPTH
    )
    simulate_parser.add_argument(
        "--random-seed", type=int, default=config.DEFAULT_RANDOM_SEED
    )
    _add_channel_parameters(
        simulate_parser,
        config.DEFAULT_SIMULATION_CHANNEL,
        include_dropout=True,
    )

    decode_parser = commands.add_parser("decode", help="recover an artifact")
    decode_parser.add_argument(
        "--reads",
        required=True,
        help="sequencing reads path",
    )
    decode_parser.add_argument(
        "--output",
        required=True,
        help="reconstructed file path",
    )
    decode_parser.add_argument(
        "--tree-seed",
        required=True,
        help="computer-side tree seed input",
    )
    decode_parser.add_argument(
        "--payload-length",
        type=int,
        default=config.DEFAULT_PAYLOAD_LENGTH,
    )
    decode_parser.add_argument(
        "--tree-start", type=int, default=config.DEFAULT_TREE_START
    )
    decode_parser.add_argument(
        "--tree-length", type=int, default=config.DEFAULT_TREE_LENGTH
    )
    decode_parser.add_argument(
        "--parity-strands",
        type=int,
        help=(
            "absolute number of Reed-Solomon parity strands; "
            "defaults to the frozen ten-percent layout inferred from seed"
        ),
    )
    decode_parser.add_argument(
        "--drift", type=int, default=config.DEFAULT_DRIFT_WINDOW
    )
    decode_parser.add_argument(
        "--workers",
        type=int,
        default=config.DEFAULT_WORKER_COUNT,
        help="BCJR worker processes; 0 uses all available logical CPUs",
    )
    _add_channel_parameters(decode_parser, config.DEFAULT_BCJR_CHANNEL)

    evaluate_parser = commands.add_parser(
        "evaluate",
        help="compare an original file with a recovered file",
    )
    evaluate_parser.add_argument("--original", required=True)
    evaluate_parser.add_argument("--recovered", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.command == "encode":
        random.seed(args.random_seed)
        return encode(
            artifact_prefix=args.name,
            strand_count=args.random_strands,
            payload_length=args.payload_length,
            tree_length=args.tree_length,
            tree_start=args.tree_start,
            source_path=args.input,
            write_artifacts=True,
            parity_strand_count=args.parity_strands,
            tree_seed_path=args.tree_seed,
            whiten=args.whiten,
        )

    if args.command == "simulate":
        random.seed(args.random_seed)
        output_path = simulate(
            artifact_prefix=args.name,
            sequencing_depth=args.depth,
            channel=_channel_from_args(args),
            output_path=args.output,
        )
        print(f"模拟读段已写入：{output_path}")
        return output_path

    if args.command == "decode":
        _, diagnostics = decode(
            reads_path=args.reads,
            tree_seed_path=args.tree_seed,
            channel=_channel_from_args(args),
            drift_window=args.drift,
            worker_count=args.workers,
            output_path=args.output,
            parity_strand_count=args.parity_strands,
            payload_length=args.payload_length,
            tree_start=args.tree_start,
            tree_length=args.tree_length,
            return_diagnostics=True,
        )
        return diagnostics

    result = compare_file_hashes(args.original, args.recovered)
    print("File hashes match." if result["match"] else "File hashes do not match.")
    return result


if __name__ == "__main__":
    main()
