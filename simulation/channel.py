"""Independent insertion, deletion, substitution, and dropout channel."""

from pathlib import Path
import random

try:  # Package execution: ``python -m Beezy.main``.
    from ..pipeline.config import (
        DEFAULT_SIMULATION_CHANNEL,
        DEFAULT_SIMULATION_DEPTH,
    )
except ImportError:  # Direct execution through ``python main.py``.
    from pipeline.config import (
        DEFAULT_SIMULATION_CHANNEL,
        DEFAULT_SIMULATION_DEPTH,
    )


DNA_ALPHABET = "ATGC"
DNA_TO_INDEX = {base: index for index, base in enumerate(DNA_ALPHABET)}


def transmit_sequence(sequence, channel):
    """Sample one noisy read from a transmitted DNA sequence."""
    if any(base not in DNA_TO_INDEX for base in sequence):
        raise ValueError("sequence must contain only A, C, G, and T")
    probabilities = {
        name: float(channel.get(name, 0.0))
        for name in ("pi", "pd", "ps", "column")
    }
    if any(not 0.0 <= value <= 1.0 for value in probabilities.values()):
        raise ValueError("channel probabilities must lie in [0, 1]")
    if random.random() < probabilities["column"]:
        return ""

    received = []
    for base in sequence:
        if random.random() < probabilities["pi"]:
            received.append(random.choice(DNA_ALPHABET))
        if random.random() < probabilities["pd"]:
            continue
        if random.random() < probabilities["ps"]:
            offset = random.randint(1, 3)
            received.append(
                DNA_ALPHABET[(DNA_TO_INDEX[base] + offset) % 4]
            )
        else:
            received.append(base)
    return "".join(received)


def write_simulated_reads(
    output_path,
    encoded_strands,
    sequencing_depth=DEFAULT_SIMULATION_DEPTH,
    channel=None,
):
    """Write random-with-replacement reads from an encoded strand pool."""
    strands = tuple(encoded_strands)
    if not strands:
        raise ValueError("encoded_strands must not be empty")
    sequencing_depth = int(sequencing_depth)
    if sequencing_depth <= 0:
        raise ValueError("sequencing_depth must be positive")
    probabilities = dict(
        DEFAULT_SIMULATION_CHANNEL if channel is None else channel
    )
    read_count = len(strands) * sequencing_depth
    resolved_output_path = Path(output_path)
    with resolved_output_path.open("w", encoding="utf-8") as output_file:
        for _ in range(read_count):
            received = transmit_sequence(
                random.choice(strands),
                probabilities,
            )
            if received:
                output_file.write(received + "\n")
    return resolved_output_path


def simulate(
    artifact_prefix,
    sequencing_depth=DEFAULT_SIMULATION_DEPTH,
    channel=None,
    output_path=None,
):
    """Simulate reads from an encoded Beezy artifact."""
    artifact_prefix = str(artifact_prefix)
    encoded_path = Path(artifact_prefix + "_encoded_strands.txt")
    if not encoded_path.exists():
        raise ValueError(f"missing encoded strand file: {encoded_path}")
    encoded_strands = [
        line.strip()
        for line in encoded_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    resolved_output_path = output_path or artifact_prefix + "_reads"
    return write_simulated_reads(
        resolved_output_path,
        encoded_strands,
        sequencing_depth=sequencing_depth,
        channel=channel,
    )


__all__ = ["simulate", "transmit_sequence", "write_simulated_reads"]
