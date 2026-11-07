"""Streaming raw-read frequency counting for production decoding."""

from array import array
from pathlib import Path


_UINT64_ITEMSIZE = array("Q").itemsize
if _UINT64_ITEMSIZE != 8:
    raise RuntimeError('array("Q") must provide 64-bit unsigned integers')
UINT64_MAX = (1 << 64) - 1


def _increment_uint64(counts, slot):
    if counts[slot] == UINT64_MAX:
        raise OverflowError("raw read frequency exceeds uint64 capacity")
    counts[slot] += 1


class CompactRawFrequencyTable:
    """Frequency-sorted unique reads backed by compact uint64 arrays."""

    __slots__ = (
        "_counts",
        "_frequency_order",
        "_unique_reads",
        "raw_read_count",
    )

    def __init__(
        self,
        unique_reads,
        counts,
        frequency_order,
        raw_read_count,
    ):
        self._unique_reads = unique_reads
        self._counts = counts
        self._frequency_order = frequency_order
        self.raw_read_count = int(raw_read_count)

    def __len__(self):
        return len(self._frequency_order)

    def __getitem__(self, position):
        slot = self._frequency_order[position]
        return self._unique_reads[slot], int(self._counts[slot])

    def __iter__(self):
        return zip(
            map(self._unique_reads.__getitem__, self._frequency_order),
            map(self._counts.__getitem__, self._frequency_order),
        )

    @property
    def unique_read_count(self):
        return len(self)

    @property
    def duplicate_read_count(self):
        return self.raw_read_count - self.unique_read_count

    def iter_reads(self):
        for slot in self._frequency_order:
            yield self._unique_reads[slot]


def read_raw_frequency_table(reads_path):
    """Count non-empty stripped reads without retaining every observation."""
    unique_reads = []
    counts = array("Q")
    repeated_slots = array("Q")
    slot_by_read = {}
    raw_read_count = 0

    with Path(reads_path).open(encoding="utf-8") as reads_file:
        for line in reads_file:
            raw_read = line.strip()
            if not raw_read:
                continue
            if raw_read_count == UINT64_MAX:
                raise OverflowError("raw read count exceeds uint64 capacity")
            raw_read_count += 1

            slot = slot_by_read.get(raw_read)
            if slot is None:
                if len(unique_reads) == UINT64_MAX:
                    raise OverflowError(
                        "unique raw read count exceeds uint64 capacity"
                    )
                slot = len(unique_reads)
                slot_by_read[raw_read] = slot
                unique_reads.append(raw_read)
                counts.append(1)
            else:
                if counts[slot] == 1:
                    repeated_slots.append(slot)
                _increment_uint64(counts, slot)

    repeated_order = sorted(
        repeated_slots,
        key=lambda slot: (-counts[slot], slot),
    )
    frequency_order = array("Q", repeated_order)
    frequency_order.extend(
        slot for slot, count in enumerate(counts) if count == 1
    )
    return CompactRawFrequencyTable(
        unique_reads=unique_reads,
        counts=counts,
        frequency_order=frequency_order,
        raw_read_count=raw_read_count,
    )


__all__ = [
    "CompactRawFrequencyTable",
    "UINT64_MAX",
    "read_raw_frequency_table",
]
