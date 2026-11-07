"""Headerless fixed-width retrieval permutations and lossless XZ compression.

The public API accepts Beezy's one-based tree seeds. Packed values are
zero-based, most-significant bit first, with zero padding in the final byte.
"""

import argparse
import hashlib
import json
import lzma
from numbers import Integral
from pathlib import Path


def _layout(count):
    if isinstance(count, bool) or not isinstance(count, Integral) or count < 1:
        raise ValueError('strand count must be a positive integer')
    count = int(count)
    width = (count - 1).bit_length()
    return width, (count * width + 7) // 8


def _validate(values):
    count = len(values)
    _layout(count)
    seen = bytearray(count)
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError('tree seed must contain integer identifiers')
        if not 1 <= value <= count or seen[value - 1]:
            raise ValueError('tree seed must be a permutation of 1..N')
        seen[value - 1] = 1


def pack_key(tree_seed):
    """Return the packed permutation body, without application metadata."""
    values = list(tree_seed)
    _validate(values)
    width, size = _layout(len(values))
    result = bytearray()
    buffer = bits = 0
    for value in values:
        buffer = (buffer << width) | (int(value) - 1)
        bits += width
        while bits >= 8:
            bits -= 8
            result.append((buffer >> bits) & 255)
        buffer &= (1 << bits) - 1
    if bits:
        result.append(buffer << (8 - bits))
    assert len(result) == size
    return bytes(result)


def unpack_key(body, count):
    """Restore a one-based permutation using its externally supplied count."""
    width, size = _layout(count)
    if len(body) != size:
        raise ValueError('packed key length does not match strand count')
    if width == 0:
        return [1]
    values = []
    buffer = bits = offset = 0
    for _ in range(count):
        while bits < width:
            buffer = (buffer << 8) | body[offset]
            offset += 1
            bits += 8
        bits -= width
        values.append((buffer >> bits) + 1)
        buffer &= (1 << bits) - 1
    if buffer:
        raise ValueError('nonzero final-byte padding')
    _validate(values)
    return values


def compress_key(tree_seed):
    """Return an XZ/LZMA2 stream containing the fixed-width body."""
    return lzma.compress(pack_key(tree_seed), format=lzma.FORMAT_XZ, preset=6)


def decompress_key(stream, count):
    """Decode one XZ stream; reject excess output, trailing data or truncation."""
    _, size = _layout(count)
    decoder = lzma.LZMADecompressor(format=lzma.FORMAT_XZ, memlimit=128 << 20)
    try:
        body = decoder.decompress(stream, max_length=size + 1)
    except lzma.LZMAError as exc:
        raise ValueError('invalid compressed key') from exc
    if not decoder.eof or decoder.unused_data or len(body) != size:
        raise ValueError('compressed key length or stream boundary mismatch')
    return unpack_key(body, count)


def pack_file(source, prefix):
    """Save raw/compressed bodies and separate restoration metadata."""
    payload = json.loads(Path(source).read_text(encoding='utf-8'))
    if isinstance(payload, list):
        values = payload
    elif isinstance(payload, dict) and isinstance(payload.get('tree_seed'), list):
        values = payload['tree_seed']
    else:
        raise ValueError('expected a Beezy JSON seed object or permutation list')
    body = pack_key(values)
    compressed = compress_key(values)
    prefix = str(prefix)
    paths = [Path(prefix + suffix) for suffix in ('.packed', '.packed.xz', '.metadata.json')]
    if any(path.exists() for path in paths):
        raise FileExistsError('output exists; use a new prefix')
    metadata = dict(version=1, strand_count=len(values),
                    body_sha256=hashlib.sha256(body).hexdigest(),
                    seed_fields={k: v for k, v in payload.items() if k != 'tree_seed'}
                    if isinstance(payload, dict) else {})
    if isinstance(payload, list):
        metadata['seed_container'] = 'list'
    for path, data in zip(paths, [body, compressed, (json.dumps(metadata, indent=2)+'\n').encode()]):
        with path.open('xb') as handle:
            path.chmod(0o600)
            handle.write(data)
    return dict(strand_count=len(values), packed_bytes=len(body), compressed_bytes=len(compressed))


def unpack_file(source, metadata_path, target):
    """Restore the original JSON object or legacy experiment permutation list."""
    source = Path(source)
    metadata = json.loads(Path(metadata_path).read_text())
    if metadata['version'] != 1 or 'tree_seed' in metadata['seed_fields']:
        raise ValueError('invalid restoration metadata')
    container = metadata.get('seed_container', 'object')
    if container not in ('list', 'object') or (container == 'list' and metadata['seed_fields']):
        raise ValueError('invalid seed container')
    raw = source.read_bytes()
    count = metadata['strand_count']
    values = decompress_key(raw, count) if source.suffix == '.xz' else unpack_key(raw, count)
    if hashlib.sha256(pack_key(values)).hexdigest() != metadata['body_sha256']:
        raise ValueError('key body checksum mismatch')
    payload = values if container == 'list' else dict(metadata['seed_fields'], tree_seed=values)
    with Path(target).open('x', encoding='utf-8') as handle:
        Path(target).chmod(0o600)
        json.dump(payload, handle, ensure_ascii=False)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    pack = commands.add_parser('pack')
    pack.add_argument('source')
    pack.add_argument('prefix')
    unpack = commands.add_parser('unpack')
    unpack.add_argument('source')
    unpack.add_argument('metadata')
    unpack.add_argument('target')
    args = parser.parse_args()
    if args.command == 'pack':
        print(json.dumps(pack_file(args.source, args.prefix)))
    else:
        unpack_file(args.source, args.metadata, args.target)
        print(args.target)


if __name__ == '__main__':
    main()
