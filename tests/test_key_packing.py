import json
import lzma
import random
from pathlib import Path

import pytest

from ..pipeline.key_packing import (
    pack_key, unpack_key, compress_key, decompress_key, pack_file, unpack_file,
)
from ..pipeline.decoder import _load_artifact_layout


@pytest.mark.parametrize('count', [1, 2, 3, 7, 8, 9, 255, 256, 257, 10000])
def test_roundtrip(count):
    values = list(range(1, count + 1))
    random.Random(17).shuffle(values)
    body = pack_key(values)
    assert len(body) == (count * (count - 1).bit_length() + 7) // 8
    assert unpack_key(body, count) == values
    assert decompress_key(compress_key(values), count) == values


def test_bit_order():
    # Zero-based [2, 0, 1] -> 10 00 01 00 (last two bits are padding).
    assert pack_key([3, 1, 2]) == b'\x84'


@pytest.mark.parametrize('values', [[], [0], [2], [1, 1], [1, 3], [True], [1.0], ['1']])
def test_reject_invalid_permutations(values):
    with pytest.raises(ValueError):
        pack_key(values)


@pytest.mark.parametrize('body,count', [(b'', 3), (b'\x84\x00', 3), (b'\x85', 3), (b'\x00', 3), (b'', 0)])
def test_reject_invalid_bodies(body, count):
    with pytest.raises(ValueError):
        unpack_key(body, count)


def test_reject_invalid_streams():
    stream = compress_key([3, 1, 2])
    for bad in [stream[:-1], stream + b'x', stream + stream, b'bad', lzma.compress(b'x' * 10000)]:
        with pytest.raises(ValueError):
            decompress_key(bad, 3)


@pytest.mark.parametrize('unwhitened', [False, True])
def test_file_conversion_and_decoder_layout(tmp_path, unwhitened):
    payload = dict(tree_seed=[3, 1, 2], source_size_bytes=33)
    if unwhitened:
        payload['file_whitening'] = False
    source = tmp_path/'seed.json'
    source.write_text(json.dumps(payload))
    prefix = tmp_path/'compact'
    sizes = pack_file(source, prefix)
    assert sizes['packed_bytes'] == 1
    for suffix in ['.packed', '.packed.xz']:
        target = tmp_path/('restored'+suffix+'.json')
        assert unpack_file(str(prefix)+suffix, str(prefix)+'.metadata.json', target) == payload
        assert _load_artifact_layout(target, {'parity_strand_count': 0}) == _load_artifact_layout(source, {'parity_strand_count': 0})
    with pytest.raises(FileExistsError):
        pack_file(source, prefix)


def test_checksum_detects_valid_but_changed_permutation(tmp_path):
    source = tmp_path/'seed.json'
    source.write_text(json.dumps(dict(tree_seed=[3, 1, 2], source_size_bytes=33)))
    prefix = tmp_path/'compact'
    pack_file(source, prefix)
    Path(str(prefix)+'.packed').write_bytes(pack_key([1, 3, 2]))
    with pytest.raises(ValueError, match='checksum'):
        unpack_file(str(prefix)+'.packed', str(prefix)+'.metadata.json', tmp_path/'bad.json')


def test_legacy_experiment_list_preserves_container(tmp_path):
    source = tmp_path/'seed.json'
    values = [3, 1, 2]
    source.write_text(json.dumps(values))
    prefix = tmp_path/'compact'
    pack_file(source, prefix)
    for suffix in ['.packed', '.packed.xz']:
        target = tmp_path/('restored'+suffix+'.json')
        assert unpack_file(str(prefix)+suffix, str(prefix)+'.metadata.json', target) == values
        assert json.loads(target.read_text()) == values
