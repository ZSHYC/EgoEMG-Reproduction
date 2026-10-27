"""Byte-preserving compact blocks, independent format fixtures and strict bounds."""
import hashlib
import importlib
import lzma

import pytest


LIMIT=1024**2


def api():return importlib.import_module('egoemg_repro.emg2pose_compact_codec')
def sha(raw):return hashlib.sha256(raw).hexdigest()
def read(encoded,raw,**changes):
    options=dict(raw_bytes=len(raw),raw_sha256=sha(raw),encoded_sha256=sha(encoded))
    options.update(changes)
    return api().decode_block(encoded,**options)


def test_encoder_matches_independent_two_row_wire_fixture():
    encoded=api().encode_block(b'\x07'*152+b'\x0b'*152)
    assert lzma.decompress(encoded)==b'\x07\x04'*152


def test_decoder_recovers_independently_constructed_two_row_fixture():
    raw=b'\x07'*152+b'\x0b'*152
    encoded=lzma.compress(b'\x07\x04'*152,format=lzma.FORMAT_XZ,preset=6)
    assert read(encoded,raw)==raw


@pytest.mark.parametrize('size',[1,151,152,153,303,304,LIMIT-1,LIMIT])
def test_all_bytes_and_partial_rows_recover_exactly(size):
    # Includes unsigned wraparound and padding that must be checked then removed.
    raw=(bytes(range(256))*((size+255)//256))[:size]
    encoded=api().encode_block(raw)
    assert read(encoded,raw)==raw


def test_float_bit_patterns_are_preserved_without_numeric_conversion():
    raw=bytes.fromhex('00000000 00000080 0100c07f 0000807f 000080ff ffffffff')*1001
    assert read(api().encode_block(raw),raw)==raw


@pytest.mark.parametrize('raw',[b'',b'x'*(LIMIT+1),bytearray(b'x'),memoryview(b'x'),'x',None],
                         ids=['empty','oversized','bytearray','memoryview','text','none'])
def test_invalid_or_unbounded_encoder_input_is_refused(raw):
    with pytest.raises(ValueError):api().encode_block(raw)


@pytest.mark.parametrize('size',[True,False,0,-1,LIMIT+1,1.0,'1',None])
def test_decoder_rejects_invalid_raw_size_before_output(size):
    raw=b'x';encoded=lzma.compress(b'x'+b'\0'*151)
    with pytest.raises(ValueError):read(encoded,raw,raw_bytes=size)


@pytest.mark.parametrize('encoded',[b'',b'x'*(LIMIT+65537),bytearray(b'x'),memoryview(b'x'),'x',None],
                         ids=['empty','oversized','bytearray','memoryview','text','none'])
def test_invalid_or_unbounded_encoded_input_is_refused(encoded):
    with pytest.raises(ValueError):api().decode_block(encoded,raw_bytes=1,raw_sha256='0'*64,encoded_sha256='0'*64)


@pytest.mark.parametrize('field',['raw_sha256','encoded_sha256'])
@pytest.mark.parametrize('value',['0'*64,'F'*64,'f'*63,None,True])
def test_each_separately_frozen_digest_is_required(field,value):
    raw=b'source bytes';encoded=api().encode_block(raw)
    with pytest.raises(ValueError):read(encoded,raw,**{field:value})


@pytest.mark.parametrize('damage',['truncated','corrupt','trailing','concatenated'])
def test_resealed_malformed_xz_stream_is_refused(damage):
    raw=b'x';encoded=api().encode_block(raw)
    if damage=='truncated':encoded=encoded[:-1]
    elif damage=='corrupt':encoded=encoded[:20]+bytes([encoded[20]^1])+encoded[21:]
    elif damage=='trailing':encoded+=b'\0'
    else:encoded+=encoded
    with pytest.raises(ValueError):read(encoded,raw)


@pytest.mark.parametrize('payload',[b'\x2a\x01'+b'\0'*150,b'\x2a'+b'\0'*150,b'\x2a'+b'\0'*1024],
                         ids=['nonzero-padding','short-output','output-overrun'])
def test_nonzero_padding_wrong_length_or_bounded_output_overrun_is_refused(payload):
    # Valid compressed hash and correct first raw byte cannot excuse bad padding,
    # a short decoded block or a stream that expands past the declared bound.
    encoded=lzma.compress(payload)
    with pytest.raises(ValueError):read(encoded,b'\x2a')
