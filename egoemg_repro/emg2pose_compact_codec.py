"""Bounded lossless blocks for a separately versioned compact source format.

All operations act on bytes, including original TAR/HDF headers and padding.
This codec grants no source, dataset, model or scientific completion status.
It does not change the existing persistent XZ archive or raw-TAR fd format.
"""
from __future__ import annotations

import hashlib
import lzma
import re

import numpy as np


CODEC='delta152-plane152-xz6-v1'
CHUNK_BYTES=1024**2
MAX_ENCODED_BYTES=CHUNK_BYTES+65536
DECODE_MEMORY_BYTES=64*1024**2
ROW_BYTES=152
_SHA=re.compile('[0-9a-f]{64}')


def encode_block(raw: bytes) -> bytes:
    """Encode 1..1MiB without interpreting or rounding any original value."""
    if type(raw) is not bytes or not 0<len(raw)<=CHUNK_BYTES:
        raise ValueError('compact raw block must contain 1..1MiB exact bytes')
    rows=(len(raw)+ROW_BYTES-1)//ROW_BYTES
    original=np.zeros((rows,ROW_BYTES),np.uint8)
    original.ravel()[:len(raw)]=np.frombuffer(raw,np.uint8)
    delta=original.copy()
    delta[1:]=original[1:]-original[:-1]
    encoded=lzma.compress(delta.T.copy().tobytes(),format=lzma.FORMAT_XZ,preset=6)
    if len(encoded)>MAX_ENCODED_BYTES:
        raise ValueError('compact encoded block exceeds fixed storage bound')
    return encoded


def decode_block(encoded: bytes, *, raw_bytes: int, raw_sha256: str, encoded_sha256: str) -> bytes:
    """Require both frozen hashes, bounded single-stream output and zero padding."""
    if type(raw_bytes) is not int or not 0<raw_bytes<=CHUNK_BYTES:
        raise ValueError('compact declared raw size must be 1..1MiB')
    if type(encoded) is not bytes or not 0<len(encoded)<=MAX_ENCODED_BYTES:
        raise ValueError('compact encoded block exceeds fixed input bound')
    if any(type(s) is not str or _SHA.fullmatch(s) is None for s in (raw_sha256,encoded_sha256)):
        raise ValueError('separately frozen lowercase SHA256 values required')
    if hashlib.sha256(encoded).hexdigest()!=encoded_sha256:
        raise ValueError('compact encoded block SHA256 differs')
    padded=((raw_bytes+ROW_BYTES-1)//ROW_BYTES)*ROW_BYTES
    try:
        decoder=lzma.LZMADecompressor(format=lzma.FORMAT_XZ,memlimit=DECODE_MEMORY_BYTES)
        planes=decoder.decompress(encoded,max_length=padded+1)
    except lzma.LZMAError as error:
        raise ValueError('invalid or over-memory compact XZ block') from error
    if not decoder.eof or decoder.unused_data or len(planes)!=padded:
        raise ValueError('compact block must have exact bounded output and a single complete stream')
    delta=np.frombuffer(planes,np.uint8).reshape(ROW_BYTES,-1).T.copy()
    restored=np.add.accumulate(delta,axis=0,dtype=np.uint8).tobytes()
    if any(restored[raw_bytes:]):
        raise ValueError('compact block has nonzero original-length padding')
    raw=restored[:raw_bytes]
    if hashlib.sha256(raw).hexdigest()!=raw_sha256:
        raise ValueError('compact original raw block SHA256 differs')
    return raw
