"""Strict sequential recovery of reviewed public crop assets using HTTP ranges.

The original acquisition/runtime files stay frozen. This separate adapter uses
their capacity, final-size, SHA256 and receipt policy through ``download_one``.
No account authentication, cookies, proxy route or CAPTCHA handling is used.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import uuid

from .download import Asset, DownloadError, MIN_FREE_BYTES, download_one, ensure_capacity, _atomic_json

PUBLIC_ENDPOINT = 'https://drive.usercontent.google.com/download'
CHUNK_BYTES = 16 * 1024**2
MAX_ERROR_BYTES = 64 * 1024


def _looks_html(payload):
    prefix = payload.lstrip(b'\xef\xbb\xbf \t\r\n').lower()
    return prefix.startswith((b'<!doctype html', b'<html'))


def quarantine_quota_html(part):
    """Preserve only recognized small Drive quota documents, never binary data."""
    part = Path(part)
    if not part.exists():
        return None
    size = part.stat().st_size
    with part.open('rb') as handle:
        prefix = handle.read(MAX_ERROR_BYTES)
    if not _looks_html(prefix):
        return None
    text = prefix.decode('utf-8', errors='replace')
    title = re.search(r'<title[^>]*>\s*Google Drive\s*-\s*Quota exceeded\s*</title>', text, re.I)
    if size > MAX_ERROR_BYTES or not title or 'Too many users have viewed or downloaded this file recently' not in text:
        raise DownloadError('unrecognized HTML partial retained for inspection; refusing binary resume')
    digest = hashlib.sha256(prefix).hexdigest()
    quarantine = part.parent / '.quota-quarantine'
    quarantine.mkdir(exist_ok=True)
    destination = quarantine / (part.name + '.' + digest + '.' + uuid.uuid4().hex + '.html')
    part.replace(destination)
    receipt = {'format': 'egoemg_quarantined_error_v1', 'original_partial': part.name,
               'bytes': size, 'sha256': digest, 'diagnosis': 'google_drive_quota_html',
               'quarantined_at': datetime.now(timezone.utc).isoformat()}
    # If receipt publication fails, stop; preserved bytes are never redownloaded
    # over or silently deleted, and the digest also remains in their filename.
    _atomic_json(destination.with_suffix('.json'), receipt)
    return {'path': str(destination), **receipt}


def _capacity(root, assets):
    # Explicit runtime disk_usage keeps the check current at every chunk. The
    # frozen helper also recovers a single legitimate gdown staging partial.
    ensure_capacity(Path(root), assets, reserve_bytes=MIN_FREE_BYTES, disk_usage=shutil.disk_usage)


class StrictRangeDownloader:
    """Validate a complete bounded chunk in memory before appending any bytes."""

    def __init__(self, asset: Asset, *, chunk_bytes=CHUNK_BYTES, session_factory=None):
        if type(chunk_bytes) is not int or not 1 <= chunk_bytes <= CHUNK_BYTES:
            raise ValueError('chunk_bytes must be positive and at most 16 MiB')
        self.asset, self.chunk_bytes, self.session_factory = asset, chunk_bytes, session_factory

    def __call__(self, drive_id, part):
        import requests
        if drive_id != self.asset.drive_id:
            raise DownloadError('range transport asset identity mismatch')
        part = Path(part)
        quarantine_quota_html(part)
        start = part.stat().st_size if part.exists() else 0
        if start > self.asset.expected_bytes:
            raise DownloadError('partial file exceeds reviewed asset size')
        session_factory = self.session_factory or requests.Session
        while start < self.asset.expected_bytes:
            _capacity(part.parent, [self.asset])
            end = min(start + self.chunk_bytes, self.asset.expected_bytes) - 1
            expected = end - start + 1
            with session_factory() as session:
                # Fresh per chunk: no ambient netrc, environment proxies, account
                # cookies, redirect-following or response-cookie reuse.
                session.trust_env = False
                session.auth = None
                session.cookies.clear()
                session.proxies.clear()
                with session.get(PUBLIC_ENDPOINT,
                                 params={'id': drive_id, 'export': 'download', 'confirm': 't'},
                                 headers={'User-Agent': requests.utils.default_user_agent(),
                                          'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'},
                                 stream=True, timeout=(15, 60), allow_redirects=False) as response:
                    if response.status_code != 206:
                        raise DownloadError('public endpoint did not return HTTP 206; stop and back off')
                    kind = response.headers.get('Content-Type', '').split(';', 1)[0].strip().lower()
                    if kind != 'application/octet-stream':
                        raise DownloadError('public range returned a nonbinary content type; no bytes appended')
                    span = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
                    if not span or tuple(map(int, span.groups())) != (start, end, self.asset.expected_bytes):
                        raise DownloadError('public Content-Range differs from requested range or reviewed total')
                    if response.headers.get('Content-Length') != str(expected):
                        raise DownloadError('public range Content-Length differs from exact requested size')
                    if response.headers.get('Content-Encoding', 'identity').lower() != 'identity':
                        raise DownloadError('encoded range responses cannot preserve exact byte offsets')
                    body = bytearray()
                    for block in response.iter_content(chunk_size=64 * 1024):
                        if len(body) + len(block) > expected:
                            raise DownloadError('public range body exceeds its declared byte size')
                        body.extend(block)
                    if len(body) != expected:
                        raise DownloadError('public range body is shorter than its declared byte size')
                    if _looks_html(body):
                        raise DownloadError('HTML response masqueraded as binary data; no bytes appended')
            # Never credit uncommitted in-memory bytes as available disk space.
            _capacity(part.parent, [self.asset])
            current = part.stat().st_size if part.exists() else 0
            if current != start:
                raise DownloadError('partial changed concurrently; stop competing acquisition processes')
            with part.open('ab') as handle:
                handle.write(body)
                handle.flush()
                os.fsync(handle.fileno())
            start = end + 1


def validate_lmdb(path):
    """Open the fully received crop file read-only and close all handles."""
    import lmdb
    environment = lmdb.open(str(path), subdir=False, readonly=True, create=False,
                            lock=False, readahead=False)
    try:
        with environment.begin(write=False) as transaction:
            return dict(transaction.stat())
    finally:
        environment.close()


def recover_assets(inventory, root, *, episodes=None, chunk_bytes=CHUNK_BYTES, session_factory=None):
    """Reuse the official reviewed crop selection and frozen final receipt gate."""
    from .crops import select_crop_assets
    assets, _alignment_fields = select_crop_assets(inventory, episodes=episodes)
    if not assets or len({asset.filename for asset in assets}) != len(assets):
        raise DownloadError('crop recovery requires a nonempty, unique reviewed selection')
    if any(not asset.filename.endswith(('.data.mdb', '.lmdb')) for asset in assets):
        raise DownloadError('this recovery helper accepts crop LMDB assets only')
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    _capacity(root, assets)
    entries, quarantined = 0, 0
    for index, asset in enumerate(assets, 1):  # Sequential; stop on first failure.
        if quarantine_quota_html(root / (asset.filename + '.part')) is not None:
            quarantined += 1
        _capacity(root, [asset])
        adapter = StrictRangeDownloader(asset, chunk_bytes=chunk_bytes, session_factory=session_factory)
        download_one(asset, root, downloader=adapter, reserve_bytes=MIN_FREE_BYTES)
        # The frozen acquisition receipt attests size/SHA256, not LMDB schema.
        # Only report success after the independent readonly database check.
        stats = validate_lmdb(root / asset.filename)
        entries += int(stats['entries'])
        print(json.dumps({'event': 'validated_crop', 'completed_files': index,
                          'selected_files': len(assets)}), flush=True)
    return {'validated_files': len(assets), 'lmdb_entries': entries, 'quarantined_error_pages': quarantined,
            'network_policy': 'direct-public; no ambient authentication, cookies or proxies',
            'chunk_bytes': chunk_bytes, 'workers': 1}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', required=True, help='Reviewed official crop inventory JSON')
    parser.add_argument('--root', required=True, help='Existing crop destination; one acquisition process only')
    parser.add_argument('--episodes', nargs='+', help='Exact reviewed episode identifiers to recover')
    parser.add_argument('--direct-public', action='store_true', required=True,
                        help='Explicitly use direct unauthenticated public HTTPS; ignore ambient proxies/netrc/cookies')
    args = parser.parse_args(argv)
    try:
        result = recover_assets(json.loads(Path(args.inventory).read_text()), args.root, episodes=args.episodes)
    except Exception as exc:
        parser.exit(1, f'range recovery stopped; retained verified chunks and error evidence: {exc}\n')
    print(json.dumps(result, sort_keys=True))
    return result


if __name__ == '__main__':
    main()
