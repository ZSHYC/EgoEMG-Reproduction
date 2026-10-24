"""Selective public legacy release acquisition; never downloads a Drive folder.

Array sizes come from the inspected v2 dtype/shape declarations. Metadata hashes
pin that declaration; array hashes attest the local transfer, not publisher
provenance. The default signals retain official whole-episode filtering.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import threading
from typing import Callable
from urllib.parse import urlencode

MIN_FREE_BYTES = 15 * 1024**3
LEDGER_NAME = "acquisition_ledger.json"
_LEDGER_LOCK = threading.Lock()
_DTYPE_BYTES = {"bool": 1, "int8": 1, "uint8": 1, "int16": 2, "uint16": 2,
                "int32": 4, "uint32": 4, "int64": 8, "uint64": 8,
                "float32": 4, "float64": 8}


class DownloadError(RuntimeError):
    """A transfer failed validation or cannot preserve the storage reserve."""


@dataclass(frozen=True)
class Asset:
    filename: str
    drive_id: str
    expected_bytes: int
    sha256: str | None = None
    group: str = "legacy_prefiltered"

    def __post_init__(self):
        if not self.filename or Path(self.filename).name != self.filename or self.filename in {".", ".."}:
            raise ValueError("Asset filename must be a single safe path component")
        if not self.drive_id or self.expected_bytes <= 0:
            raise ValueError("Asset requires a public Drive id and positive byte length")
        if self.sha256 is not None and (len(self.sha256) != 64 or any(c not in "0123456789abcdef" for c in self.sha256)):
            raise ValueError("SHA256 must be 64 lowercase hexadecimal characters")


def read_inventory(path: Path | None = None) -> dict:
    if path is None:
        # The source checkout keeps the reviewed inventory at repository root.
        path = Path(__file__).resolve().parent.parent / "assets_manifest.json"
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def select_assets(inventory: dict, metadata_only: bool = False) -> list[Asset]:
    assets = [Asset(filename=entry["filename"], drive_id=entry["drive_id"],
                    expected_bytes=entry["expected_bytes"], sha256=entry.get("sha256"),
                    group=entry["group"])
              for entry in inventory["files"]
              if not metadata_only or entry["group"] == "metadata"]
    if len({asset.filename for asset in assets}) != len(assets):
        raise DownloadError("Duplicate destination filenames in inventory")
    return assets


def field_byte_sizes(manifest: dict) -> dict[str, int]:
    sizes = {}
    for field in manifest["fields"].values():
        shape = field["shape"]
        if not shape or any(type(n) is not int or n <= 0 for n in shape):
            raise DownloadError("Invalid manifest shape")
        try:
            sizes[field["filename"]] = math.prod(shape) * _DTYPE_BYTES[field["dtype"]]
        except KeyError as exc:
            raise DownloadError(f"Unsupported manifest dtype: {field['dtype']}") from exc
    return sizes


def validate_manifest_sizes(manifest: dict, assets: list[Asset]) -> None:
    if manifest.get("format_version") != "egoemg_v2_memmap":
        raise DownloadError("Expected the inspected egoemg_v2_memmap manifest")
    sizes = field_byte_sizes(manifest)
    for asset in assets:
        if asset.group != "metadata" and sizes.get(asset.filename) != asset.expected_bytes:
            raise DownloadError(f"Release manifest byte-size disagreement: {asset.filename}")


def _recover_partial(part: Path) -> None:
    # gdown leaves randomly named temporary files after SIGKILL; preserve the
    # exact bytes under our stable resumable name before measuring free space.
    staged = list(part.parent.glob(part.name + "*.part"))
    if not staged:
        return
    if len(staged) > 1 or part.exists():
        raise DownloadError(f"Multiple partial files require inspection: {part}")
    os.replace(staged[0], part)


def ensure_capacity(root: Path, assets: list[Asset], reserve_bytes: int = MIN_FREE_BYTES,
                    disk_usage: Callable = shutil.disk_usage) -> None:
    remaining = 0
    for asset in assets:
        final = root / asset.filename
        part = root / (asset.filename + ".part")
        if final.exists():
            if final.stat().st_size != asset.expected_bytes:
                raise DownloadError(f"Existing file size mismatch: {final}")
            continue
        _recover_partial(part)
        existing = part.stat().st_size if part.exists() else 0
        if existing > asset.expected_bytes:
            raise DownloadError(f"Partial file size exceeds expected bytes: {part}")
        remaining += asset.expected_bytes - existing
    free = disk_usage(root).free
    if free < remaining + reserve_bytes:
        raise DownloadError(f"Insufficient storage to preserve {reserve_bytes} byte reserve: "
                            f"{free} free, {remaining} remaining transfer bytes")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_ledger(root: Path) -> dict:
    path = root / LEDGER_NAME
    if not path.exists():
        return {"ledger_version": 1, "files": {}}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _atomic_json(path: Path, value: dict) -> None:
    part = path.with_name(path.name + ".part")
    with part.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(part, path)


def public_download(drive_id: str, part: Path) -> None:
    """Use gdown's public confirmation/resume protocol, without account cookies."""
    try:
        import gdown
        from requests.utils import default_user_agent
    except ImportError as exc:
        raise DownloadError("Install the pinned project dependencies (gdown is required)") from exc
    _recover_partial(part)
    if part.exists():
        # gdown skips an existing output; it resumes only its staging files.
        os.replace(part, part.with_name(part.name + ".resume.part"))
    try:
        # Use Drive's current public confirmation endpoint directly. The old
        # /uc path plus gdown's legacy browser agent can select obsolete doc-*
        # hosts which are unreachable on some networks.
        url = 'https://drive.usercontent.google.com/download?' + urlencode(
            {'id': drive_id, 'export': 'download', 'confirm': 't'})
        result = gdown.download(url=url, output=str(part), quiet=True,
                                resume=True, use_cookies=False,
                                user_agent=default_user_agent())
    finally:
        _recover_partial(part)
    if result is None:
        raise DownloadError(f"Public Google Drive download failed for {drive_id}")


def download_one(asset: Asset, root: Path, *, downloader: Callable = public_download,
                 reserve_bytes: int = MIN_FREE_BYTES) -> dict:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    ensure_capacity(root, [asset], reserve_bytes=reserve_bytes)
    final = root / asset.filename
    part = root / (asset.filename + ".part")
    with _LEDGER_LOCK:
        previous = _read_ledger(root)["files"].get(asset.filename, {})
    expected_hash = asset.sha256 or previous.get("sha256")
    if not final.exists():
        if not part.exists() or part.stat().st_size < asset.expected_bytes:
            try:
                downloader(asset.drive_id, part)
            except Exception as exc:
                raise DownloadError(f"Transfer interrupted for {asset.filename}; resume by rerunning: {exc}") from exc
        candidate = part
    else:
        candidate = final
    if not candidate.exists() or candidate.stat().st_size != asset.expected_bytes:
        actual = candidate.stat().st_size if candidate.exists() else 0
        raise DownloadError(f"Downloaded file size mismatch for {asset.filename}: "
                            f"expected {asset.expected_bytes}, got {actual}; partial retained")
    digest = _sha256(candidate)
    if expected_hash and digest != expected_hash:
        raise DownloadError(f"SHA256 mismatch for {asset.filename}; file retained for inspection")
    if candidate == part:
        os.replace(part, final)
    entry = {"drive_id": asset.drive_id, "url": f"https://drive.google.com/uc?id={asset.drive_id}",
             "bytes": asset.expected_bytes, "sha256": digest,
             "checksum_basis": "pinned_metadata" if asset.sha256 else "local_transfer",
             "verified_at": datetime.now(timezone.utc).isoformat()}
    with _LEDGER_LOCK:
        # Other files can finish while this one transfers/hashes. Merge with
        # their current receipt instead of writing the pre-transfer snapshot.
        ledger = _read_ledger(root)
        ledger["files"][asset.filename] = entry
        _atomic_json(root / LEDGER_NAME, ledger)
    return entry


def _download_logged(asset: Asset, root: Path, downloader: Callable,
                     reserve_bytes: int) -> dict:
    print(f"Starting {asset.filename} ({asset.expected_bytes} bytes)", flush=True)
    entry = download_one(asset, root, downloader=downloader, reserve_bytes=reserve_bytes)
    print(f"Validated {asset.filename}", flush=True)
    return entry


def download_many(assets: list[Asset], root: Path, *, workers: int = 1,
                  downloader: Callable = public_download,
                  reserve_bytes: int = MIN_FREE_BYTES) -> None:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if len({asset.filename for asset in assets}) != len(assets):
        raise DownloadError("Duplicate destination filenames in parallel acquisition")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    ensure_capacity(root, assets, reserve_bytes=reserve_bytes)
    def priority(asset):
        if asset.filename == "frame_split_id.dat":
            return 0
        return 1 if asset.filename.startswith("emg_") else 2
    ordered = sorted(assets, key=priority)
    errors = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = {executor.submit(_download_logged, asset, root, downloader, reserve_bytes): asset
                   for asset in ordered}
        for future in as_completed(pending):
            try:
                future.result()
            except Exception as exc:
                asset = pending[future]
                errors.append(f"{asset.filename}: {exc}")
                print(f"Failed {asset.filename}: {exc}", flush=True)
    if errors:
        raise DownloadError("; ".join(errors))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Destination for the selected v2 files")
    parser.add_argument("--metadata-only", action="store_true", help="Acquire only the three pinned small metadata files")
    parser.add_argument("--inventory", type=Path, help="Reviewed inventory JSON (default: repository assets_manifest.json)")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent bulk-file transfers (default: 1)")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    inventory = read_inventory(args.inventory)
    assets = select_assets(inventory, metadata_only=args.metadata_only)
    args.root.mkdir(parents=True, exist_ok=True)
    try:
        ensure_capacity(args.root, assets)
        for asset in assets:
            if asset.group == "metadata":
                _download_logged(asset, args.root, public_download, MIN_FREE_BYTES)
        with (args.root / "manifest.json").open(encoding="utf-8") as handle:
            validate_manifest_sizes(json.load(handle), assets)
        download_many([asset for asset in assets if asset.group != "metadata"],
                      args.root, workers=args.workers)
        ledger = _read_ledger(args.root)
        ledger.update({"source_url": inventory["source_url"], "release_format": inventory["release_format"],
                       "manifest_sha256": _sha256(args.root / "manifest.json"),
                       "selected_bytes": sum(a.expected_bytes for a in assets),
                       "preprocessing_limitation": inventory["preprocessing_limitation"]})
        _atomic_json(args.root / LEDGER_NAME, ledger)
    except (DownloadError, OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Acquisition stopped: {exc}\n")
    print(f"Validated {len(assets)} selected public assets in {args.root}")
    print(inventory["preprocessing_limitation"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
