"""Downloader boundary tests: transfers use tiny local payloads, never Drive."""
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
import unittest
from unittest.mock import patch

try:
    download = importlib.import_module("egoemg_repro.download")
except ImportError:
    download = None


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(download, "Selective downloader has not been implemented")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.payload = b"sixteen-bytes!!!\n"
        self.asset = download.Asset("tiny.dat", "public-id", len(self.payload))

    def transfer(self, drive_id, path):
        with path.open("ab") as handle:
            handle.write(self.payload[path.stat().st_size:])

    def test_metadata_only_never_selects_bulk_arrays(self):
        assets = download.select_assets(download.read_inventory(), metadata_only=True)
        self.assertEqual({a.filename for a in assets},
                         {"manifest.json", "metadata.npz", "metadata_summary.json"})
        self.assertLess(sum(a.expected_bytes for a in assets), 40000)

    def test_default_inventory_derives_bulk_sizes_from_real_v2_fields(self):
        inventory = download.read_inventory()
        assets = download.select_assets(inventory)
        self.assertEqual(sum(a.expected_bytes for a in assets), 18392993230)
        manifest = {"fields": {"signal": {"filename": "signal.dat", "dtype": "float32", "shape": [7, 8]}}}
        self.assertEqual(download.field_byte_sizes(manifest), {"signal.dat": 224})
        self.assertFalse(any("raw" in a.filename or "video" in a.filename for a in assets))

    def test_wrong_size_never_promotes_partial_to_final(self):
        def truncated(drive_id, path):
            path.write_bytes(b"short")
        with self.assertRaisesRegex(download.DownloadError, "size"):
            download.download_one(self.asset, self.root, downloader=truncated, reserve_bytes=0)
        self.assertFalse((self.root / "tiny.dat").exists())
        self.assertEqual((self.root / "tiny.dat.part").read_bytes(), b"short")
        self.assertFalse((self.root / "acquisition_ledger.json").exists())

    def test_interrupted_transfer_resumes_and_records_sha256(self):
        part = self.root / "tiny.dat.part"
        part.write_bytes(self.payload[:4])
        entry = download.download_one(self.asset, self.root, downloader=self.transfer, reserve_bytes=0)
        self.assertEqual((self.root / "tiny.dat").read_bytes(), self.payload)
        self.assertFalse(part.exists())
        self.assertEqual(entry["sha256"], hashlib.sha256(self.payload).hexdigest())
        ledger = json.loads((self.root / "acquisition_ledger.json").read_text())
        self.assertEqual(ledger["files"]["tiny.dat"]["sha256"], entry["sha256"])

    def test_existing_wrong_size_file_is_rejected_without_overwrite(self):
        final = self.root / "tiny.dat"
        final.write_bytes(b"existing")
        with self.assertRaisesRegex(download.DownloadError, "size"):
            download.download_one(self.asset, self.root, downloader=self.transfer, reserve_bytes=0)
        self.assertEqual(final.read_bytes(), b"existing")

    def test_known_hash_rejects_same_size_corruption(self):
        asset = download.Asset("tiny.dat", "public-id", len(self.payload), sha256=hashlib.sha256(self.payload).hexdigest())
        def corrupt(drive_id, path):
            path.write_bytes(b"x" * len(self.payload))
        with self.assertRaisesRegex(download.DownloadError, "SHA256"):
            download.download_one(asset, self.root, downloader=corrupt, reserve_bytes=0)
        self.assertFalse((self.root / "tiny.dat").exists())

    def test_repeat_run_detects_existing_same_size_tampering_using_ledger(self):
        download.download_one(self.asset, self.root, downloader=self.transfer, reserve_bytes=0)
        (self.root / "tiny.dat").write_bytes(b"x" * len(self.payload))
        with self.assertRaisesRegex(download.DownloadError, "SHA256"):
            download.download_one(self.asset, self.root, downloader=self.transfer, reserve_bytes=0)

    def test_reserve_accounts_for_remaining_partial_bytes(self):
        (self.root / "tiny.dat.part").write_bytes(self.payload[:4])
        # Exactly remaining bytes plus reserve is allowed; one fewer is rejected.
        usage = lambda path: shutil._ntuple_diskusage(100, 50, len(self.payload) - 4 + 9)
        with self.assertRaisesRegex(download.DownloadError, "reserve"):
            download.ensure_capacity(self.root, [self.asset], reserve_bytes=10, disk_usage=usage)
        download.ensure_capacity(self.root, [self.asset], reserve_bytes=9, disk_usage=usage)

    def test_public_adapter_resumes_existing_part_instead_of_gdown_skipping_it(self):
        part = self.root / "tiny.dat.part"
        part.write_bytes(self.payload[:4])

        def gdown_boundary(*, url, output, quiet, resume, use_cookies, user_agent):
            self.assertEqual(url, 'https://drive.usercontent.google.com/download?id=public-id&export=download&confirm=t')
            self.assertTrue(resume)
            self.assertFalse(use_cookies)
            self.assertTrue(user_agent.startswith('python-requests/'))
            # gdown 5.2 skips an existing output before searching its temp files.
            target = Path(output)
            if target.exists():
                return output
            candidates = list(target.parent.glob(target.name + "*.part"))
            staging = candidates[0] if candidates else target.with_name(target.name + "random.part")
            offset = staging.stat().st_size if staging.exists() else 0
            with staging.open("ab") as handle:
                handle.write(self.payload[offset:])
            staging.replace(target)
            return output

        with patch("gdown.download", side_effect=gdown_boundary):
            download.public_download("public-id", part)
        self.assertEqual(part.read_bytes(), self.payload)
        self.assertEqual(list(self.root.iterdir()), [part])

    def test_recovers_gdown_partial_after_hard_interruption_before_capacity_check(self):
        interrupted = self.root / "tiny.dat.parttemporary.part"
        interrupted.write_bytes(self.payload[:4])
        usage = lambda path: shutil._ntuple_diskusage(100, 50, len(self.payload) - 4 + 10)
        download.ensure_capacity(self.root, [self.asset], reserve_bytes=10, disk_usage=usage)
        self.assertEqual((self.root / "tiny.dat.part").read_bytes(), self.payload[:4])
        self.assertFalse(interrupted.exists())

    def test_parallel_file_completion_preserves_every_receipt_entry(self):
        assets = [download.Asset(f"signal_{i}.dat", f"id-{i}", len(self.payload)) for i in range(4)]
        started = threading.Barrier(4)

        def simultaneous_transfer(drive_id, path):
            started.wait(timeout=5)
            path.write_bytes(self.payload)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(download.download_one, asset, self.root,
                                       downloader=simultaneous_transfer, reserve_bytes=0) for asset in assets]
            for future in futures:
                future.result()
        ledger = json.loads((self.root / "acquisition_ledger.json").read_text())
        self.assertEqual(set(ledger["files"]), {f"signal_{i}.dat" for i in range(4)})
        for asset in assets:
            self.assertEqual((self.root / asset.filename).read_bytes(), self.payload)
            self.assertEqual(ledger["files"][asset.filename]["sha256"], hashlib.sha256(self.payload).hexdigest())

    def test_parallel_failure_keeps_other_completed_files_and_receipts(self):
        self.assertTrue(hasattr(download, "download_many"), "parallel acquisition is missing")
        assets = [download.Asset(f"signal_{i}.dat", f"id-{i}", len(self.payload)) for i in range(4)]
        started = threading.Barrier(4)

        def one_truncated_transfer(drive_id, path):
            started.wait(timeout=5)
            path.write_bytes(b"short" if drive_id == "id-1" else self.payload)

        with self.assertRaisesRegex(download.DownloadError, "signal_1.dat"):
            download.download_many(assets, self.root, workers=4,
                                   downloader=one_truncated_transfer, reserve_bytes=0)
        ledger = json.loads((self.root / "acquisition_ledger.json").read_text())
        self.assertEqual(set(ledger["files"]), {"signal_0.dat", "signal_2.dat", "signal_3.dat"})
        self.assertFalse((self.root / "signal_1.dat").exists())
        self.assertEqual((self.root / "signal_1.dat.part").read_bytes(), b"short")

    def test_parallel_selection_rejects_duplicate_destinations(self):
        self.assertTrue(hasattr(download, "download_many"), "parallel acquisition is missing")
        with self.assertRaisesRegex(download.DownloadError, "Duplicate"):
            download.download_many([self.asset, self.asset], self.root, workers=4,
                                   downloader=self.transfer, reserve_bytes=0)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_inventory_rejects_unsafe_destination_paths(self):
        with self.assertRaises(ValueError):
            download.Asset("../outside.dat", "id", 4)

    def test_remote_manifest_disagreement_stops_before_bulk_download(self):
        manifest = {"format_version": "egoemg_v2_memmap", "fields": {
            "tiny": {"filename": "tiny.dat", "dtype": "float32", "shape": [100]}}}
        with self.assertRaisesRegex(download.DownloadError, "manifest"):
            download.validate_manifest_sizes(manifest, [self.asset])


if __name__ == "__main__":
    unittest.main()
