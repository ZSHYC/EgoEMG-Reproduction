# Selective legacy acquisition

From the installed source checkout, run:

```sh
python -m egoemg_repro.download --root /path/to/legacy --metadata-only
python -m egoemg_repro.download --root /path/to/legacy --workers 4
```

The first command transfers three public metadata files totaling 33,680 bytes. `--workers` defaults to 1; use a bounded value such as 4 for concurrent independent bulk files. Metadata is always validated sequentially before bulk workers start. Bulk scheduling gives `frame_split_id.dat` priority, followed by EMG and label arrays. The second transfers the reviewed 21-file inventory in `assets_manifest.json`: 18,392,993,230 bytes total, including metadata. It acquires the two filtered EMG signals, generated finger angles and validity, wrist angles and validity, split/gesture labels, timestamps, and frame/episode/subject indices. It does not enumerate or download the full legacy Drive folder.

At least 15 GiB must remain free after the remaining selected bytes have been written. The downloader checks the aggregate requirement before starting and checks each file again before its transfer. The inventory's array lengths are derived from the inspected v2 dtype and shape declarations; the pinned downloaded manifest must agree before any array transfer starts.

Downloads use gdown's public confirmation handling with account-cookie loading disabled. Incomplete transfers remain resumable `.part` files. Rerun the same command after an interruption. Only files with exactly the expected byte count and any available pinned SHA256 are atomically promoted to their final names. Existing final files are validated again, including against their earlier local receipt when available. Wrong-sized final files, oversized partials, multiple competing partials, or checksum mismatches stop acquisition for inspection; existing files are not silently overwritten.

`acquisition_ledger.json` records each validated file's source ID, exact bytes, SHA256, and verification timestamp. Metadata checksums pin the three inspected public files. Bulk checksums record the received local bytes because the release does not publish authenticated bulk SHA256 values. The final CLI receipt also records the manifest hash, selected bytes, and preprocessing limitation. Keep this receipt with run provenance. Threaded workers in one process update the receipt under a lock and merge the latest entries. If one file fails, other scheduled files finish and retain their validated receipts before the CLI reports failure. Multiple acquisition processes into the same destination are unsupported; stop an existing process before restarting with another worker count.

The default inventory uses the official whole-episode prefiltered EMG. This can carry filtering dependencies across training/validation time partitions, even when windows do not overlap. Runs must retain that stated limitation. Strict preprocessing isolation requires separately obtaining raw EMG, filtering within assigned intervals, and discarding boundaries; raw signals and visual crops are outside this initial inventory. There is no `--include-raw` switch until their public inventory is independently reviewed.

The downloader CLI expects the repository inventory alongside the source package. For a separately packaged installation, pass its reviewed location explicitly with `--inventory /path/to/assets_manifest.json`. Dataset assets and downloaded participant metadata belong outside public version control.
