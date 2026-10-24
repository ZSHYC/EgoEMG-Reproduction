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

## Strict recovery of reviewed crop assets

If the public Drive endpoint returns a tiny HTML quota page, the original size gate rejects it and retains the `.part` for inspection. Such a page is **not a resumable binary prefix**. A separate [`range_download.py`](../egoemg_repro/range_download.py) adapter can recover the same reviewed crop inventory without changing any frozen training, data, crop or original acquisition module:

```sh
python -m egoemg_repro.range_download \
  --inventory /path/to/assets_crops_manifest.json \
  --root /path/to/crops \
  --direct-public
```

Use `--episodes episode_000023 episode_000024` to select exact inventory entries when recovering only missing assets. Otherwise all reviewed crop entries are checked in order. Stop any existing acquisition process first; concurrent processes writing the same crop directory are unsupported. This command selects existing official crop IDs only; it does not enumerate a Drive folder, choose a different release/cohort, or acquire alignment fields. Install the project's vision dependencies for the final read-only LMDB check.

`--direct-public` is an explicit network policy: fresh unauthenticated Requests sessions connect directly to the existing official public `drive.usercontent.google.com` endpoint, using the current Requests user agent. Environment proxies and netrc authentication are ignored; no account cookies are loaded, response cookies are not reused across chunks, redirects are not followed, and there is no account/CAPTCHA handling. This is ordinary public HTTPS retrieval and does not grant access to restricted files. If direct HTTPS is unavailable, the helper stops; it does not change system network settings or silently choose another route.

Transfers are sequential, with at most 16 MiB per HTTP range request. Every chunk must return HTTP 206, `application/octet-stream`, an exact `Content-Range` start/end/full size, matching `Content-Length`, identity encoding and the complete expected body. A full bounded chunk is checked in memory before any of it is appended. HTTP 200 quota HTML, redirects, permission errors, mismatched ranges/totals, truncated bodies and HTML disguised as binary stop the run. Previously completed binary chunks remain resumable. There is no automatic retry loop: respect quota/backoff and inspect the error before a later explicit retry. [Google's official help](https://support.google.com/drive/answer/2423534?hl=en) describes high-volume restrictions as temporary, typically resetting within 24 hours; a successful small range probe alone does not prove a full transfer will complete.

Only small HTML partials confidently recognized by both the Google Drive quota title and its specific high-volume message are moved to `.quota-quarantine/`. Their exact bytes and SHA256/length/timestamp receipt are preserved before requesting fresh payload bytes. Unrecognized HTML remains in place and stops recovery. Genuine binary partials are not quarantined or rewritten; recovery continues at their existing byte offset. Preserving a quota page still consumes disk space: it receives no credit toward remaining payload bytes.

The helper reuses `Asset`, `download_one`, the existing 15 GiB storage reserve, exact final byte count, known/past SHA256 validation, atomic promotion and `acquisition_ledger.json`. It rechecks capacity for the selection and before each chunk request/write. After the complete file passes the byte/hash gate, it additionally opens that LMDB read-only, inspects a read transaction's statistics and closes the handle. A schema failure stops the helper even though the original transfer receipt remains available; that receipt alone is not proof of LMDB or center-key coverage. Actual center coverage and final experiment provenance still require the separate checks described in the [vision guide](vision.md) and [metric audit](metric-audit.md).

This recovery adapter and its local fixtures are not evidence that the full 41-episode crop acquisition has finished. Publish completion only after every required real payload, receipt, database and center population has been verified. Do not replace quota-limited legacy assets with a newer preview or another cohort without a separately approved identity change.
