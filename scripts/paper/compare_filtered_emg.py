"""Compare complete native FFT output with a released legacy float32 EMG array.

Reads generated data from a file or stdin, without changing either array.
Metadata supplies episode boundaries. SHA256 values must be supplied explicitly.
"""

import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def stamp(path):
    s = Path(path).stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns,
            s.st_nlink, s.st_mode, s.st_uid, s.st_gid]


def actual_owner():
    import os
    pid = os.getpid()
    fields = Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, start_ticks=fields[19])


def empty_stats(channels):
    return dict(rows=0, unequal=np.zeros(channels, np.int64),
                unequal_bits=np.zeros(channels, np.int64),
                abs_sum=np.zeros(channels, np.float64),
                square_sum=np.zeros(channels, np.float64),
                maximum=np.zeros(channels, np.float64))


def accumulate(stats, generated, released):
    assert generated.shape == released.shape and generated.ndim == 2
    assert generated.dtype == released.dtype == np.dtype('<f4')
    assert np.isfinite(generated).all() and np.isfinite(released).all()
    delta = generated.astype(np.float64) - released.astype(np.float64)
    stats['rows'] += len(generated)
    stats['unequal'] += np.count_nonzero(generated != released, axis=0)
    stats['unequal_bits'] += np.count_nonzero(generated.view('<u4') != released.view('<u4'), axis=0)
    stats['abs_sum'] += np.abs(delta).sum(axis=0, dtype=np.float64)
    stats['square_sum'] += (delta * delta).sum(axis=0, dtype=np.float64)
    stats['maximum'] = np.maximum(stats['maximum'], np.max(np.abs(delta), axis=0))


def finish(stats):
    rows = stats['rows']
    assert rows > 0
    channels = len(stats['unequal'])
    channel = [dict(channel=i, values=rows, numeric_unequal=int(stats['unequal'][i]),
                    bitwise_unequal=int(stats['unequal_bits'][i]),
                    mean_abs_difference=float(stats['abs_sum'][i] / rows),
                    rms_difference=math.sqrt(float(stats['square_sum'][i] / rows)),
                    max_abs_difference=float(stats['maximum'][i]),
                    sum_abs_difference=float(stats['abs_sum'][i]),
                    sum_squared_difference=float(stats['square_sum'][i])) for i in range(channels)]
    values = rows * channels
    return dict(rows=rows, channels=channel, values=values,
                numeric_unequal=int(stats['unequal'].sum()),
                bitwise_unequal=int(stats['unequal_bits'].sum()),
                mean_abs_difference=float(stats['abs_sum'].sum() / values),
                rms_difference=math.sqrt(float(stats['square_sum'].sum() / values)),
                max_abs_difference=float(stats['maximum'].max()),
                sum_abs_difference=float(stats['abs_sum'].sum()),
                sum_squared_difference=float(stats['square_sum'].sum()))


def exact_read(stream, count):
    blocks = bytearray()
    while len(blocks) < count:
        part = stream.read(count - len(blocks))
        if not part:
            raise EOFError(f'generated stream ended after {len(blocks)} of {count} requested bytes')
        blocks.extend(part)
    return bytes(blocks)


def compare(args):
    assert sys.byteorder == 'little' and 'torch' not in sys.modules
    started = time.monotonic()
    baseline = args.baseline.resolve()
    metadata = args.metadata.resolve()
    before = {str(p): stamp(p) for p in (baseline, metadata, Path(__file__))}
    assert sha(metadata) == args.metadata_sha256
    with np.load(metadata, allow_pickle=False) as data:
        starts = data['episode_start_idx'].astype(np.int64)
        ends = data['episode_end_idx'].astype(np.int64)
    assert len(starts) == len(ends) == 41
    assert starts[0] == 0 and ends[-1] == 66161725
    assert (ends > starts).all() and np.array_equal(starts[1:], ends[:-1])
    expected_bytes = int(ends[-1]) * 8 * 4
    assert baseline.stat().st_size == expected_bytes
    generated_path = None if args.generated == '-' else Path(args.generated).resolve()
    if generated_path is not None:
        assert generated_path != baseline
        assert generated_path.stat().st_size == expected_bytes
        before[str(generated_path)] = stamp(generated_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    private = args.output.with_name(args.output.stem + '-owner.json')
    if private.exists():
        raise FileExistsError(private)
    private.write_text(json.dumps(dict(worker=actual_owner(), input_stamps=before), sort_keys=True) + '\n')
    source = sys.stdin.buffer if generated_path is None else generated_path.open('rb')
    generated_sha = hashlib.sha256()
    released_sha = hashlib.sha256()
    episodes = []
    total = empty_stats(8)
    with baseline.open('rb') as released:
        try:
            for index, (begin, end) in enumerate(zip(starts, ends)):
                episode = empty_stats(8)
                for first in range(int(begin), int(end), 65536):
                    rows = min(65536, int(end) - first)
                    generated_block = exact_read(source, rows * 8 * 4)
                    released_block = exact_read(released, rows * 8 * 4)
                    generated_sha.update(generated_block); released_sha.update(released_block)
                    a = np.frombuffer(generated_block, dtype='<f4').reshape(rows, 8)
                    b = np.frombuffer(released_block, dtype='<f4').reshape(rows, 8)
                    accumulate(episode, a, b)
                    accumulate(total, a, b)
                episodes.append(dict(episode_index=index, start_row=int(begin), end_row=int(end),
                                     **finish(episode)))
                print(json.dumps(dict(phase='episode-compared', hand=args.hand, episode_index=index,
                                      rows_compared=total['rows'])), file=sys.stderr, flush=True)
            assert source.read(1) == b'', 'generated stream has extra bytes'
            assert released.read(1) == b'', 'released file has extra bytes'
        finally:
            if generated_path is not None:
                source.close()
    assert generated_sha.hexdigest() == args.generated_sha256
    assert released_sha.hexdigest() == args.baseline_sha256
    assert all(stamp(name) == saved for name, saved in before.items())
    assert 'torch' not in sys.modules
    result = dict(format='egoemg_full_filtered_emg_numeric_comparison_v1', hand=args.hand,
                  actual_completed_episodes=len(episodes), summary=finish(total), episodes=episodes,
                  generated_sha256=generated_sha.hexdigest(), released_sha256=released_sha.hexdigest(),
                  bytes_per_array=expected_bytes, metadata_sha256=sha(metadata),
                  nine_input_attributes_unchanged=True, exact_expected_stream_length=True,
                  generated_stream_mode=generated_path is None, script_sha256=sha(__file__),
                  observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  seconds=time.monotonic() - started, python=platform.python_version(), numpy=np.__version__,
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  reduction='float64 sum over fixed 65536-row blocks; per-episode boundaries preserved',
                  difference_units='unchanged stored EMG units; no independent hardware calibration',
                  input_arrays_modified=False, fft_rerun=False, torch_imported=False,
                  optimizer_updates=0, scientific_training_performed=False,
                  current_release_or_pdf_equivalence_proven=False, full_paper_complete=False)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(hand=args.hand, completed_episodes=len(episodes),
                          summary=result['summary'], seconds=result['seconds'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--generated', required=True, help='complete generated array path, or - for stdin')
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--hand', choices=('left', 'right'), required=True)
    for name in ('baseline', 'generated', 'metadata'):
        parser.add_argument('--' + name + '-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    compare(parser.parse_args())


if __name__ == '__main__':
    main()
