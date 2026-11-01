from pathlib import Path
import collections
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import time

REVISION = '30fe9674782f32e1e5edba98476b6ff4300132c5'
RESERVE = 200 * 1024**3


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def checked_assets(inventory):
    need(inventory['dataset'] == 'bop-benchmark/hot3d' and inventory['dataset_revision'] == REVISION, 'source dataset/revision differs')
    assets = inventory['files']
    need(len(assets) == inventory['archive_count'] == 2804, 'complete public train file count differs')
    need(sum(a['expected_bytes'] for a in assets) == inventory['archive_bytes'] == 294201743360, 'complete public train byte count differs')
    need(len({a['path'] for a in assets}) == len(assets), 'duplicate public source path')
    groups = collections.Counter()
    for asset in assets:
        need(re.fullmatch(r'train_(aria|quest3)/clip-\d{6}\.tar', asset['path']) is not None, 'unsafe or nontrain archive path')
        need(asset['role'] == 'train' and type(asset['expected_bytes']) is int and asset['expected_bytes'] > 0, 'source role/length differs')
        need(re.fullmatch('[0-9a-f]{64}', asset['lfs_sha256']) is not None, 'missing publisher LFS digest')
        need(asset['url'] == 'https://huggingface.co/datasets/bop-benchmark/hot3d/resolve/' + REVISION + '/' + asset['path'], 'source URL differs from frozen revision')
        groups[asset['path'].split('/')[0]] += 1
    need(dict(groups) == {'train_aria': 1516, 'train_quest3': 1288}, 'device partition differs')
    return assets


def verify_capacity(root, *, remaining, reserve):
    need(type(remaining) is int and remaining >= 0 and type(reserve) is int and reserve >= 0, 'invalid capacity requirement')
    if shutil.disk_usage(root).free < remaining + reserve:
        raise RuntimeError('Insufficient capacity to preserve the unchanged disk reserve')


def verify_payload(path, asset):
    path = Path(path)
    need(path.is_file() and not path.is_symlink(), 'nonregular archive payload')
    before = path.stat()
    need(before.st_size == asset['expected_bytes'], 'archive byte length differs')
    need(sha(path) == asset['lfs_sha256'], 'archive SHA256 differs from publisher LFS')
    after = path.stat()
    need((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns), 'payload generation changed while hashing')
    return {'bytes': after.st_size, 'sha256': asset['lfs_sha256']}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def exclusive(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def atomic(path, value):
    part = path.with_name(path.name + '.pending')
    with part.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(part, path)


def actor(pid):
    fields = Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()
    return {'pid': pid, 'start_ticks': int(fields[19])}


def main():
    root = Path(__file__).parent
    os.nice(10)
    config = json.loads((root / 'config.json').read_text())
    owner = actor(os.getpid())
    binary = config['native_binary']
    child = None
    current = None
    admitted = 0
    received_bytes = 0
    last_progress = 0.0

    def guard():
        need(sha(Path(__file__)) == config['source_sha256'], 'frozen receiving helper changed')
        need(sha(root / 'manifest.json') == config['manifest_sha256'], 'frozen full inventory changed')
        need(sha(binary) == config['native_binary_sha256'], 'native downloader changed')
        verify_capacity(root, remaining=0, reserve=RESERVE)

    def publish_progress():
        atomic(root / 'progress.json', {'observed_utc': now(), 'actor': owner, 'completed_archive_count': admitted, 'completed_archive_bytes': received_bytes, 'current_archive_path': current, 'current_worker': actor(child.pid) if child is not None and child.poll() is None else None, 'target_archive_count': 2804, 'target_archive_bytes': 294201743360, 'all_public_train_payloads_received': False, 'numeric_qualification_complete': False, 'actual_gpu_actions': 0, 'full_paper_complete': False})

    try:
        need(not root.is_symlink(), 'symlinked intake root')
        guard()
        assets = checked_assets(json.loads((root / 'manifest.json').read_text()))
        archives = root / 'archives'
        expected_seeds = set(config['seed_archive_paths'])
        need({str(p.relative_to(archives)) for p in archives.rglob('*') if p.is_file()} == expected_seeds, 'unowned/existing archive paths at fresh launch')
        verify_capacity(root, remaining=sum(a['expected_bytes'] for a in assets if a['path'] not in expected_seeds), reserve=RESERVE)
        exclusive(root / 'started.json', {'observed_utc': now(), 'actor': owner, 'source_sha256': config['source_sha256'], 'manifest_sha256': config['manifest_sha256'], 'native_binary_sha256': config['native_binary_sha256'], 'whole_public_train_archives': 2804, 'whole_public_train_bytes': 294201743360, 'reserve_bytes': RESERVE, 'no_automatic_retry': True, 'source_revision': REVISION, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')})
        with (root / 'receipts.jsonl').open('x') as journal, (root / 'aria2.log').open('xb') as log:
            for asset in assets:
                guard()
                current = asset['path']
                target = archives / current
                reused = current in expected_seeds
                worker = None
                if not reused:
                    need(not target.exists() and not target.with_name(target.name + '.aria2').exists(), 'existing partial payload requires independent terminal audit before resume')
                    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                    argv = [binary, '--no-conf=true', '--max-tries=1', '--connect-timeout=15', '--timeout=60', '--max-connection-per-server=1', '--split=1', '--max-download-limit=32M', '--file-allocation=none', '--allow-overwrite=false', '--auto-file-renaming=false', '--continue=false', '--check-integrity=true', '--checksum=sha-256=' + asset['lfs_sha256'], '--console-log-level=warn', '--summary-interval=60', '--stop-with-process=' + str(os.getpid()), '--dir=' + str(target.parent), '--out=' + target.name, asset['url']]
                    child = subprocess.Popen(argv, cwd=root, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    worker = actor(child.pid)
                    publish_progress()
                    while child.poll() is None:
                        guard()
                        need((root / 'aria2.log').stat().st_size <= 32 * 1024**2, 'native download log exceeded bound')
                        if time.monotonic() - last_progress >= 15:
                            publish_progress()
                            last_progress = time.monotonic()
                        time.sleep(1)
                    need(child.returncode == 0, 'native source download failed; partial files retained without automatic retry')
                identity = verify_payload(target, asset)
                need(not target.with_name(target.name + '.aria2').exists(), 'native control file remains after successful checksum')
                admitted += 1
                received_bytes += identity['bytes']
                row = {'observed_utc': now(), 'path': current, **identity, 'matches_publisher_lfs_sha256': True, 'reused_closed_qualification_archive': reused, 'native_worker': worker, 'native_exit_code': 0 if worker is not None else None}
                journal.write(json.dumps(row, sort_keys=True) + '\n')
                journal.flush()
                os.fsync(journal.fileno())
                publish_progress()
                print(json.dumps({'completed_archives': admitted, 'completed_bytes': received_bytes, 'last_path': current, 'reused': reused}), flush=True)
            need(admitted == 2804 and received_bytes == 294201743360, 'complete acquisition totals differ')
            guard()
            exclusive(root / 'completed.json', {'observed_utc': now(), 'actor': owner, 'source_sha256': config['source_sha256'], 'manifest_sha256': config['manifest_sha256'], 'archive_count': admitted, 'archive_bytes': received_bytes, 'all_publisher_lfs_sha256_exact': True, 'source_numeric_qualification_complete': False, 'original_closed_qualification_archives_reused': len(expected_seeds), 'actual_gpu_actions': 0, 'actual_model_runs': 0, 'author_mixture_reconstructed': False, 'full_paper_complete': False})
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
        exclusive(root / 'failed.json', {'observed_utc': now(), 'actor': owner, 'current_archive_path': current, 'completed_archive_count': admitted, 'completed_archive_bytes': received_bytes, 'error_type': type(error).__name__, 'message': str(error)[:400], 'owned_native_worker_exited': child is None or child.poll() is not None, 'original_payloads_and_partials_preserved': True, 'automatic_retry': False})
        raise


if __name__ == '__main__':
    main()
