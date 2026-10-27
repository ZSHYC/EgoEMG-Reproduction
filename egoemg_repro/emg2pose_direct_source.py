"""One deep original-data admission, then concrete readonly file bindings.

There is no archive descriptor, immutable seal or alternate dataset format.
The source owns no mappings; each original dataset owns its ordinary views.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import subprocess

from . import emg2pose_data as storage,emg2pose_runtime as native
from . import emg2pose_disk_pipeline as pipeline,emg2pose_disk_prepare as preparation

FORMAT='emg2pose_verified_direct_data_identity_v1'
_CODE=('__init__.py','download.py','emg2pose_archive.py','emg2pose_data.py','emg2pose_runtime.py',
    'emg2pose_disk_download.py','emg2pose_tar_preservation.py','emg2pose_disk_prepare.py','emg2pose_disk_pipeline.py',
    'emg2pose_memory_data.py','emg2pose_memory_archive.py','emg2pose_memory_process.py','emg2pose_direct_source.py')
_ROOT=Path(__file__).parent
_IMPORTED={n:storage._sha256(_ROOT/n) for n in _CODE}
_TOKEN=object()


def _stamp(path):
    s=storage._safe_path(path).stat()
    return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)


def _official_files(upstream):
    return sorted({*upstream.glob('egoemg/**/*.py'),*upstream.glob('config/**/*.yaml'),
        *upstream.glob('egoemg/UmeTrack/**/*.json'),upstream/'assets/per_dataset_norm_stats.json'})


def _git_bindings(upstream):
    result=subprocess.run(['git','-C',str(upstream),'rev-parse','--absolute-git-dir'],capture_output=True,text=True)
    if result.returncode:raise ValueError('pinned original Git identity unavailable')
    git=Path(result.stdout.strip());head=git/'HEAD';paths=[head]
    value=head.read_text().strip()
    if value.startswith('ref: '):
        ref=git/value[5:]
        if ref.exists():paths.append(ref)
    if (git/'packed-refs').exists():paths.append(git/'packed-refs')
    return paths


def _bound_document(path,expected):
    if not preparation.retained._digest(expected) or storage._sha256(path)!=expected:
        raise ValueError('frozen original pipeline/transaction identity differs')
    return storage._json(path)


def _proof(root,expected,csv):
    result=_bound_document(root/'completion.json',expected)
    if (result.get('format')!=pipeline.FORMAT or result.get('pipeline_complete') is not True or
            result.get('data_directory')!='data' or result.get('preparation_directory')!='prepared' or
            result.get('producer_files_sha256')!=pipeline._IMPORTED or
            any(result.get(k) is not True for k in ('all_original_bytes_reconstructable','redundant_tar_retired')) or
            any(result.get(k) is not False for k in ('model_training_performed','scientific_complete','full_paper_complete')) or
            result.get('cohort') not in ('engineering','mini','full') or
            result.get('full_corpus_complete') is not (result['cohort']=='full')):
        raise ValueError('complete original-data pipeline/scope required')
    if set(p.name for p in root.iterdir())!={'data','prepared','completion.json'}:
        raise ValueError('exact completed original source directory required')
    prepared=root/'prepared'
    p=_bound_document(prepared/'preparation.json',result['preparation_sha256'])
    retired=_bound_document(prepared/'retirement.json',result['retirement_sha256'])
    if (p.get('format')!=preparation.FORMAT or p.get('preparation_complete') is not True or
            p.get('producer_files_sha256')!=preparation._IMPORTED or p.get('cohort')!=result['cohort'] or
            p.get('full_corpus_complete') is not result['full_corpus_complete'] or
            retired.get('format')!=preparation.RETIREMENT or retired.get('redundant_tar_retired') is not True or
            retired.get('all_original_bytes_reconstructable') is not True or
            retired.get('preparation_sha256')!=result['preparation_sha256'] or
            retired.get('preservation_completion_sha256')!=p.get('preservation_completion_sha256') or
            retired.get('source_inventory_sha256')!=p.get('source_inventory_sha256') or
            any(x.get(k) is not False for x in (p,retired) for k in ('scientific_complete','full_paper_complete'))):
        raise ValueError('original preparation/retirement/scope binding differs')
    for k in ('archive_bytes','archive_sha256'):
        if retired.get(k)!=p.get(k):raise ValueError('retained original archive identity differs')
    if (not preparation.retained._integer(p.get('archive_bytes'),1024) or p['archive_bytes']%512 or
            not preparation.retained._digest(p.get('archive_sha256'))):raise ValueError('actual original archive identity required')
    if result['cohort'] in ('mini','full'):
        url,length=preparation.disk.transport._ASSETS[result['cohort']]
        if p['archive_bytes']!=length:raise ValueError('official source archive geometry differs')
        if result['cohort']=='mini' and p['archive_sha256']!=storage.MINI_SHA256:raise ValueError('official mini archive SHA differs')
    _bound_document(prepared/'source-inventory.json',p['source_inventory_sha256'])
    original=_bound_document(prepared/'download-receipt.json',result['download_receipt_sha256'])
    if (p.get('download_receipt_sha256')!=result['download_receipt_sha256'] or
            original.get('format')!=preparation.disk.FORMAT or original.get('complete_bytes') is not True or
            original.get('full_sha256')!=p['archive_sha256'] or original.get('source_bytes_received')!=p['archive_bytes'] or
            original.get('source')!=original.get('final_source') or original['source']['expected_bytes']!=p['archive_bytes']):
        raise ValueError('original completed byte acquisition binding differs')
    if result['cohort'] in ('mini','full') and (original.get('asset')!=result['cohort'] or original['source']['url']!=url):
        raise ValueError('original official byte asset differs')
    _bound_document(prepared/'preserved/completion.json',p['preservation_completion_sha256'])
    preserved=storage._json(prepared/'preserved/completion.json')
    _bound_document(prepared/'preserved/manifest.json',preserved['manifest_sha256'])
    _bound_document(root/'data/completion.json',result['conversion_completion_sha256'])
    if storage._sha256(csv)!=p.get('metadata_csv_sha256'):raise ValueError('original metadata CSV differs')
    return result,p


class VerifiedDirectSource:
    """An admitted ordinary source; content identities are returned as copies."""
    def __init__(self,root,upstream,csv,identity,stamps,completion_sha256,*,_token):
        if _token is not _TOKEN:raise TypeError('use open_source for actual direct source admission')
        self._root,self._upstream,self._csv=root,upstream,csv
        self._identity,self._stamps=deepcopy(identity),stamps
        self._completion_sha256,self._closed=completion_sha256,False

    def assert_current(self):
        if self._closed:raise ValueError('verified direct source is closed')
        try:
            if any(_stamp(p)!=v for p,v in self._stamps.items()):raise ValueError('verified direct source/code/file identity changed')
        except OSError as e:raise ValueError('verified direct source binding disappeared') from e

    @property
    def index(self):self.assert_current();return self._root/'data'
    @property
    def source_root(self):self.assert_current();return self._root
    @property
    def metadata_csv(self):self.assert_current();return self._csv
    @property
    def upstream(self):self.assert_current();return self._upstream
    @property
    def completion_sha256(self):self.assert_current();return self._completion_sha256
    @property
    def data_identity(self):self.assert_current();return deepcopy(self._identity)
    def close(self):self._closed=True
    def __enter__(self):self.assert_current();return self
    def __exit__(self,*_):self.close()


def open_source(source_root,*,expected_completion_sha256,upstream,metadata_csv):
    """Hash every actual converted field through the unchanged native validator."""
    try:
        root=storage._safe_path(source_root);csv=storage._safe_path(metadata_csv,file=True)
        if root.name.endswith('.incomplete'):raise ValueError('incomplete original source cannot enter')
        upstream,_,_=storage._official(upstream)
        if any(storage._sha256(_ROOT/n)!=h for n,h in _IMPORTED.items()):raise ValueError('imported direct-source producer changed')
        result,p=_proof(root,expected_completion_sha256,csv)
        paths=[root,root/'data',root/'prepared',root/'completion.json',csv,
            *(root/'data'/n for n in ('emg.dat','joint_angles.dat','time.dat','valid_mask.dat','completion.json','manifest.json','metadata.npz','source-receipt.json')),
            *(root/'prepared'/n for n in ('preparation.json','retirement.json','source-inventory.json','download-receipt.json','preserved/completion.json','preserved/manifest.json','preserved/tar-overhead.bin')),
            *(_ROOT/n for n in _CODE),*_official_files(upstream),*_git_bindings(upstream)]
        if (root/'prepared/mini-acquisition.json').exists():paths.append(root/'prepared/mini-acquisition.json')
        stamps={path:_stamp(path) for path in paths}
        identity=native.validate_completed_data(upstream,root/'data',root/'prepared/source-inventory.json',metadata_csv=csv)
        if (identity!=result['data_identity'] or identity['cohort']!=result['cohort'] or
                identity['full_corpus_complete'] is not result['full_corpus_complete']):raise ValueError('actual original completed fields differ from pipeline')
        if storage._sha256(upstream/'assets/per_dataset_norm_stats.json')!=native.NORM_SHA256:
            raise ValueError('original normalization asset differs')
        if any(_stamp(path)!=s for path,s in stamps.items()):raise ValueError('original source changed during deep admission')
        identity=dict(identity,format=FORMAT,source_pipeline_completion_sha256=expected_completion_sha256,
            archive_bytes=p['archive_bytes'],archive_sha256=p['archive_sha256'],
            preservation_completion_sha256=p['preservation_completion_sha256'],retirement_sha256=result['retirement_sha256'])
        return VerifiedDirectSource(root,upstream,csv,identity,stamps,expected_completion_sha256,_token=_TOKEN)
    except (OSError,KeyError,TypeError,IndexError) as e:raise ValueError('invalid completed original ordinary-data source') from e
