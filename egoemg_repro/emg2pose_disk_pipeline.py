"""Compose original source preparation, proved retirement and native conversion.

Full callers run this finite phase under the qualified owned-stage guardian
with continuous resource/code guards. This creates data, never model results.
"""
from __future__ import annotations

from pathlib import Path

from . import emg2pose_data as storage,emg2pose_runtime as native
from . import emg2pose_disk_prepare as preparation

FORMAT='emg2pose_original_disk_source_v1'
_CODE=(Path(__file__),Path(storage.__file__),Path(native.__file__),Path(preparation.__file__))
_IMPORTED={p.name:storage._sha256(p) for p in _CODE}
_STAMPS={p:preparation.retained._stamp(p) for p in _CODE}


def _guard(parent):
    preparation._guard(parent)
    if any(preparation.retained._stamp(p)!=v for p,v in _STAMPS.items()):raise ValueError('imported original source pipeline changed')


def prepare_original_data(download,output,*,upstream,metadata_csv,cohort,expected_download_receipt_sha256,on_progress=None):
    """Execute unchanged full conversion after proved redundant-TAR retirement.

    The original mini converter reopens its legacy TAR receipt, so only mini
    converts before retirement. All original HDF5 and opaque source bytes stay.
    """
    output=storage._safe_path(output)
    if output.exists():raise FileExistsError('existing source pipeline is never overwritten')
    if not output.parent.is_dir():raise ValueError('existing source pipeline parent required')
    if cohort not in ('engineering','mini','full'):raise ValueError('explicit original source cohort required')
    _guard(output.parent);output.mkdir(mode=0o700)
    prepared=preparation.prepare_disk_source(download,output/'prepared',upstream=upstream,metadata_csv=metadata_csv,
        cohort=cohort,expected_download_receipt_sha256=expected_download_receipt_sha256,on_progress=on_progress)
    preparation_sha=storage._sha256(output/'prepared/preparation.json')
    if cohort!='mini':
        preparation.retire_managed_tar(output/'prepared',expected_preparation_sha256=preparation_sha,on_progress=on_progress)
    _guard(output)
    storage.convert_corpus(output/'prepared/source-inventory.json',output/'data',upstream=upstream)
    if cohort=='mini':
        preparation.retire_managed_tar(output/'prepared',expected_preparation_sha256=preparation_sha,on_progress=on_progress)
    _guard(output)
    identity=native.validate_completed_data(upstream,output/'data',output/'prepared/source-inventory.json',metadata_csv=metadata_csv)
    if identity['cohort']!=cohort or identity['full_corpus_complete'] is not (cohort=='full'):
        raise ValueError('original converted data cohort differs')
    if cohort=='full' and (identity['file_count']!=25253 or identity['counts']['split']!={'train':17136,'val':1950,'test':6167}):
        raise ValueError('full original25253 source and exact independent roles required')
    if (storage._sha256(output/'prepared/preparation.json')!=preparation_sha or
            storage._sha256(output/'prepared/source-inventory.json')!=prepared['source_inventory_sha256']):
        raise ValueError('bound source preparation changed during conversion/admission')
    result=dict(format=FORMAT,pipeline_complete=True,cohort=cohort,data_directory='data',preparation_directory='prepared',
        data_identity=identity,download_receipt_sha256=expected_download_receipt_sha256,
        preparation_sha256=preparation_sha,retirement_sha256=storage._sha256(output/'prepared/retirement.json'),
        conversion_completion_sha256=storage._sha256(output/'data/completion.json'),producer_files_sha256=_IMPORTED,
        all_original_bytes_reconstructable=True,redundant_tar_retired=True,full_corpus_complete=identity['full_corpus_complete'],
        model_training_performed=False,scientific_complete=False,full_paper_complete=False)
    _guard(output);preparation._publish_preparation(output/'completion.json',result);return result
