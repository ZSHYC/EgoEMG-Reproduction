"""Retain every original TAR byte, real payload files and independent rebuilds."""
from copy import deepcopy
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import tarfile
from types import SimpleNamespace

import pytest


def api():return importlib.import_module('egoemg_repro.emg2pose_tar_preservation')
def sha(body):return hashlib.sha256(body).hexdigest()


def tar_bytes(entries,*,format=tarfile.PAX_FORMAT):
    stream=io.BytesIO()
    with tarfile.open(fileobj=stream,mode='w',format=format) as archive:
        for name,kind,payload in entries:
            member=tarfile.TarInfo(name);member.type=kind;member.size=len(payload);member.uid=1037
            member.gid=123;member.mtime=1234567;member.mode=0o777
            if kind in (tarfile.SYMTYPE,tarfile.LNKTYPE):member.linkname='outside'
            archive.addfile(member,io.BytesIO(payload) if member.isreg() else None)
    return stream.getvalue()


@pytest.fixture
def case(tmp_path,monkeypatch):
    module=api();monkeypatch.setattr(module,'available_memory_bytes',lambda:2*1024**4)
    monkeypatch.setattr(module.storage.shutil,'disk_usage',lambda _:SimpleNamespace(free=64*1024**3))
    raw=bytearray(tar_bytes([('corpus',tarfile.DIRTYPE,b''),('corpus/a.hdf5',tarfile.REGTYPE,b'original\0\xff'),
        ('corpus/empty',tarfile.REGTYPE,b''),('corpus/b.bin',tarfile.REGTYPE,bytes(range(251))*5)]))
    # Padding is legal opaque source information, not necessarily zero.
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:') as archive:m=archive.getmember('corpus/a.hdf5')
    raw[m.offset_data+m.size:m.offset_data+512]=b'P'*(512-m.size)
    raw=bytes(raw);source=tmp_path/'source.tar';source.write_bytes(raw)
    return dict(root=tmp_path,archive=source,raw=raw,output=tmp_path/'preserved')


def preserve(case,**changes):
    return api().preserve_tar(case['archive'],changes.pop('output',case['output']),
        expected_bytes=changes.pop('expected_bytes',len(case['raw'])),expected_sha256=changes.pop('expected_sha256',sha(case['raw'])),**changes)


def verify(case,receipt,**changes):
    return api().verify_preservation(changes.pop('output',case['output']),
        expected_completion_sha256=changes.pop('expected',receipt['completion_sha256']),**changes)


def test_exact_all_bytes_new_real_files_and_independent_rebuild_without_original(case):
    receipt=preserve(case);root=case['output']
    assert sorted(x.name for x in root.iterdir())==['completion.json','contents','manifest.json','tar-overhead.bin']
    assert (root/'contents/corpus/a.hdf5').read_bytes()==b'original\0\xff'
    assert (root/'contents/corpus/empty').read_bytes()==b''
    assert (root/'contents/corpus/b.bin').read_bytes()==bytes(range(251))*5
    assert (root/'contents/corpus/a.hdf5').stat().st_mode&0o777==0o600
    assert receipt['completion_sha256']==sha((root/'completion.json').read_bytes())
    case['archive'].unlink()  # Tiny generated fixture only; actual sources stay.
    value=verify(case,receipt)
    assert value['archive_bytes']==len(case['raw']) and value['archive_sha256']==sha(case['raw'])
    assert value['regular_files']==3 and value['complete_bytes'] is True
    assert value['source_qualified'] is False and value['scientific_complete'] is False and value['full_paper_complete'] is False
    target=case['root']/'rebuilt.tar'
    result=api().reconstruct_tar(root,target,expected_completion_sha256=receipt['completion_sha256'])
    assert result['archive_sha256']==sha(case['raw']) and target.read_bytes()==case['raw']
    with tarfile.open(target,'r:') as archive:
        assert archive.getmember('corpus/a.hdf5').uid==1037 and archive.getmember('corpus/a.hdf5').mode==0o777


@pytest.mark.parametrize('format',[tarfile.PAX_FORMAT,tarfile.GNU_FORMAT])
def test_long_unicode_extended_headers_and_all_padding_are_retained(case,format):
    name='nested/'*35+'原始-手.hdf5'
    raw=tar_bytes([(name,tarfile.REGTYPE,b'bits\0\x80')],format=format)
    case['archive'].write_bytes(raw);case['raw']=raw;receipt=preserve(case)
    target=case['root']/'extended.tar';api().reconstruct_tar(case['output'],target,expected_completion_sha256=receipt['completion_sha256'])
    assert target.read_bytes()==raw and (case['output']/'contents'/name).read_bytes()==b'bits\0\x80'


@pytest.mark.parametrize('name',['/outside','../outside','corpus/../outside','corpus//x','corpus\\x'])
def test_unsafe_original_member_names_refuse_before_publication(case,name):
    case['raw']=tar_bytes([(name,tarfile.REGTYPE,b'x')]);case['archive'].write_bytes(case['raw'])
    with pytest.raises(ValueError):preserve(case)
    assert not case['output'].exists()


@pytest.mark.parametrize('kind',[tarfile.SYMTYPE,tarfile.LNKTYPE,tarfile.CHRTYPE,tarfile.FIFOTYPE,tarfile.GNUTYPE_SPARSE])
def test_links_devices_and_sparse_members_are_refused(case,kind):
    case['raw']=tar_bytes([('corpus/x',kind,b'')]);case['archive'].write_bytes(case['raw'])
    with pytest.raises(ValueError):preserve(case)
    assert not case['output'].exists()


@pytest.mark.parametrize('damage',['duplicate','hidden_archive','nonzero_trailer','truncated'])
def test_ambiguous_or_unaccounted_original_tar_bytes_are_refused(case,damage):
    if damage=='duplicate':raw=tar_bytes([('same',tarfile.REGTYPE,b'a'),('same',tarfile.REGTYPE,b'b')])
    elif damage=='hidden_archive':raw=case['raw']+tar_bytes([('hidden',tarfile.REGTYPE,b'x')])
    elif damage=='nonzero_trailer':raw=case['raw'][:-1]+b'x'
    else:raw=case['raw'][:1536]
    case['raw']=raw;case['archive'].write_bytes(raw)
    with pytest.raises(ValueError):preserve(case)
    assert not case['output'].exists()


@pytest.mark.parametrize('changes',[{'expected_sha256':'0'*64},{'expected_sha256':None},
    {'expected_bytes':True},{'expected_bytes':1023},{'expected_bytes':1024}])
def test_unfrozen_or_wrong_complete_source_identity_refuses_before_output(case,changes):
    with pytest.raises(ValueError):preserve(case,**changes)
    assert not case['output'].exists()


def test_existing_output_or_failed_staging_and_reconstruction_target_are_never_overwritten(case):
    receipt=preserve(case);before=(case['output']/'completion.json').read_bytes()
    with pytest.raises(FileExistsError):preserve(case)
    assert (case['output']/'completion.json').read_bytes()==before
    output=case['root']/'other';output.with_name('other.incomplete').mkdir()
    with pytest.raises(FileExistsError):preserve(case,output=output)
    target=case['root']/'existing.tar';target.write_bytes(b'keep')
    with pytest.raises(FileExistsError):api().reconstruct_tar(case['output'],target,expected_completion_sha256=receipt['completion_sha256'])
    assert target.read_bytes()==b'keep'


@pytest.mark.parametrize('damage',['completion','manifest','overhead','payload','extra_file','symlink'])
def test_frozen_proof_or_any_original_storage_mutation_is_refused(case,damage):
    receipt=preserve(case);root=case['output']
    if damage=='extra_file':(root/'contents/extra').write_bytes(b'x')
    elif damage=='symlink':
        p=root/'contents/corpus/a.hdf5';p.unlink();p.symlink_to(case['archive'])
    else:
        p=root/{'completion':'completion.json','manifest':'manifest.json','overhead':'tar-overhead.bin','payload':'contents/corpus/a.hdf5'}[damage]
        with p.open('ab') as f:f.write(b'x')
    with pytest.raises(ValueError):verify(case,receipt)


@pytest.mark.parametrize('damage',['gap','overlap','reorder','unsafe','science'])
def test_resealed_metadata_cannot_bypass_geometry_path_or_scope_rules(case,damage):
    receipt=preserve(case);root=case['output'];manifest=json.loads((root/'manifest.json').read_text())
    if damage=='gap':manifest['segments'][0]['offset']+=1
    elif damage=='overlap':manifest['segments'][1]['offset']-=1
    elif damage=='reorder':manifest['segments'].reverse()
    elif damage=='unsafe':next(s for s in manifest['segments'] if s['kind']=='payload')['name']='../outside'
    else:manifest['scientific_complete']=True
    (root/'manifest.json').write_text(json.dumps(manifest))
    completion=json.loads((root/'completion.json').read_text());completion['manifest_sha256']=sha((root/'manifest.json').read_bytes())
    (root/'completion.json').write_text(json.dumps(completion));receipt['completion_sha256']=sha((root/'completion.json').read_bytes())
    with pytest.raises(ValueError):verify(case,receipt)


@pytest.mark.parametrize('resource',['memory','disk'])
def test_actual_resource_thresholds_refuse_before_copy_allocation(case,monkeypatch,resource):
    if resource=='memory':monkeypatch.setattr(api(),'available_memory_bytes',lambda:0)
    else:monkeypatch.setattr(api().storage.shutil,'disk_usage',lambda _:SimpleNamespace(free=0))
    with pytest.raises(RuntimeError):preserve(case)
    assert not case['output'].exists()


def test_pressure_during_copy_retains_incomplete_without_complete_marker(case,monkeypatch):
    def changed(event):
        if event['phase']=='copy':monkeypatch.setattr(api(),'available_memory_bytes',lambda:0)
    with pytest.raises(RuntimeError):preserve(case,on_progress=changed)
    assert not case['output'].exists()
    stage=case['output'].with_name('preserved.incomplete')
    assert stage.exists() and not (stage/'completion.json').exists()
    assert json.loads((stage/'status.json').read_text())['complete_bytes'] is False
    assert case['archive'].read_bytes()==case['raw']


def test_changed_original_source_is_detected_even_if_original_bytes_are_restored(case):
    modified=[]
    def changed(event):
        if event['phase']=='copy' and not modified:
            modified.append(True);case['archive'].write_bytes(case['raw'])
    with pytest.raises(ValueError):preserve(case,on_progress=changed)
    assert not case['output'].exists()


def test_publication_failure_never_leaves_a_complete_staging_claim(case,monkeypatch):
    def fail(*_):raise OSError('publication interrupted')
    monkeypatch.setattr(api().storage,'_publish_directory',fail)
    with pytest.raises(OSError,match='publication interrupted'):preserve(case)
    stage=case['output'].with_name('preserved.incomplete')
    assert not case['output'].exists() and not (stage/'completion.json').exists()
    assert json.loads((stage/'status.json').read_text())['complete_bytes'] is False


def test_changed_member_during_reconstruction_retains_incomplete_and_no_target(case):
    receipt=preserve(case);changed=[];target=case['root']/'changed.tar'
    def mutate(event):
        if event['phase']=='reconstruct' and not changed:
            changed.append(True);(case['output']/'contents/corpus/b.bin').write_bytes(b'changed')
    with pytest.raises(ValueError):api().reconstruct_tar(case['output'],target,
        expected_completion_sha256=receipt['completion_sha256'],on_progress=mutate)
    assert not target.exists() and target.with_name('changed.tar.incomplete').exists()


def test_file_and_implicit_directory_collision_is_refused_before_copy(case):
    case['raw']=tar_bytes([('same',tarfile.REGTYPE,b'x'),('same/child',tarfile.REGTYPE,b'y')])
    case['archive'].write_bytes(case['raw'])
    with pytest.raises(ValueError):preserve(case)
    assert not case['output'].exists() and not case['output'].with_name('preserved.incomplete').exists()


def test_racing_reconstruction_target_is_not_overwritten(case):
    receipt=preserve(case);target=case['root']/'raced.tar';created=[]
    def racing(event):
        if event['phase']=='reconstruct' and event['bytes_processed']==len(case['raw']) and not created:
            target.write_bytes(b'other-owned-result');created.append(True)
    with pytest.raises(FileExistsError):api().reconstruct_tar(case['output'],target,
        expected_completion_sha256=receipt['completion_sha256'],on_progress=racing)
    assert target.read_bytes()==b'other-owned-result'
    assert target.with_name('raced.tar.incomplete').read_bytes()==case['raw']


@pytest.mark.parametrize('field,value',[('offset',False),('offset',0.0),('storage_offset',False),
    ('storage_offset',0.0),('bytes',512.0),('overhead_bytes',None)])
def test_resealed_equal_valued_noninteger_geometry_is_rejected(case,field,value):
    receipt=preserve(case);root=case['output'];manifest=json.loads((root/'manifest.json').read_text())
    if field=='overhead_bytes':manifest[field]=float(manifest[field])
    else:manifest['segments'][0][field]=value
    (root/'manifest.json').write_text(json.dumps(manifest))
    completion=json.loads((root/'completion.json').read_text());completion['manifest_sha256']=sha((root/'manifest.json').read_bytes())
    (root/'completion.json').write_text(json.dumps(completion));receipt['completion_sha256']=sha((root/'completion.json').read_bytes())
    with pytest.raises(ValueError):verify(case,receipt)
