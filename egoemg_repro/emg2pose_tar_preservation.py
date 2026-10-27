"""Lossless ordinary TAR storage: original files plus exact opaque overhead.

No source is deleted. A frozen completion identity verifies every byte without
consulting the original archive pathname; corpus/model admission is separate.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path,PurePosixPath
import re
import stat
import tarfile

from . import emg2pose_data as storage,emg2pose_memory_data as original
from .emg2pose_memory_archive import available_memory_bytes,RESERVE_BYTES


FORMAT='emg2pose_original_tar_preservation_v1'
MANIFEST='emg2pose_lossless_tar_tree_v1'
BLOCK=8*1024**2
_FLAGS=dict(complete_bytes=True,source_qualified=False,full_corpus_complete=False,
            scientific_complete=False,full_paper_complete=False)
_HASH=re.compile('[0-9a-f]{64}')
_CODE=(Path(__file__),Path(storage.__file__),Path(original.__file__))


def _stamp(path):
    p=storage._safe_path(path);s=p.stat()
    return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)


def _sources():return {p.name:storage._sha256(p) for p in _CODE}
_IMPORTED=_sources();_CODE_STAMPS={p:_stamp(p) for p in _CODE}


def _guard(parent,extra=0):
    if available_memory_bytes()<RESERVE_BYTES+256*1024**2:raise RuntimeError('unchanged256GiB host reserve required')
    if storage.shutil.disk_usage(parent).free<15*1024**3+extra:raise RuntimeError('unchanged15GiB disk reserve required')
    if any(_stamp(p)!=v for p,v in _CODE_STAMPS.items()):raise ValueError('imported TAR preservation producer changed')


def _sync(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def _integer(value,minimum=0):return type(value) is int and value>=minimum

def _digest(value):return type(value) is str and _HASH.fullmatch(value) is not None


def _name(name,kind):
    if type(name) is not str or kind not in ('file','directory'):raise ValueError('invalid original member name/type')
    m=tarfile.TarInfo(name);m.type=tarfile.DIRTYPE if kind=='directory' else tarfile.REGTYPE
    if original._member_name(m)!=name:raise ValueError('noncanonical original member name')
    return name


def _skeleton(members,length):
    segments=[];cursor=0;overhead=0;previous=0;names=set();files=set();dirs=set()
    for member in members:
        if set(member)!={'name','kind','offset','offset_data','bytes'}:raise ValueError('exact original member schema required')
        name=_name(member['name'],member['kind'])
        if name in names:raise ValueError('duplicate original TAR member')
        names.add(name);a,b,n=member['offset'],member['offset_data'],member['bytes']
        if (not all(_integer(x) for x in (a,b,n)) or a<previous or b<a+512 or b%512 or
                b+((n+511)//512)*512+1024>length or (member['kind']=='directory' and n)):
            raise ValueError('invalid/truncated/overlapping original TAR extent')
        previous=b+((n+511)//512)*512
        if b>cursor:
            segments.append(dict(kind='overhead',offset=cursor,bytes=b-cursor,storage_offset=overhead));overhead+=b-cursor
        if member['kind']=='file':segments.append(dict(kind='payload',offset=b,bytes=n,name=name));files.add(name)
        else:dirs.add(name)
        dirs.update(str(p) for p in PurePosixPath(name).parents if str(p)!='.')
        cursor=b+n
    if files&dirs:raise ValueError('TAR file conflicts with an original/implicit directory')
    if cursor<length:segments.append(dict(kind='overhead',offset=cursor,bytes=length-cursor,storage_offset=overhead));overhead+=length-cursor
    return segments,overhead,previous,files,dirs


def _source_guard(fd,path,stamp):
    s=os.fstat(fd);actual=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    if actual!=stamp or _stamp(path)!=stamp:raise ValueError('original TAR changed during preservation')


def _chunks(fd,offset,length,guard):
    done=0
    while done<length:
        guard();n=min(BLOCK,length-done);value=os.pread(fd,n,offset+done)
        if len(value)!=n:raise ValueError('short original/storage extent read')
        done+=n;yield value
        guard()


def _archive_members(fd,length,guard):
    members=[]
    try:
        with os.fdopen(os.dup(fd),'rb') as stream,tarfile.open(fileobj=stream,mode='r:') as archive:
            for member in archive:
                guard();name=original._member_name(member)
                if (member.sparse is not None or member.type==tarfile.GNUTYPE_SPARSE or
                        any(k.startswith('GNU.sparse') for k in member.pax_headers) or
                        not (member.isdir() or member.isreg())):raise ValueError('unsupported/sparse original TAR member')
                members.append(dict(name=name,kind='directory' if member.isdir() else 'file',
                    offset=member.offset,offset_data=member.offset_data,bytes=member.size))
        segments,overhead,end,files,dirs=_skeleton(members,length)
        if length-end<1024:raise ValueError('complete two-record TAR trailer required')
        for block in _chunks(fd,end,length-end,guard):
            if block.strip(b'\0'):raise ValueError('hidden archive or nonzero original TAR trailer')
        return members,segments,overhead,files,dirs
    except (tarfile.TarError,EOFError,OSError) as error:raise ValueError('invalid or truncated original TAR') from error


def _layout(manifest):
    if (type(manifest) is not dict or manifest.get('format')!=MANIFEST or
            not _integer(manifest.get('archive_bytes'),1024) or manifest['archive_bytes']%512 or
            not _digest(manifest.get('archive_sha256')) or type(manifest.get('members')) is not list or
            any(manifest.get(k) is not v for k,v in _FLAGS.items())):
        raise ValueError('invalid complete-byte preservation manifest/scope')
    expected,overhead,end,files,dirs=_skeleton(manifest['members'],manifest['archive_bytes'])
    if (not _integer(manifest.get('overhead_bytes')) or manifest['overhead_bytes']!=overhead or
            not _digest(manifest.get('overhead_sha256')) or
            type(manifest.get('segments')) is not list or len(manifest['segments'])!=len(expected)):
        raise ValueError('original storage coverage differs')
    for actual,want in zip(manifest['segments'],expected):
        if (set(actual)!=set(want)|{'sha256'} or
                any(type(actual.get(k)) is not type(v) or actual[k]!=v for k,v in want.items()) or
                not _digest(actual.get('sha256'))):raise ValueError('original storage range/path/order differs')
    return end,files,dirs


def _tree(root,manifest):
    end,files,dirs=_layout(manifest);contents=storage._safe_path(root/'contents')
    actual_files=set();actual_dirs=set();paths=[root,root/'manifest.json',root/'tar-overhead.bin',contents]
    if not contents.is_dir():raise ValueError('ordinary original contents directory required')
    for path in contents.rglob('*'):
        storage._safe_path(path);name=path.relative_to(contents).as_posix();paths.append(path)
        if path.is_file():actual_files.add(name)
        elif path.is_dir():actual_dirs.add(name)
        else:raise ValueError('unsupported preserved filesystem object')
    if actual_files!=files or actual_dirs!=dirs:raise ValueError('preserved original tree membership differs')
    if (root/'tar-overhead.bin').stat().st_size!=manifest['overhead_bytes']:raise ValueError('opaque overhead size differs')
    for segment in manifest['segments']:
        if segment['kind']=='payload' and (contents/segment['name']).stat().st_size!=segment['bytes']:
            raise ValueError('preserved original member length differs')
    return end,{p:_stamp(p) for p in paths}


def _stream(root,manifest,guard,*,phase,on_progress=None,writer=None):
    end,stamps=_tree(root,manifest);digest=hashlib.sha256();overhead=hashlib.sha256();done=0
    bindings={p:v for p,v in stamps.items() if p in (root,root/'manifest.json',root/'tar-overhead.bin')}
    def current():
        guard()
        if any(_stamp(p)!=v for p,v in bindings.items()):raise ValueError('preservation metadata binding changed')
    for segment in manifest['segments']:
        path=root/'tar-overhead.bin' if segment['kind']=='overhead' else root/'contents'/segment['name']
        offset=segment.get('storage_offset',0);expected=stamps[path];part=hashlib.sha256();at=0
        fd=os.open(storage._safe_path(path,file=True),os.O_RDONLY|os.O_NOFOLLOW)
        try:
            def piece_guard():
                current();_source_guard(fd,path,expected)
            for value in _chunks(fd,offset,segment['bytes'],piece_guard):
                if segment['offset']+at+len(value)>end:
                    if value[max(0,end-segment['offset']-at):].strip(b'\0'):raise ValueError('preserved trailer is not zero')
                digest.update(value);part.update(value)
                if segment['kind']=='overhead':overhead.update(value)
                if writer is not None:writer(value)
                at+=len(value);done+=len(value)
                if on_progress is not None:on_progress(dict(phase=phase,bytes_processed=done,total_bytes=manifest['archive_bytes']))
            piece_guard()
        finally:os.close(fd)
        if part.hexdigest()!=segment['sha256']:raise ValueError('preserved member/overhead extent digest differs')
    current()
    if any(_stamp(p)!=v for p,v in stamps.items()):raise ValueError('preserved source tree changed during reconstruction')
    if (done!=manifest['archive_bytes'] or digest.hexdigest()!=manifest['archive_sha256'] or
            overhead.hexdigest()!=manifest['overhead_sha256']):raise ValueError('independently reconstructed complete original TAR digest differs')
    return dict(archive_bytes=done,archive_sha256=digest.hexdigest(),regular_files=len([m for m in manifest['members'] if m['kind']=='file']),**_FLAGS)


def preserve_tar(archive,output,*,expected_bytes,expected_sha256,on_progress=None):
    """Copy actual files and all opaque TAR bytes; never modify/delete input."""
    if not _integer(expected_bytes,1024) or expected_bytes%512 or not _digest(expected_sha256):raise ValueError('frozen complete TAR byte identity required')
    if on_progress is not None and not callable(on_progress):raise ValueError('callable progress observer required')
    archive=storage._safe_path(archive,file=True);output=storage._safe_path(output);stage=output.with_name(output.name+'.incomplete')
    if output.exists() or stage.exists():raise FileExistsError('existing preservation transaction is never overwritten')
    if not output.parent.is_dir():raise ValueError('preservation parent must exist')
    _guard(output.parent,expected_bytes+32*1024**2)
    fd=os.open(archive,os.O_RDONLY|os.O_NOFOLLOW);stamp=_stamp(archive)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode) or stamp[2]!=expected_bytes:raise ValueError('actual complete original TAR extent differs')
        def current():_guard(output.parent);_source_guard(fd,archive,stamp)
        digest=hashlib.sha256();processed=0
        for value in _chunks(fd,0,expected_bytes,current):
            digest.update(value);processed+=len(value)
            if on_progress is not None:on_progress(dict(phase='hash',bytes_processed=processed,total_bytes=expected_bytes))
        if digest.hexdigest()!=expected_sha256:raise ValueError('whole original TAR SHA256 differs')
        members,segments,overhead_bytes,files,dirs=_archive_members(fd,expected_bytes,current)
        _guard(output.parent,expected_bytes+32*1024**2+len(members)*8192)
        stage.mkdir(mode=0o700);contents=stage/'contents';contents.mkdir(mode=0o700)
        try:
            for name in sorted(dirs,key=lambda p:(len(PurePosixPath(p).parts),p)):(contents/name).mkdir(mode=0o700)
            opaque=os.open(stage/'tar-overhead.bin',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
            overhead=hashlib.sha256();processed=0
            with os.fdopen(opaque,'wb') as extra:
                for segment in segments:
                    out=extra;owned=None;h=hashlib.sha256()
                    if segment['kind']=='payload':
                        target=contents/segment['name'];owned=os.fdopen(os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'wb');out=owned
                    try:
                        for value in _chunks(fd,segment['offset'],segment['bytes'],current):
                            _guard(output.parent,len(value))
                            if out.write(value)!=len(value):raise OSError('short original byte preservation write')
                            h.update(value)
                            if segment['kind']=='overhead':overhead.update(value)
                            processed+=len(value)
                            if on_progress is not None:on_progress(dict(phase='copy',bytes_processed=processed,total_bytes=expected_bytes))
                        segment['sha256']=h.hexdigest();out.flush();os.fsync(out.fileno())
                    finally:
                        if owned is not None:owned.close()
            manifest=dict(format=MANIFEST,archive_bytes=expected_bytes,archive_sha256=expected_sha256,members=members,
                segments=segments,overhead_bytes=overhead_bytes,overhead_sha256=overhead.hexdigest(),**_FLAGS)
            storage._write_json(stage/'manifest.json',manifest)
            result=_stream(stage,manifest,current,phase='verify',on_progress=on_progress)
            if _sources()!=_IMPORTED:raise ValueError('preservation producer bytes changed')
            completion=dict(format=FORMAT,manifest_sha256=storage._sha256(stage/'manifest.json'),producer_files_sha256=_IMPORTED,**_FLAGS)
            storage._write_json(stage/'completion.json',completion)
            for p in (contents,*[contents/name for name in dirs],stage):_sync(p)
            current();completion_sha=storage._sha256(stage/'completion.json')
            storage._publish_directory(stage,output);_sync(output.parent)
            return dict(completion_sha256=completion_sha,**result)
        except BaseException as error:
            if stage.exists():
                (stage/'completion.json').unlink(missing_ok=True)
                storage._write_json(stage/'status.json',dict(complete_bytes=False,error_type=type(error).__name__,error=str(error)))
            raise
    finally:os.close(fd)


def _load(output,expected):
    try:
        if not _digest(expected):raise ValueError('separately frozen preservation completion SHA required')
        root=storage._safe_path(output)
        if root.name.endswith('.incomplete') or set(p.name for p in root.iterdir())!={'contents','manifest.json','completion.json','tar-overhead.bin'}:
            raise ValueError('exact completed preservation directory required')
        _guard(root)
        if storage._sha256(root/'completion.json')!=expected:raise ValueError('frozen preservation completion differs')
        completion=storage._json(root/'completion.json');manifest=storage._json(root/'manifest.json')
        if (completion.get('format')!=FORMAT or any(completion.get(k) is not v for k,v in _FLAGS.items()) or
                completion.get('producer_files_sha256')!=_sources() or
                completion.get('manifest_sha256')!=storage._sha256(root/'manifest.json')):
            raise ValueError('preservation producer/manifest/scope binding differs')
        _layout(manifest)
        return root,manifest,_stamp(root/'completion.json')
    except (OSError,KeyError,TypeError,IndexError) as error:raise ValueError('invalid preservation index or storage') from error


def verify_preservation(output,*,expected_completion_sha256,on_progress=None):
    """Independently hash exact retained bytes without opening the original TAR."""
    root,manifest,stamp=_load(output,expected_completion_sha256)
    def current():
        _guard(root)
        if _stamp(root/'completion.json')!=stamp:raise ValueError('frozen completion changed during verification')
    result=_stream(root,manifest,current,phase='verify',on_progress=on_progress)
    return dict(completion_sha256=expected_completion_sha256,**result)


def reconstruct_tar(output,target,*,expected_completion_sha256,on_progress=None):
    """Emit the original TAR bytes into a new no-overwrite ordinary file."""
    root,manifest,stamp=_load(output,expected_completion_sha256);target=storage._safe_path(target)
    temporary=target.with_name(target.name+'.incomplete')
    if root==target or root in target.parents:raise ValueError('reconstruction target must be outside retained source')
    if target.exists() or temporary.exists():raise FileExistsError('existing reconstruction target is never overwritten')
    _guard(target.parent,manifest['archive_bytes'])
    def current():
        _guard(target.parent)
        if _stamp(root/'completion.json')!=stamp:raise ValueError('frozen completion changed during reconstruction')
    fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as stream:
        def write(value):
            _guard(target.parent,len(value))
            if stream.write(value)!=len(value):raise OSError('short original TAR reconstruction write')
        result=_stream(root,manifest,current,phase='reconstruct',on_progress=on_progress,writer=write)
        stream.flush();os.fsync(stream.fileno());current()
    os.link(temporary,target,follow_symlinks=False);temporary.unlink();_sync(target.parent)
    return dict(completion_sha256=expected_completion_sha256,**result)
