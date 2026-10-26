"""Pinned source and completed-run metadata preflight; no runtime imports or inference."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys

FORMAT='locked_legacy_preflight_v1'
UPSTREAM_COMMIT='f7cbd048b7092139852270491d3859cd472e0311'
UPSTREAM_SHA='46446093fff601875f3b7f76f9fe9ef3d898c555f1e5cc10a3bf0df6ddd3ff69'
CORE_FILES=tuple('egoemg_repro/'+n+'.py' for n in ('__init__','train','data','partitions','metrics','download'))
VISUAL_FILES=('egoemg_repro/visual.py','egoemg_repro/crops.py')
LEGACY_FILES=dict(zip(CORE_FILES+VISUAL_FILES+('egoemg_repro/audit.py','egoemg_repro/report.py'), (
    '6103cfcd4a90339b804308dcb82a86448749c728fd093ebfab546ff9bfdb3e96',
    'f40f375d11bd535bd54edb4a6d98f399e3c74ea6c9f846ba94210a203e808418',
    'de7a716c6616277f6189c94abb09803f950dd27bc1f9f7193b1a000758a88f0e',
    'ec816483e5d567cec45df77bd6823e97c201d3fd20f45b0d8004827565fdf678',
    '12d6eb170997ae78ddbf5602d1252d05a778f1357f17b28aa3b96d06a618798e',
    '5063fc0a5a0dbb1929c054a9cac86d3f90c01b375171660b6d75e2707e684e75',
    '588a18cfd0bbd106c6843fbbe6e436635784cd2b0ea7b59f3db33878ab1f084b',
    '1aa0871d6ede77d93988d455d0930e1f42470c1a92a47cfc8e49252bbfc0695e',
    '86cf0c74086f2581f430ee88d8725df10c3389379535ef9036e2d7a892b27cdd',
    'ec2855b86331863465a4500293a453203b408c6076fd8ec32d88402c5040ea42')))
REFERENCE=dict(partition_sha256='4a355040cac040e5fa4527879e862a57c3dd2e42f46e62e3eba838b65c009812',
    normalization_sha256='bdfba74af8f084160a770732767d1081bb949b9d6dcf2d0cb1b1cecc20f7e827',
    data_sha256='571a93fa21f6c1edb2f31b3948883880ec61934949c07d3ec2e67de2a4e66592')
BASELINES={}
for _key,_kind,_experiment,_step,_policy,_recipe,_config,_summary,_init in (
 ('s','emg','emgformer/egoemg_emgformer_small',635,'random','5bf43671557bac3d0f1be27e49d812de61bd6d5e9d8fa62507507e573cfad6a9','bbdd08cfedf4a52062bc68ed18754e4c6e03a361375c4a698dbd30424a46d89c','056669015739c432e116d7d35043e20834ee02e71c3d3fa26fbc1eb607036771','fe92e4683c2391e02e4f88be46e29f3c65cc3c8a7be71d397d8fb51fbb54a874'),
 ('m','emg','emgformer/egoemg_emgformer_middle',635,'random','2af1ea564a946d814866d09f836727fe655639107bce0194384677a1cb676a6c','692b6a161c270a3d888e74309706af08f1cf6ad08737f7200da8a595c377855f','93960f6cbff831c9fbec9255773a05d82d227791132529e3c523586146937abb','fe92e4683c2391e02e4f88be46e29f3c65cc3c8a7be71d397d8fb51fbb54a874'),
 ('rn18','vision','fusion/vision_resnet18',6319,'ImageNet1K_V1 backbone','ad3d43fd4757408f8357cb15769fe314ec1ef41913e03fdc3c1566b5a2ebf33c','aa46eaac0b603200f9a1b7da909ab421ddc6f8aa7578757ffbdef8a9230084f2','8014c41af5dcf7598549f52d1bdfaefa5a09ee528a61b3eb8821e74ca16cd568','6f9f666d748013345d3a5a613dfa9bb033010f97423f209185ba62232aecc6b2'),
 ('fusion','fusion','fusion/fusion_rn18_s_center_8ch',36024,'clean branches then official residual-zero initialization','e6b2cc78742caae64bac8b4698467bfcf2ebad02d20c20ebe12db399ad5a0a74','fdddb1e3ccab168ebb061ee33c26fa186440d17a2b00b11f3c951831300d54fe','6b55289cb498f4c04b0f9f42c887cb981457187ba9a56e3f8f0ed4eb768cb5a8','bc216598e1bfbfdecd18310bd924bdb1ba22ae2aadcc0f6207879c4239352f86')):
    BASELINES[_key]=dict(kind=_kind,experiment=_experiment,step=_step,policy=_policy,
        recipe_sha256=_recipe,config_sha256=_config,summary_sha256=_summary,initialization_sha256=_init)
RAW_FILES={
    "s": {
        "config.json": "c181d53f78e77a4ffbf12d527e6a4c05613af24908764ddae24455b19b396e7e",
        "recipe_source.json": "5bf43671557bac3d0f1be27e49d812de61bd6d5e9d8fa62507507e573cfad6a9",
        "partitions.json": "534a20d8710b0706362a6b26e02c7c4e06bf854e00fb9e073a8179ecc048d896",
        "normalization.json": "62734cd4766804b1e405ad0d64c172b9b6a9fe367988c6310ab5d4652b341f49",
        "provenance.json": "dedc2f4fe764d011dc61e769ede70ab6b510568b563aa818318ad9d93c23c381",
        "summary.json": "056669015739c432e116d7d35043e20834ee02e71c3d3fa26fbc1eb607036771",
        "environment.json": "4fc33923f2fa69d8e5d5f04065895e8b63207ca622e37d9d9878708b0ea61ec5",
        "run.json": "0f1df4118e14d06f14bc8c3793ac9108b6e060a598aedd77f376f6938272bc59"
    },
    "m": {
        "config.json": "4438bda6548ffc5fb2192b6c63e0ca084b44ae63b61b9ffe4d2077c6a27927b6",
        "recipe_source.json": "2af1ea564a946d814866d09f836727fe655639107bce0194384677a1cb676a6c",
        "partitions.json": "534a20d8710b0706362a6b26e02c7c4e06bf854e00fb9e073a8179ecc048d896",
        "normalization.json": "62734cd4766804b1e405ad0d64c172b9b6a9fe367988c6310ab5d4652b341f49",
        "provenance.json": "938047d38cc7b6f42efe7335909eca6eabdaadbc1d443fb47f45fc189fd92775",
        "summary.json": "93960f6cbff831c9fbec9255773a05d82d227791132529e3c523586146937abb",
        "environment.json": "4fc33923f2fa69d8e5d5f04065895e8b63207ca622e37d9d9878708b0ea61ec5",
        "run.json": "72d8fe7268a2ae86c12148c41614243770e73b64764f232d031cce5443664086"
    },
    "rn18": {
        "config.json": "2ab2c4ab20e58fdb71db7ac1606b6b2b127ee507d872dd42782b8cde38f9bac8",
        "recipe_source.json": "ad3d43fd4757408f8357cb15769fe314ec1ef41913e03fdc3c1566b5a2ebf33c",
        "partitions.json": "534a20d8710b0706362a6b26e02c7c4e06bf854e00fb9e073a8179ecc048d896",
        "normalization.json": "62734cd4766804b1e405ad0d64c172b9b6a9fe367988c6310ab5d4652b341f49",
        "provenance.json": "ae260b37ea588f188d26619c54ca2123a5c7d01de652b56b4d5057c02e4bd871",
        "summary.json": "8014c41af5dcf7598549f52d1bdfaefa5a09ee528a61b3eb8821e74ca16cd568",
        "environment.json": "093227449f34c2e572f87aa9efa2d15a265d1ca6d0e3bacef974f6acae3a6bdb",
        "run.json": "2f81eab3d0ce73a804583ca8f854ffc26e80d779678046e551838aeb7baffa17"
    },
    "fusion": {
        "config.json": "a45b6d399ec5b56edb066a2fac9c232fd212314dbf907c6de466736a5e4a75aa",
        "recipe_source.json": "e6b2cc78742caae64bac8b4698467bfcf2ebad02d20c20ebe12db399ad5a0a74",
        "partitions.json": "534a20d8710b0706362a6b26e02c7c4e06bf854e00fb9e073a8179ecc048d896",
        "normalization.json": "62734cd4766804b1e405ad0d64c172b9b6a9fe367988c6310ab5d4652b341f49",
        "provenance.json": "23fd796cf3194d20c6b580318096c1776a2adcd3b59d1a9c83e65a5e71052e31",
        "summary.json": "6b55289cb498f4c04b0f9f42c887cb981457187ba9a56e3f8f0ed4eb768cb5a8",
        "environment.json": "1c883abf43fab80840b0de157f74d2350c06bea503a31ea28fa2018dc7393d9d",
        "run.json": "4dcc19ea3400c1be60a016bab1c94353a625bac7700bf055e871fa512f2c79e3"
    }
}
SMALL_FILES={'config.json','recipe_source.json','partitions.json','normalization.json',
             'provenance.json','summary.json','environment.json','run.json'}
MAX_SMALL_BYTES=128*1024**2


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def _canonical(value):
    return _hash(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode())


def _safe(path):
    path=Path(path).absolute()
    if '..' in path.parts or any(p.is_symlink() for p in (path,*path.parents)):
        raise ValueError('unsafe input namespace')
    return path


def _stamp(value):
    return tuple(getattr(value,n) for n in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns'))


def _file_stamp(path):
    try:
        value=_safe(path).stat()
        if not stat.S_ISREG(value.st_mode):raise ValueError('source is not regular')
        return _stamp(value)
    except OSError:
        raise ValueError('required source unavailable') from None


def _read_small(path):
    path=_safe(path)
    try:
        with path.open('rb') as stream:
            before=os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size>MAX_SMALL_BYTES:
                raise ValueError('invalid small metadata source')
            raw=stream.read(MAX_SMALL_BYTES+1)
            if len(raw)!=before.st_size or _stamp(os.fstat(stream.fileno()))!=_stamp(before) or _stamp(path.stat())!=_stamp(before):
                raise ValueError('metadata source changed')
        return raw
    except OSError:
        raise ValueError('required small source unavailable') from None


def _pairs(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise ValueError('duplicate metadata key')
        result[key]=value
    return result


def read_json(path):
    try:
        return json.loads(_read_small(path),object_pairs_hook=_pairs,
                          parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfinite JSON')))
    except (UnicodeError,json.JSONDecodeError):
        raise ValueError('invalid metadata JSON') from None


def _reject_preimports():
    if any(n=='egoemg_repro' or n.startswith('egoemg_repro.') or n=='egoemg' or n.startswith('egoemg.') for n in sys.modules):
        raise ValueError('same-named runtime preimport requires a fresh process')


def _upstream(root):
    root=_safe(root)
    def git(*args):
        p=subprocess.run(['git','-C',str(root),*args],capture_output=True,text=True)
        if p.returncode:raise ValueError('official source unavailable')
        return p.stdout.strip()
    if git('rev-parse','HEAD')!=UPSTREAM_COMMIT or git('status','--porcelain','--untracked-files=all','--','egoemg','config'):
        raise ValueError('official source pin or clean policy differs')
    paths=sorted({*root.glob('egoemg/**/*.py'),*root.glob('config/**/*.yaml'),*root.glob('egoemg/UmeTrack/**/*.json')})
    digest=hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode());digest.update(_read_small(path))
    return dict(revision=UPSTREAM_COMMIT,sha256=digest.hexdigest())


def validate_source_manifest(legacy_root,upstream_root,expected=None):
    _reject_preimports()
    if expected is not None and expected!=LEGACY_FILES:raise ValueError('unapproved runtime source map')
    root=_safe(legacy_root)
    stamps={name:_file_stamp(root/name) for name in LEGACY_FILES}
    for name,sha in LEGACY_FILES.items():
        if _hash(_read_small(root/name))!=sha:raise ValueError('legacy runtime source differs')
    official=_upstream(upstream_root)
    if official!={'revision':UPSTREAM_COMMIT,'sha256':UPSTREAM_SHA}:raise ValueError('official source content differs')
    # Second read closes mutations during the separate official-source inspection.
    for name,sha in LEGACY_FILES.items():
        if _hash(_read_small(root/name))!=sha or _file_stamp(root/name)!=stamps[name]:raise ValueError('legacy runtime source changed')
    return dict(legacy_source_sha256=_canonical(LEGACY_FILES),official_source_sha256=UPSTREAM_SHA)


def _run(spec,root):
    baseline=BASELINES[spec['model']];directory=_safe(spec['run'])
    if set(spec)!= {'model','run','files_sha256'} or set(spec['files_sha256'])!=SMALL_FILES:
        raise ValueError('run metadata schema differs')
    if spec['files_sha256']!=RAW_FILES[spec['model']]:raise ValueError('approved raw metadata identity differs')
    raw={name:_read_small(directory/name) for name in SMALL_FILES}
    if any(_hash(value)!=spec['files_sha256'][name] for name,value in raw.items()):raise ValueError('raw metadata hash differs')
    values={name:json.loads(value,object_pairs_hook=_pairs,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfinite JSON')))
            for name,value in raw.items() if name!='recipe_source.json'}
    cfg=values['config.json'];prov=values['provenance.json'];summary=values['summary.json'];env=values['environment.json']
    for filename,key in [('config.json','config_sha256'),('partitions.json','partition_sha256'),('normalization.json','normalization_sha256')]:
        if _canonical(values[filename])!=prov.get(key):raise ValueError('canonical metadata hash differs')
    if any(prov.get(k)!=v for k,v in REFERENCE.items() if k!='data_sha256') or prov.get('data',{}).get('sha256')!=REFERENCE['data_sha256']:
        raise ValueError('frozen data or protocol identity differs')
    if (cfg.get('model_kind','emg')!=baseline['kind'] or cfg.get('upstream_experiment')!=baseline['experiment'] or
        _canonical(cfg)!=baseline['config_sha256'] or _hash(raw['recipe_source.json'])!=baseline['recipe_sha256']):
        raise ValueError('selected model or recipe identity differs')
    recipe=prov.get('recipe',{})
    if recipe.get('source_utf8','').encode()!=raw['recipe_source.json'] or recipe.get('effective_config')!=cfg:
        raise ValueError('raw and resolved recipe differ')
    if (summary.get('status')!='training_complete' or summary.get('test_evaluated') is not True or
        summary.get('test',{}).get('bounded') is not False or summary.get('selected_checkpoint_step')!=baseline['step'] or
        summary.get('provenance')!=prov or _hash(raw['summary.json'])!=baseline['summary_sha256']):
        raise ValueError('completed-run summary differs')
    if prov.get('source',{}).get('revision')!=UPSTREAM_COMMIT or prov.get('source',{}).get('sha256')!=UPSTREAM_SHA:
        raise ValueError('recorded official source differs')
    names=CORE_FILES+(VISUAL_FILES if baseline['kind']!='emg' else ())
    rep=prov.get('reproduction',{});digest=hashlib.sha256()
    for name in names:
        digest.update(name.encode());digest.update(_read_small(root/name))
    digest.update(b'selected_recipe');digest.update(raw['recipe_source.json'])
    if (rep.get('sha256')!=digest.hexdigest() or rep.get('selected_recipe_sha256')!=baseline['recipe_sha256'] or
        rep.get('runtime_files_sha256')!={n:LEGACY_FILES[n] for n in names}):raise ValueError('original reproduction identity differs')
    init=prov.get('initialization',{})
    if (_canonical(init)!=baseline['initialization_sha256'] or init.get('policy')!=baseline['policy'] or
        init.get('seed_before_model')!=cfg.get('seed') or env.get('initialization')!=init or
        cfg.get('precision')!='bfloat16' or env.get('precision')!='bfloat16' or env.get('matmul_precision')!=cfg.get('matmul_precision','high')):
        raise ValueError('original initialization or precision identity differs')
    for name,value in raw.items():
        if _read_small(directory/name)!=value:raise ValueError('metadata changed during preflight')
    return dict(config_sha256=baseline['config_sha256'],recipe_sha256=baseline['recipe_sha256'],
                summary_sha256=baseline['summary_sha256'],reproduction_sha256=rep['sha256'])


def preflight_manifest(manifest):
    try:
        if set(manifest)!={'format','legacy_root','upstream_root','runs'} or manifest['format']!=FORMAT:
            raise ValueError('preflight manifest schema differs')
        specs=manifest['runs']
        if not isinstance(specs,list) or len(specs)!=4 or {s['model'] for s in specs}!=set(BASELINES):
            raise ValueError('exact four selected models required')
        if _tool_source()!=TOOL_SNAPSHOT:raise ValueError('executed tool source changed after import')
        source=validate_source_manifest(manifest['legacy_root'],manifest['upstream_root'])
        paths=[_safe(s['run'])/name for s in specs for name in SMALL_FILES]
        stamps={path:_file_stamp(path) for path in paths}
        models={s['model']:_run(s,_safe(manifest['legacy_root'])) for s in specs}
        if validate_source_manifest(manifest['legacy_root'],manifest['upstream_root'])!=source:
            raise ValueError('source changed during metadata preflight')
        if any(_file_stamp(path)!=before for path,before in stamps.items()):
            raise ValueError('metadata source changed before admission ended')
        if _tool_source()!=TOOL_SNAPSHOT:raise ValueError('tool source changed during preflight')
        source=dict(source,tool_source_sha256=TOOL_SNAPSHOT)
        return dict(format=FORMAT,status='preflight_only',scope='no_inference',benchmark_eligible=False,
                    checkpoint_tensors_read=False,inference_performed=False,identities=source,models=models)
    except (KeyError,TypeError,UnicodeError,json.JSONDecodeError):
        raise ValueError('invalid preflight metadata contract') from None


def _tool_source():
    root=Path(__file__).resolve().parent
    return _canonical({name:_hash(_read_small(root/name)) for name in ('__init__.py','preflight.py','reducers.py','cli.py')})


TOOL_SNAPSHOT=_tool_source()
