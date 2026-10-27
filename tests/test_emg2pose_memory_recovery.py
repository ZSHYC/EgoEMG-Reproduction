"""Real native checkpoint recovery, frozen inputs and owned checkpoint writes."""
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('pytorch_lightning')
pytest.importorskip('hydra')
pytest.importorskip('tensorboard')
from test_emg2pose_memory_runtime import corpus, memory, UPSTREAM
from egoemg_repro import emg2pose_data as storage


def api():
    return importlib.import_module('egoemg_repro.emg2pose_memory_recovery')


@pytest.fixture(scope='module')
def prefix(memory, tmp_path_factory):
    from egoemg_repro.emg2pose_memory_index import open_source
    output = tmp_path_factory.mktemp('recovery-owned')/'run'
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv('CUDA_VISIBLE_DEVICES','')
        with open_source(memory.fd,memory.index,
                         expected_completion_sha256=memory.receipt['completion_sha256'],
                         upstream=UPSTREAM,metadata_csv=memory.corpus.metadata_csv) as source:
            result = api().run_recovery_phase(UPSTREAM,source,memory.corpus.metadata_csv,
                'small',output,phase='prefix',device='cpu')
            assert source.fileno() >= 0
    return output,result


def test_real_native_prefix_keeps_official_scheduler_callbacks_and_complete_val(prefix):
    output,result = prefix
    assert result['phase'] == 'prefix'
    assert result['actual_optimizer_updates'] == 2 and result['global_step'] == 2
    assert len(result['validations']) == 2
    assert all(v['samples'] == 1 for v in result['validations'])
    assert result['profile']['batch_size'] == 1 and result['profile']['workers'] == 0
    assert result['profile']['scientific_batch_size'] == 320
    assert result['profile']['scientific_workers'] == 12
    assert result['profile']['scheduler_epochs'] == 250
    assert result['profile']['logger'] == 'pytorch_lightning.loggers.TensorBoardLogger'
    assert result['scientific_training_complete'] is False
    assert result['full_paper_complete'] is False
    checkpoint = torch.load(output/'prefix.ckpt',map_location='cpu',weights_only=False)
    assert checkpoint['lr_schedulers'][0]['T_max'] == 250
    assert checkpoint['global_step'] == 2
    states = checkpoint['callbacks']
    assert not any(k.startswith('EarlyStopping') for k in states)
    ck = [v for k,v in states.items() if k.startswith('ModelCheckpoint{')]
    assert len(ck) == 1 and len(ck[0]['best_k_models']) == 2
    assert Path(ck[0]['last_model_path']).is_file()
    assert checkpoint['emg2pose_recovery']['runtime_identity'] == result['runtime_identity']
    assert not (output/'completion.json').exists()


def command(memory,output,phase,sha):
    return [sys.executable,'-m','egoemg_repro.emg2pose_memory_recovery',
        '--upstream',str(UPSTREAM),'--index',str(memory.index),
        '--expected-index-completion-sha256',memory.receipt['completion_sha256'],
        '--metadata-csv',str(memory.corpus.metadata_csv),'--size','small',
        '--output',str(output),'--phase',phase,'--device','cpu',
        '--expected-prefix-sha256',sha]


def test_new_process_native_restore_replays_every_window_and_executes_one_update(memory,prefix):
    output,before = prefix
    sha = storage._sha256(output/'prefix.json')
    env = dict(os.environ,CUDA_VISIBLE_DEVICES='',EGOEMG_STAGE_FD_ARCHIVE=str(memory.fd),
               OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    result = subprocess.run(command(memory,output,'restore',sha),env=env,
        pass_fds=(memory.fd,),capture_output=True,text=True,timeout=180)
    assert result.returncode == 0, result.stdout[-3000:]+result.stderr[-6000:]
    restored = json.loads((output/'restore.json').read_text())
    assert restored['process'] != before['process']
    assert restored['actual_optimizer_updates'] == 1 and restored['global_step'] == 3
    assert restored['restore_matches'] == dict.fromkeys(
        ('model','optimizer','scheduler','callbacks','loops','rng'),True)
    assert restored['fresh_validation'] == before['validations'][-1]
    assert restored['parameters_changed'] is True
    assert restored['uninterrupted_trajectory_qualified'] is False
    assert restored['scientific_training_complete'] is False
    assert storage._sha256(output/'prefix.json') == sha
    assert storage._sha256(output/'prefix.ckpt') == before['checkpoint_sha256']


@pytest.mark.parametrize('damage',['receipt','checkpoint','identity','missing_sha','same_process'])
def test_restore_refuses_changed_or_unfrozen_prefix_before_any_training(memory,prefix,tmp_path,damage):
    from egoemg_repro.emg2pose_memory_index import open_source
    output,before = prefix
    owned = tmp_path/'owned'
    owned.mkdir()
    for name in ('prefix.json','prefix.ckpt'):shutil.copy2(output/name,owned/name)
    packet = json.loads((owned/'prefix.json').read_text())
    if damage == 'identity':
        packet['runtime_identity']['size'] = 'large'
        (owned/'prefix.json').write_text(json.dumps(packet))
    sha = storage._sha256(owned/'prefix.json')
    if damage == 'receipt':(owned/'prefix.json').write_text('{}')
    elif damage == 'checkpoint':
        with (owned/'prefix.ckpt').open('ab') as h:h.write(b'changed')
    elif damage == 'missing_sha':sha = None
    # All direct calls are in the real prefix process, which is independently
    # rejected after integrity/identity checks if no earlier damage exists.
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv('CUDA_VISIBLE_DEVICES','')
        with open_source(memory.fd,memory.index,
                         expected_completion_sha256=memory.receipt['completion_sha256'],
                         upstream=UPSTREAM,metadata_csv=memory.corpus.metadata_csv) as source:
            errors={'receipt':'prefix receipt','checkpoint':'frozen prefix checkpoint',
                    'identity':'identity differs','missing_sha':'prefix receipt','same_process':'fresh actual process'}
            with pytest.raises(ValueError,match=errors[damage]):
                api().run_recovery_phase(UPSTREAM,source,memory.corpus.metadata_csv,
                    'small',owned,phase='restore',device='cpu',expected_prefix_sha256=sha)
            source.assert_current()
    assert not (owned/'restore.json').exists()


def test_prefix_never_overwrites_existing_output(memory,prefix):
    from egoemg_repro.emg2pose_memory_index import open_source
    output,_ = prefix
    sha = storage._sha256(output/'prefix.ckpt')
    with open_source(memory.fd,memory.index,
                     expected_completion_sha256=memory.receipt['completion_sha256'],
                     upstream=UPSTREAM,metadata_csv=memory.corpus.metadata_csv) as source:
        with pytest.raises(FileExistsError):
            api().run_recovery_phase(UPSTREAM,source,memory.corpus.metadata_csv,
                'small',output,phase='prefix',device='cpu')
    assert storage._sha256(output/'prefix.ckpt') == sha


def test_checkpoint_io_preserves_frozen_prefix_and_limits_native_deletions(tmp_path):
    owned = tmp_path/'owned';owned.mkdir();(owned/'checkpoints').mkdir()
    io = api().checkpoint_io(owned)
    payload = {'state_dict':{'weight':torch.arange(4)},'global_step':2}
    io.save_checkpoint(payload,str(owned/'checkpoints/a.ckpt'))
    io.save_checkpoint(payload,str(owned/'checkpoints/a.ckpt'))
    io.remove_checkpoint(str(owned/'checkpoints/a.ckpt'))
    assert not (owned/'checkpoints/a.ckpt').exists()
    io.save_checkpoint(payload,str(owned/'prefix.ckpt'))
    sha = storage._sha256(owned/'prefix.ckpt')
    reader = api().checkpoint_io(owned,expected_checkpoint_sha256=sha)
    assert torch.equal(reader.load_checkpoint(str(owned/'prefix.ckpt'))['state_dict']['weight'],torch.arange(4))
    with pytest.raises(FileExistsError):io.save_checkpoint(payload,str(owned/'prefix.ckpt'))
    for path in (owned/'prefix.ckpt',tmp_path/'other.ckpt'):
        with pytest.raises(ValueError):io.remove_checkpoint(str(path))
    with pytest.raises(ValueError):io.save_checkpoint(payload,str(tmp_path/'other.ckpt'))
    with pytest.raises(ValueError):api().checkpoint_io(owned,expected_checkpoint_sha256='0'*64).load_checkpoint(str(owned/'prefix.ckpt'))
    assert storage._sha256(owned/'prefix.ckpt') == sha


def test_failed_final_publication_retains_failure_without_success_receipt(memory,tmp_path,monkeypatch):
    from egoemg_repro.emg2pose_memory_index import open_source
    recovery=api();output=tmp_path/'failed-final-publication'
    original=recovery.checks._write_json;failed=[]
    def interrupted(path,value):
        if Path(path) == output/'status.json' and not failed:
            failed.append(True)
            raise OSError('injected final status publication failure')
        return original(path,value)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(recovery.checks,'_write_json',interrupted)
    with open_source(memory.fd,memory.index,
                     expected_completion_sha256=memory.receipt['completion_sha256'],
                     upstream=UPSTREAM,metadata_csv=memory.corpus.metadata_csv) as source:
        with pytest.raises(OSError,match='injected final status'):
            recovery.run_recovery_phase(UPSTREAM,source,memory.corpus.metadata_csv,
                'small',output,phase='prefix',device='cpu')
        source.assert_current()
    assert json.loads((output/'status.json').read_text())['status'] == 'failed'
    assert not (output/'prefix.json').exists()
    assert (output/'prefix-started.json').is_file()


def test_changed_prediction_in_frozen_checkpoint_fails_full_fresh_validation(memory,prefix,tmp_path):
    output,_=prefix;owned=tmp_path/'changed-prediction';owned.mkdir();(owned/'checkpoints').mkdir()
    packet=json.loads((output/'prefix.json').read_text())
    checkpoint=torch.load(output/'prefix.ckpt',map_location='cpu',weights_only=False)
    checkpoint['state_dict']['model.head.net.3.bias'].add_(.25)
    torch.save(checkpoint,owned/'prefix.ckpt')
    packet['checkpoint_sha256']=storage._sha256(owned/'prefix.ckpt')
    (owned/'prefix.json').write_text(json.dumps(packet))
    result=subprocess.run(command(memory,owned,'restore',storage._sha256(owned/'prefix.json')),
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',EGOEMG_STAGE_FD_ARCHIVE=str(memory.fd)),
        pass_fds=(memory.fd,),capture_output=True,text=True,timeout=180)
    assert result.returncode != 0
    status=json.loads((owned/'status.json').read_text())
    assert status['status'] == 'failed'
    assert 'fresh complete native validation differs' in status['error']
    assert not (owned/'restore.json').exists()
    # An actual failed attempt is retained, rather than silently retried.
    retry=subprocess.run(command(memory,owned,'restore',storage._sha256(owned/'prefix.json')),
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',EGOEMG_STAGE_FD_ARCHIVE=str(memory.fd)),
        pass_fds=(memory.fd,),capture_output=True,text=True,timeout=180)
    assert retry.returncode != 0 and 'never automatically retried' in retry.stderr


@pytest.mark.parametrize('fault',['parse','index'])
def test_cli_closes_only_inherited_right_on_failure(memory,tmp_path,monkeypatch,fault):
    fd=os.dup(memory.fd)
    monkeypatch.setenv('EGOEMG_STAGE_FD_ARCHIVE',str(fd))
    args=command(memory,tmp_path/'never-trained','restore','0'*64)[3:]
    if fault == 'parse':args=[]
    else:
        args[args.index('--expected-index-completion-sha256')+1]='0'*64
    with pytest.raises((SystemExit,ValueError)):
        api().main(args)
    with pytest.raises(OSError):os.fstat(fd)
    assert os.fstat(memory.fd).st_size == len(memory.raw)
    assert not (tmp_path/'never-trained').exists()
