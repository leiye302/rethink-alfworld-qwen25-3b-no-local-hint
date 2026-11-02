"""Same SFT artifact and actual native no-hint Teacher branch; no GPUs/API."""
import ast
import hashlib
import json
from types import SimpleNamespace
import pytest
from rethink_webshop import ablation
from rethink_webshop.common import ROOT, UPSTREAM


def test_recipe_diff_is_exactly_one_boolean_and_common_files_match():
    lock=ablation.verify_recipe()
    assert lock['matched_method']['commit']=='cb6e5d93929c4e33bb431b0cdcf353ea42196a91'
    actual=json.loads((ROOT/'configs/opsd.json').read_text())
    reference=lock['matched_method_recipe']
    assert {k:(reference[k],actual[k]) for k in reference if reference[k]!=actual[k]}=={
        'algorithm.local_hints.enabled':(True,False)}
    assert reference.keys()==actual.keys()
    assert actual['algorithm.sdar.gate_beta']==5
    assert actual['trainer.total_training_steps']==150
    assert len(lock['shared_files'])>=50


def test_same_checkpoint_uses_the_public_parent_release_not_base():
    lock=json.loads((ROOT/'configs/checkpoint1500.json').read_text())
    assert lock['repository']=='leiye302/rethink-webshop-qwen25-3b'
    assert lock['tag']=='route-sft1500-v1' and lock['step']==1500
    weights=[x for x in lock['assets'] if x.endswith('.safetensors')]
    assert len(weights)==7
    assert all(lock['assets'][x]['bytes']>0 and len(lock['assets'][x]['sha256'])==64 for x in weights)


def test_baseline_flag_cannot_silently_change_initialization(capsys):
    from rethink_webshop.run import main
    with pytest.raises(SystemExit) as caught:main(['--baseline','--prepare-only'])
    assert caught.value.code==2
    assert 'same Route-SFT1500' in capsys.readouterr().err


def test_ablation_has_no_api_cost_flag():
    from rethink_webshop.run import main
    with pytest.raises(SystemExit) as caught:main(['--allow-api-cost','--prepare-only'])
    assert caught.value.code==2


def test_prepare_does_not_read_hint_credentials_or_use_cuda(tmp_path,monkeypatch):
    from rethink_webshop import run, setup, hint_api
    from omegaconf import OmegaConf
    def forbidden(*args,**kwargs):pytest.fail('Control must never use hints or CUDA')
    monkeypatch.setattr(setup,'verify_framework',lambda:None)
    monkeypatch.setattr(run,'placeholders',lambda:None)
    monkeypatch.setattr(run,'confined',lambda value:tmp_path/value)
    monkeypatch.setattr(run,'check_cuda',forbidden)
    monkeypatch.setattr(hint_api,'auth',forbidden)
    monkeypatch.setattr(hint_api,'fetch',forbidden)
    for key in ('HINT_API_KEY','HINT_MODEL','HINT_API_BASE','RETHINK_HINT_AUTH'):
        monkeypatch.setenv(key,'not-used-by-this-control')
    run.main(['--prepare-only'])
    cfg=OmegaConf.load(tmp_path/'runs/sft_start_grpo_opsd/prepared_config.yaml')
    assert cfg.actor_rollout_ref.model.path==str(tmp_path/'assets/route-sft1500')
    assert not cfg.algorithm.local_hints.enabled
    assert cfg.trainer.total_training_steps==150 and cfg.trainer.test_freq==10
    assert not any('HINT' in key for key in cfg.ray_init.runtime_env.env_vars)


def test_actual_upstream_teacher_branch_never_calls_hint_for_marker_rows(monkeypatch):
    from omegaconf import OmegaConf
    from rethink_webshop import hints, hint_api
    if not UPSTREAM.exists():pytest.skip('Prepare pinned public SDAR first')
    tree=ast.parse((UPSTREAM/'verl/trainer/ppo/skillsd_ray_trainer.py').read_text())
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign)
              and any(isinstance(t,ast.Name) and t.id=='teacher_log_probs' for t in n.targets)
              and isinstance(n.value,ast.IfExp))
    calls=[];sampled=SimpleNamespace(rethink_triggered=[True,False,True])
    def native(batch):calls.append(batch);return 'unchanged native current-Actor score'
    def forbidden(*args,**kwargs):pytest.fail('Local hint API/Teacher branch must not run')
    monkeypatch.setattr(hint_api,'fetch',forbidden)
    cfg=OmegaConf.create({'algorithm':{'local_hints':{'enabled':False}}})
    trainer=SimpleNamespace(config=cfg,_compute_teacher_log_probs=native)
    ns={'self':trainer,'batch':sampled,'enabled':hints.enabled,'score_teacher':forbidden}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'actual_native_teacher_branch','exec'),ns)
    assert calls==[sampled] and ns['teacher_log_probs']=='unchanged native current-Actor score'


def fixture_checkpoint(tmp_path):
    actor=tmp_path/'actor';actor.mkdir()
    root=tmp_path/'repository';(root/'configs').mkdir(parents=True)
    assets={}
    for name,raw in [('model.safetensors',b'fixture weight bytes'),('config.json',b'{}'),('tokenizer.json',b'{}')]:
        (actor/name).write_bytes(raw)
        assets[name]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
    (root/'configs/checkpoint1500.json').write_text(json.dumps({'step':1500,'assets':assets}))
    return actor,root,assets


def test_matching_initialization_files_are_required(tmp_path):
    actor,root,assets=fixture_checkpoint(tmp_path)
    result=ablation.verify_initialization(actor,root=root)
    assert result['same_released_actor'] and result['asset_count']==3
    (actor/'model.safetensors').write_bytes(b'fixture WEIGHT bytes')
    with pytest.raises(ValueError,match='differs from matched SFT'):
        ablation.verify_initialization(actor,root=root)


def test_missing_initialization_asset_is_not_accepted(tmp_path):
    actor,root,assets=fixture_checkpoint(tmp_path)
    (actor/'tokenizer.json').unlink()
    with pytest.raises(ValueError,match='Missing/incomplete'):
        ablation.verify_initialization(actor,root=root)


def test_second_stage_launcher_has_no_credential_loading():
    source=(ROOT/'src/rethink_webshop/run.py').read_text()
    assert 'HINT_API_KEY' not in source and 'from .hint_api import' not in source
    assert "'algorithm.local_hints.enabled':False" in source
    assert 'verify_initialization(actor,identity)' in source
