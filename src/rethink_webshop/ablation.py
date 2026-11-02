"""Audit a same-start control without importing Torch, Ray or any API client."""
import hashlib
import json
from pathlib import Path
from .common import ROOT, sha


def verify_recipe(root=None):
    root=Path(root) if root is not None else ROOT
    lock=json.loads((root/'configs/ablation.json').read_text(encoding='utf-8-sig'))
    if lock['schema']!='webshop-same-start-no-local-hint-v1':
        raise ValueError('Unknown ablation contract')
    actual=json.loads((root/'configs/opsd.json').read_text(encoding='utf-8-sig'))
    expected=dict(lock['matched_method_recipe'])
    if expected.get('algorithm.local_hints.enabled') is not True:
        raise ValueError('Matched method must use Actor-routed local hints')
    expected['algorithm.local_hints.enabled']=False
    if actual!=expected:
        raise ValueError('Ablation may change only algorithm.local_hints.enabled')
    for name,spec in lock['shared_files'].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError('Ablation lock path escapes repository')
        raw=path.read_bytes()
        if spec['mode']=='utf8_lf':raw=raw.replace(b'\r\n',b'\n')
        elif spec['mode']!='binary':raise ValueError('Unknown shared-file hash mode')
        if hashlib.sha256(raw).hexdigest()!=spec['sha256']:
            raise ValueError('Matched shared source/config/data changed: '+name)
    return lock


def verify_initialization(actor,identity=None,*,root=None):
    """Same public artifact, not merely the same architecture or SFT recipe.

    Reuse the launcher's already-computed weight/config/tokenizer hashes rather
    than hashing12GB twice. Other public manifest assets are checked directly.
    """
    root=Path(root) if root is not None else ROOT
    actor=Path(actor)
    lock=json.loads((root/'configs/checkpoint1500.json').read_text(encoding='utf-8-sig'))
    if lock['step']!=1500:
        raise ValueError('Expected the matched Route-SFT1500 initialization')
    known={}
    if identity:
        known.update(identity.get('weight_sha256',{}))
        known['config.json']=identity.get('model_config_sha256')
        known['tokenizer.json']=identity.get('tokenizer_sha256')
    weights={name for name in lock['assets'] if name.endswith('.safetensors')}
    if {p.name for p in actor.glob('*.safetensors')}!=weights:
        raise ValueError('Ablation requires the exact same released SFT weight set')
    for name,spec in lock['assets'].items():
        path=actor/name
        if not path.is_file() or path.stat().st_size!=spec['bytes']:
            raise ValueError('Missing/incomplete matched SFT asset: '+name)
        if (known.get(name) or sha(path))!=spec['sha256']:
            raise ValueError('Ablation initialization differs from matched SFT: '+name)
    return {'step':1500,'same_released_actor':True,'asset_count':len(lock['assets'])}


def main():
    lock=verify_recipe()
    print(json.dumps({'ablation':lock['schema'],'recipe_verified':True,
                      'only_algorithm_delta':'algorithm.local_hints.enabled=false',
                      'initialization':'same released Route-SFT1500; verified at training launch',
                      'GPU_used':False,'API_called':False},sort_keys=True))


if __name__=='__main__':
    main()
