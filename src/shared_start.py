"""Full-SFT2000 shared initialization: never substitute a base/trigger-only model."""
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def child(root, name):
    if Path(name).name != name or name in ('.', '..'):
        raise ValueError('Invalid initialization member name')
    path = (root / name).resolve()
    if path.parent != root.resolve() or (root / name).is_symlink():
        raise ValueError('Initialization cannot reference files outside its directory')
    return path


def checked_files(model, files):
    for name, rec in files.items():
        path = child(model, name)
        if not path.is_file() or path.stat().st_size != rec['bytes'] or sha(path) != rec['sha256']:
            raise ValueError('Shared Full-SFT2000 checksum mismatch: ' + name)
    return {name: rec['sha256'] for name, rec in sorted(files.items())}


def verify_start(model):
    model = Path(model).resolve()
    manifest = model / 'sft_stage_manifest.json'
    if not manifest.is_file():
        raise ValueError('A complete Full-Route-SFT2000 is required; base-start is not this ablation')
    lock = json.loads((ROOT / 'configs/shared_start.json').read_text(encoding='utf-8'))
    if sha(manifest) == lock['files']['sft_stage_manifest.json']['sha256']:
        return checked_files(model, lock['files'])
    # Independently reproducible common-start mode. Both Full and no-hint arms
    # must use these SAME saved files, not independent same-recipe SFT reruns.
    value = json.loads(manifest.read_text(encoding='utf-8'))
    if not (value.get('stage') == 'semantic_hr_route_bce_sft' and value.get('step') == 2000
            and value.get('checkpoint_complete') and value.get('epoch_end')):
        raise ValueError('Expected complete epoch-end Full-SFT2000')
    recipe = json.loads((ROOT / 'configs/sft.json').read_text(encoding='utf-8'))
    binding = value.get('binding', {})
    if binding.get('config') != recipe or recipe['supervision_mode'] != 'full':
        raise ValueError('Not the declared full-correction Route-SFT recipe')
    data = json.loads((ROOT / 'configs/data.json').read_text(encoding='utf-8'))
    if binding.get('dataset_manifest_sha256') != data['files']['route_sft/manifest.json']:
        raise ValueError('Different Route-SFT dataset')
    reference = json.loads((ROOT / 'configs/reference_3b.json').read_text(encoding='utf-8'))
    code = {key.replace('\\', '/'): val for key, val in binding.get('code', {}).items()}
    if code.get('route_bce/core.py') != reference['source_sha256_lf']['src/sft/route_bce/core.py']:
        raise ValueError('Full SFT loss implementation differs from the matched method')
    files = value.get('model_files', {})
    if 'config.json' not in files or not any(name.endswith('.safetensors') for name in files):
        raise ValueError('Incomplete Full-SFT model file set')
    return checked_files(model, files)


def import_start(source):
    source = Path(source).resolve()
    lock = json.loads((ROOT / 'configs/shared_start.json').read_text(encoding='utf-8'))
    checked_files(source, lock['files'])
    destination = ROOT / 'checkpoints/route_sft/global_step_2000'
    if destination.exists():
        checked_files(destination, lock['files'])
        print('Original shared Full-SFT2000 already verified.')
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(destination.name + '.import-' + uuid.uuid4().hex[:10])
    staging.mkdir()
    for name in lock['files']:
        shutil.copyfile(child(source, name), staging / name)
    for name in lock.get('legal_files', {}):
        shutil.copyfile(ROOT / 'configs' / name, staging / name)
    checked_files(staging, lock['files'])
    checked_files(staging, lock.get('legal_files', {}))
    (staging / 'shared_start_receipt.json').write_text(json.dumps({
        'schema': 'original_full_sft2000_import_v1', 'step': 2000,
        'lock_sha256': sha(ROOT / 'configs/shared_start.json'),
        'optimizer_state_copied': False,
    }, indent=2) + '\n', encoding='utf-8')
    staging.rename(destination)
    print('Exact original Full-SFT2000 imported; no optimizer, logs or credentials copied.')


def download_start(accepted):
    if not accepted:
        raise ValueError('Read the Qwen Research License and pass --accept-model-license')
    lock = json.loads((ROOT / 'configs/shared_start.json').read_text(encoding='utf-8'))
    destination = ROOT / 'checkpoints/route_sft/global_step_2000'
    if destination.exists():
        checked_files(destination, lock['files'])
        print('Original shared Full-SFT2000 already verified.')
        return
    staging = ROOT / 'checkpoints/full-sft2000-download'
    staging.mkdir(parents=True, exist_ok=True)
    base = 'https://github.com/' + lock['release_repository'] + '/releases/download/' + lock['release_tag'] + '/'
    for name, rec in {**lock['files'], **lock.get('legal_files', {})}.items():
        path = child(staging, name)
        if path.is_file():
            checked_files(staging, {name: rec})
            continue
        pending = path.with_name(name + '.partial')
        offset = pending.stat().st_size if pending.is_file() else 0
        if offset > rec['bytes']:
            raise ValueError('Oversized partial model download; inspect before retrying')
        if offset != rec['bytes']:
            request = urllib.request.Request(base + name, headers={'Range': 'bytes=' + str(offset) + '-'})
            with urllib.request.urlopen(request, timeout=120) as stream:
                mode = 'ab' if offset and stream.status == 206 else 'wb'
                if mode == 'ab' and not stream.headers.get('Content-Range', '').startswith('bytes ' + str(offset) + '-'):
                    raise ValueError('Unexpected download range response')
                with pending.open(mode) as output:
                    shutil.copyfileobj(stream, output, length=4 * 1024**2)
        if pending.stat().st_size != rec['bytes'] or sha(pending) != rec['sha256']:
            raise ValueError('Incomplete or mismatched public asset: ' + name)
        pending.rename(path)
        print('Downloaded and verified ' + name, flush=True)
    checked_files(staging, lock['files'])
    checked_files(staging, lock.get('legal_files', {}))
    (staging / 'shared_start_receipt.json').write_text(json.dumps({
        'schema': 'original_full_sft2000_import_v1', 'step': 2000,
        'lock_sha256': sha(ROOT / 'configs/shared_start.json'),
        'optimizer_state_copied': False,
    }, indent=2) + '\n', encoding='utf-8')
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(destination)
