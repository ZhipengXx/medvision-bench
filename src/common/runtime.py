"""Reproducible run metadata, checkpoints and strict configuration handling."""
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str, allow_nan=False))
    tmp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()


def source_hashes():
    return {str(p.relative_to(ROOT)): sha256(p) for folder in ['src', 'scripts']
            for p in sorted((ROOT / folder).rglob('*.py'))}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if torch.cuda.is_available() and state['cuda']:
        torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])


def checkpoint(path, model, optimizer, scheduler, scaler, counter, best, history, config, extra=None):
    state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                 scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), counter=counter,
                 best=best, history=history, config=config, rng=rng_state(), extra=extra or {})
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    torch.save(state, tmp)
    tmp.replace(path)


def restore(path, model, optimizer, scheduler, scaler):
    state = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    scaler.load_state_dict(state['scaler'])
    restore_rng(state['rng'])
    return state


def start_run(cfg, smoke, resume, split):
    run_id = cfg['run_id'] + ('_smoke' if smoke else '')
    out = ROOT / 'outputs' / run_id
    if out.exists() and (out / 'metadata.json').exists() and not resume:
        raise FileExistsError(f'{out} already exists; use --resume or a new run_id')
    out.mkdir(parents=True, exist_ok=True)
    meta = dict(run_id=run_id, formal=not smoke, seed=cfg['seed'], command=sys.argv,
                started=time.strftime('%Y-%m-%dT%H:%M:%S%z'), source_hashes=source_hashes(),
                config=cfg, split_sha256=sha256(ROOT / split), python=sys.version,
                torch=torch.__version__, cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU',
                complete=False)
    if resume:
        meta = json.loads((out / 'metadata.json').read_text())
        meta.setdefault('resumes', []).append(dict(command=sys.argv, time=time.time()))
        meta['complete'] = False
    save_json(out / 'metadata.json', meta)
    save_json(out / 'config.json', cfg)
    return out, meta


def progress(message):
    path = ROOT / 'outputs/progress.log'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        f.write(f'\n- {time.strftime("%Y-%m-%d %H:%M:%S")} {message}\n')
    print(message, flush=True)
