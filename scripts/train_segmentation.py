"""Patch training with full validation-volume model selection."""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.common.runtime import *
from src.segmentation.model import UNet
from src.segmentation.data import PatchDataset, read_volume
from src.segmentation.losses import ce_dice
from src.segmentation.inference import predict_volume


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--resume')
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed_all(cfg['seed'])
    device = torch.device(cfg['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    split = json.loads((ROOT / 'splits/la_split.json').read_text())
    data_cfg = load_config(ROOT / cfg['data_config'])['segmentation']
    root = ROOT / data_cfg['root']
    out, meta = start_run(cfg, args.smoke, args.resume, 'splits/la_split.json')
    model = UNet(cfg['dimensions'], cfg['base_channels']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['updates'])
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    start, best, history, stale, sums = 0, -1., [], 0, np.zeros(3)
    elapsed_before = 0.
    if args.resume:
        state = restore(args.resume, model, optimizer, scheduler, scaler)
        start, best, history = state['counter'], state['best'], state['history']
        stale = state['extra']['stale']
        elapsed_before = state['extra'].get('elapsed', 0.)
    budget = min(start + 4, cfg['updates']) if args.smoke else cfg['updates']
    train_rows = split['train'][:2] if args.smoke else split['train']
    ds = PatchDataset(train_rows, root, data_cfg, cfg['dimensions'], cfg['patch'], (budget - start) * cfg['batch_size'])
    # num_workers=0 preserves sampler/random state for exact continuation.
    loader = iter(DataLoader(ds, batch_size=cfg['batch_size'], num_workers=0, pin_memory=True))
    clock = time.monotonic()
    train_history = json.loads((out / 'train_history.json').read_text()) if args.resume else []
    times = []
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    for step in range(start + 1, budget + 1):
        t0 = time.monotonic()
        model.train()
        x, y = next(loader)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
            loss, ce, dl = ce_dice(model(x), y)
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite segmentation loss')
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        values = [loss.item(), ce.item(), dl.item()]
        train_history.append(dict(step=step, total=values[0], ce=values[1], dice_loss=values[2],
                                  lr=optimizer.param_groups[0]['lr']))
        times.append(time.monotonic() - t0)
        if step == 100:
            meta['stable_seconds_per_step'] = float(np.mean(times[1:101]))
            save_json(out / 'metadata.json', meta)
            progress(f"{meta['run_id']} 100-step estimate: {meta['stable_seconds_per_step']:.3f} s/update")
        if step % 100 == 0:
            print(json.dumps(train_history[-1]), flush=True)
        do_val = step % cfg['validation_every'] == 0 or step == budget
        if do_val:
            model.eval()
            val_clock, dice_rows = time.monotonic(), []
            val_rows = split['val'][:1] if args.smoke else split['val']
            for row in val_rows:
                im, gt = read_volume(root / row['path'], data_cfg)
                pred = predict_volume(model, im, cfg, device)
                n = int(pred.sum()) + int(gt.sum())
                dice = 2 * int(((pred == 1) & (gt == 1)).sum()) / n if n else 1.
                dice_rows.append(dict(case_id=row['case_id'], dice=dice))
            score = float(np.mean([r['dice'] for r in dice_rows]))
            history.append(dict(step=step, val_mean_dice=score, per_case=dice_rows,
                                validation_seconds=time.monotonic() - val_clock,
                                elapsed_seconds=elapsed_before + time.monotonic() - clock))
            improved = score > best
            if improved:
                best, stale = score, 0
                meta['best_step'] = step
            else:
                stale += 1
            extra = dict(stale=stale, elapsed=history[-1]['elapsed_seconds'])
            checkpoint(out / 'last.pt', model, optimizer, scheduler, scaler, step, best, history, cfg, extra)
            if improved:
                checkpoint(out / 'best.pt', model, optimizer, scheduler, scaler, step, best, history, cfg, extra)
            save_json(out / 'history.json', history)
            save_json(out / 'train_history.json', train_history)
            progress(f"{meta['run_id']} update {step}/{cfg['updates']}: full-volume val Dice={score:.4f}, best={best:.4f}")
            if step >= cfg['minimum_updates'] and stale >= cfg['patience']:
                break
        elif step % cfg['checkpoint_every'] == 0:
            checkpoint(out / 'last.pt', model, optimizer, scheduler, scaler, step, best, history, cfg,
                       dict(stale=stale, elapsed=elapsed_before + time.monotonic() - clock))
            save_json(out / 'train_history.json', train_history)
    meta.update(complete=True, completed_updates=step, best_val_dice=best,
                training_seconds=elapsed_before + time.monotonic() - clock,
                peak_gpu_bytes=torch.cuda.max_memory_allocated() if device.type == 'cuda' else 0,
                parameter_count=sum(p.numel() for p in model.parameters()),
                best_checkpoint_sha256=sha256(out / 'best.pt'), selection='mean full validation-volume foreground Dice')
    save_json(out / 'metadata.json', meta)
    progress(f"Completed {'FORMAL' if not args.smoke else 'SMOKE'} {meta['run_id']}; best step {meta['best_step']}")


if __name__ == '__main__':
    main()
