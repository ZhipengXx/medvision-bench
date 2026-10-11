"""Train one independent ResNet-50 experiment; formal test is never loaded here."""
import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from src.common.runtime import *
from src.classification.data import ISICDataset, CLASSES
from src.classification.model import build_model
from src.classification.losses import make_loss, inverse_frequency
from src.classification.metrics import metrics
from src.classification.selection import selection_score, selection_description, should_stop


def worker_seed(worker):
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--resume')
    args = ap.parse_args()
    cfg = load_config(args.config)
    selection_metric = cfg.get('selection_metric', 'accuracy')
    minimum_epochs = cfg.get('minimum_epochs', 0)
    selection_description(selection_metric)  # Fail before creating a run.
    if not isinstance(minimum_epochs, int) or not 0 <= minimum_epochs <= cfg['epochs']:
        raise ValueError('minimum_epochs must be an integer between 0 and epochs')
    should_stop(0, 0, cfg['patience'], minimum_epochs)
    seed_all(cfg['seed'])
    device = torch.device(cfg.get('device', 'cuda'))
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; run in the authorized GPU environment')
    split = json.loads((ROOT / 'splits/isic_split.json').read_text())
    data_cfg = load_config(ROOT / cfg['data_config'])['classification']
    counts = [sum(x['label'] == i for x in split['train']) for i in range(7)]
    train_rows = split['train'][:64] if args.smoke else split['train']
    val_rows = split['val'][:32] if args.smoke else split['val']
    train = ISICDataset(train_rows, ROOT / data_cfg['root'], True)
    val = ISICDataset(val_rows, ROOT / data_cfg['root'])
    out, meta = start_run(cfg, args.smoke, args.resume, 'splits/isic_split.json')
    model = build_model(True).to(device)
    init = ROOT / 'outputs/resnet50_initial.pt'
    if not init.exists():
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, init)
    else:
        state = torch.load(init, map_location='cpu', weights_only=True)
        for k, v in model.state_dict().items():
            if not torch.equal(v.cpu(), state[k]):
                raise RuntimeError('The three experiments must start identically')
    meta['initialization'] = dict(source='torchvision.ResNet50_Weights.IMAGENET1K_V2',
                                  initial_sha256=sha256(init), classifier_seed=cfg['seed'])
    save_json(out / 'metadata.json', meta)
    save_json(out / 'class_weights.json', dict(classes=CLASSES, train_counts=counts,
                                               inverse_weights=inverse_frequency(counts).tolist()))
    objective = make_loss(cfg['loss'], counts, device)
    backbone = [p for n, p in model.named_parameters() if not n.startswith('fc.')]
    optimizer = torch.optim.AdamW([{'params': backbone, 'lr': cfg['backbone_lr']},
                                   {'params': model.fc.parameters(), 'lr': cfg['head_lr']}],
                                  weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['epochs'])
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    best = tuple(-1. for _ in selection_score(dict(accuracy=0., macro_f1=0., macro_recall=0.), selection_metric))
    best_metrics, history, start, stale = None, [], 0, 0
    elapsed_before = 0.
    if args.resume:
        state = restore(args.resume, model, optimizer, scheduler, scaler)
        if state['config'].get('selection_metric', 'accuracy') != selection_metric:
            raise ValueError('Cannot change the checkpoint-selection rule when resuming a run')
        start, best, history = state['counter'], tuple(state['best']), state['history']
        stale = state['extra']['stale']
        elapsed_before = state['extra'].get('elapsed', 0.)
        best_metrics = state['extra'].get('best_metrics')
        if best_metrics is None:
            # Legacy accuracy-selected checkpoints contain the accuracy/F1 pair.
            best_metrics = dict(accuracy=best[0], macro_f1=best[1])
    clock = time.monotonic()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    epochs = min(start + 2, cfg['epochs']) if args.smoke else cfg['epochs']
    for epoch in range(start, epochs):
        generator = torch.Generator().manual_seed(cfg['seed'] + epoch)
        loader = DataLoader(train, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['workers'],
                            pin_memory=True, generator=generator, worker_init_fn=worker_seed)
        model.train()
        sums = dict(objective=0., ce=0., correct=0, n=0)
        stable_times = []
        for step, (x, y, _) in enumerate(loader):
            t0 = time.monotonic()
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                logits = model(x)
                loss = objective(logits, y)
                ce = F.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss')
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            n = len(y)
            sums['objective'] += loss.item() * n
            sums['ce'] += ce.item() * n
            sums['correct'] += int((logits.argmax(1) == y).sum())
            sums['n'] += n
            stable_times.append(time.monotonic() - t0)
            if step == 100 and epoch == 0:
                meta['stable_seconds_per_step'] = float(np.mean(stable_times[1:101]))
                save_json(out / 'metadata.json', meta)
                progress(f"{meta['run_id']} 100-step estimate: {meta['stable_seconds_per_step']:.3f} s/update")
        model.eval()
        ys, ps, image_ids, val_objective, val_ce = [], [], [], 0., 0.
        with torch.inference_mode():
            for x, y, ids in DataLoader(val, batch_size=cfg['batch_size'], num_workers=cfg['workers']):
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                    logits = model(x)
                    val_objective += objective(logits, y).item() * len(y)
                    val_ce += F.cross_entropy(logits, y).item() * len(y)
                ys.extend(y.cpu().tolist())
                ps.extend(logits.float().softmax(1).cpu().tolist())
                image_ids.extend(ids)
        m, _ = metrics(ys, ps)
        if cfg.get('save_val_predictions', False):
            val_dir = out / 'validation'
            val_dir.mkdir(exist_ok=True)
            with (val_dir / f'epoch_{epoch + 1:03d}.csv').open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['image_id', 'gt', 'pred', *['p_' + name for name in CLASSES]])
                for image_id, label, probability in zip(image_ids, ys, ps):
                    writer.writerow([image_id, label, int(np.argmax(probability)), *probability])
        row = dict(epoch=epoch + 1, train_objective=sums['objective'] / sums['n'], train_ce=sums['ce'] / sums['n'],
                   train_accuracy=sums['correct'] / sums['n'], val_objective=val_objective / len(val),
                   val_ce=val_ce / len(val), val_accuracy=m['accuracy'], val_macro_f1=m['macro_f1'],
                   val_macro_recall=m['macro_recall'],
                   seconds=elapsed_before + time.monotonic() - clock, lr=optimizer.param_groups[0]['lr'])
        history.append(row)
        score = selection_score(m, selection_metric)
        improved = score > best
        if improved:
            best, stale = score, 0
            best_metrics = {key: m[key] for key in ('accuracy', 'macro_f1', 'macro_recall')}
            meta['best_epoch'] = epoch + 1
        else:
            stale += 1
        scheduler.step()
        extra = dict(stale=stale, elapsed=row['seconds'], best_metrics=best_metrics)
        checkpoint(out / 'last.pt', model, optimizer, scheduler, scaler, epoch + 1, best, history, cfg, extra)
        if improved:
            checkpoint(out / 'best.pt', model, optimizer, scheduler, scaler, epoch + 1, best, history, cfg, extra)
        save_json(out / 'history.json', history)
        progress(f"{meta['run_id']} epoch {epoch+1}/{cfg['epochs']}: val acc={m['accuracy']:.4f}, macro F1={m['macro_f1']:.4f}, best {selection_metric}={best[0]:.4f}")
        if should_stop(epoch + 1, stale, cfg['patience'], minimum_epochs):
            break
    meta.update(complete=True, completed_epochs=len(history), best_val_accuracy=best_metrics['accuracy'], best_val_macro_f1=best_metrics['macro_f1'],
                training_seconds=history[-1]['seconds'], best_checkpoint_sha256=sha256(out / 'best.pt'),
                peak_gpu_bytes=torch.cuda.max_memory_allocated() if device.type == 'cuda' else 0,
                parameter_count=sum(p.numel() for p in model.parameters()), selection=selection_description(selection_metric))
    if 'macro_recall' in best_metrics:
        meta['best_val_macro_recall'] = best_metrics['macro_recall']
    save_json(out / 'metadata.json', meta)
    progress(f"Completed {'FORMAL' if not args.smoke else 'SMOKE'} {meta['run_id']}; best epoch {meta['best_epoch']}")


if __name__ == '__main__':
    main()
