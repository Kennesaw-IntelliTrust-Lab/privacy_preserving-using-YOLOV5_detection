# Ultralytics 🚀 AGPL-3.0 License
"""
Train a YOLOv5 model on a custom dataset with Differential Privacy (Opacus).
"""

import argparse
import math
import os
import random
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path

try:
    import comet_ml
except ImportError:
    comet_ml = None

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from tqdm import tqdm
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

# YOLOv5
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

import val as validate
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.autobatch import check_train_batch_size
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.downloads import attempt_download, is_url
from utils.general import (
    LOGGER, TQDM_BAR_FORMAT, check_amp, check_dataset, check_file, check_git_info, check_git_status,
    check_img_size, check_requirements, check_suffix, check_yaml, colorstr, get_latest_run,
    increment_path, init_seeds, intersect_dicts, labels_to_class_weights, labels_to_image_weights,
    one_cycle, print_args, strip_optimizer, yaml_save
)
from utils.loss import ComputeLoss
from utils.metrics import fitness
from utils.plots import plot_evolve
from utils.torch_utils import (
    EarlyStopping, ModelEMA, de_parallel, select_device, smart_DDP,
    smart_optimizer, smart_resume, torch_distributed_zero_first
)

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
GIT_INFO = check_git_info()


def train(hyp, opt, device, callbacks):
    save_dir, epochs, batch_size, weights, single_cls, data, cfg, resume, workers = (
        Path(opt.save_dir),
        opt.epochs,
        opt.batch_size,
        opt.weights,
        opt.single_cls,
        opt.data,
        opt.cfg,
        opt.resume,
        opt.workers,
    )
    callbacks.run('on_pretrain_routine_start')

    w = save_dir / "weights"
    w.mkdir(parents=True, exist_ok=True)
    last, best = w / "last.pt", w / "best.pt"

    if isinstance(hyp, (str, Path)):
        with open(hyp, errors="ignore") as f:
            hyp = yaml.safe_load(f)
    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    opt.hyp = hyp.copy()

    yaml_save(save_dir / "hyp.yaml", hyp)
    yaml_save(save_dir / "opt.yaml", vars(opt))

    data_dict = check_dataset(data)
    train_path, val_path = data_dict['train'], data_dict['val']
    nc = 1 if single_cls else int(data_dict['nc'])
    names = {0: "item"} if single_cls else data_dict['names']

    pretrained = str(weights).endswith('.pt')
    if pretrained:
        weights = attempt_download(weights)
        ckpt = torch.load(weights, map_location='cpu')
        model = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)
        exclude = [] if resume else ['anchor']
        csd = ckpt['model'].float().state_dict()
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)
        model.load_state_dict(csd, strict=False)
        LOGGER.info(f'Transferred {len(csd)}/{len(model.state_dict())} layers from {weights}')
    else:
        model = Model(cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)

    amp = check_amp(model)
    gs = max(int(model.stride.max()), 32)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs*2)

    train_loader, dataset = create_dataloader(
        train_path, imgsz, batch_size // WORLD_SIZE, gs, single_cls,
        hyp=hyp, augment=True, cache=None, rect=opt.rect, rank=LOCAL_RANK,
        workers=workers, shuffle=True, seed=opt.seed, image_weights=False, quad=False, prefix=colorstr('train: ')
    )

    optimizer = smart_optimizer(model, opt.optimizer, hyp["lr0"], hyp["momentum"], hyp["weight_decay"])

    # ✅ Fix privacy before optimizer step
    if opt.dp:
        model = ModuleValidator.fix(model)
        optimizer = smart_optimizer(model, opt.optimizer, hyp["lr0"], hyp["momentum"], hyp["weight_decay"])  # New optimizer
        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=opt.noise_multiplier,
            max_grad_norm=opt.max_grad_norm,
            poisson_sampling=False
        )
        print("✅ Model is now private:", hasattr(model, "autograd_grad_sample_modules"))
        LOGGER.info(f"Privacy Engine attached: Noise Multiplier={opt.noise_multiplier}, Max Grad Norm={opt.max_grad_norm}")

    model.nc = nc
    model.hyp = hyp
    model.names = names
    compute_loss = ComputeLoss(model)

    lf = one_cycle(1, hyp['lrf'], epochs) if opt.cos_lr else lambda x: (1 - x/epochs) * (1.0 - hyp['lrf']) + hyp['lrf']
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    ema = ModelEMA(model) if RANK in {-1, 0} else None

    best_fitness, start_epoch = 0.0, 0
    if pretrained and resume:
        best_fitness, start_epoch, epochs = smart_resume(ckpt, optimizer, ema, weights, epochs, resume)
        del ckpt, csd

    # Adjust loss scaling for number of detection layers
    nl = de_parallel(model).model[-1].nl
    hyp['box'] *= 3. / nl
    hyp['cls'] *= nc / 80. * 3. / nl
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl
    hyp['label_smoothing'] = opt.label_smoothing
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc
    model.names = names

    t0 = time.time()
    nb = len(train_loader)
    last_opt_step = -1
    maps = np.zeros(nc)
    results = (0, 0, 0, 0, 0, 0, 0)
    scheduler.last_epoch = start_epoch - 1
    scaler = torch.cuda.amp.GradScaler(enabled=amp and not opt.dp)
    stopper, stop = EarlyStopping(patience=opt.patience), False

    LOGGER.info(f"Image sizes {imgsz} train, {imgsz} val\nUsing {train_loader.num_workers*WORLD_SIZE} dataloader workers")
    LOGGER.info(f"Logging results to {colorstr('bold', save_dir)}\nStarting training for {epochs} epochs...")

    for epoch in range(start_epoch, epochs):
        model.train()
        mloss = torch.zeros(3, device=device)
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)
        pbar = tqdm(enumerate(train_loader), total=nb, bar_format=TQDM_BAR_FORMAT) if RANK in {-1, 0} else enumerate(train_loader)
        optimizer.zero_grad()

        for i, (imgs, targets, paths, _) in pbar: 
            ni = i + nb * epoch
            imgs = imgs.to(device, non_blocking=True).float() / 255
            accumulate = 1 if opt.dp else max(1, np.interp(ni, [0, 1000], [1, 64//batch_size]).round())

            with torch.cuda.amp.autocast(enabled=amp and not opt.dp):
                pred = model(imgs)
                loss, loss_items = compute_loss(pred, targets.to(device))
                if RANK != -1:
                    loss *= WORLD_SIZE

            if opt.dp:
                loss.backward()
            else:
                scaler.scale(loss).backward()

            if ni - last_opt_step >= accumulate:
                if opt.dp:
                    optimizer.step()
                    optimizer.zero_grad()
                else:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                if ema:
                    ema.update(model)
                last_opt_step = ni

            if RANK in {-1, 0}:
                mloss = (mloss * i + loss_items) / (i + 1)
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
                pbar.set_description(f"{epoch}/{epochs-1} {mem} {mloss[0]:.4g} {mloss[1]:.4g} {mloss[2]:.4g} {targets.shape[0]} {imgs.shape[-1]}")

        scheduler.step()

        if RANK in {-1, 0}:
            final_epoch = epoch + 1 == epochs
            if not opt.noval or final_epoch:
                results, maps, _ = validate.run(
                    data_dict,
                    batch_size=batch_size // WORLD_SIZE * 2,
                    imgsz=imgsz,
                    half=amp,
                    model=ema.ema if ema else model,
                    single_cls=single_cls,
                    dataloader=create_dataloader(val_path, imgsz, batch_size // WORLD_SIZE * 2, gs, single_cls, hyp=hyp, rect=True, rank=-1, workers=workers*2, prefix=colorstr('val: '))[0],
                    save_dir=save_dir,
                    plots=False,
                    callbacks=callbacks,
                    compute_loss=compute_loss,
                )

            fi = fitness(np.array(results).reshape(1, -1))
            stop = stopper(epoch=epoch, fitness=fi)
            if fi > best_fitness:
                best_fitness = fi

            ckpt = {
                'epoch': epoch,
                'best_fitness': best_fitness,
                'model': deepcopy(de_parallel(model)).half(),
                'ema': deepcopy(ema.ema).half() if ema else None,
                'updates': ema.updates if ema else None,
                'optimizer': optimizer.state_dict(),
                'opt': vars(opt),
                'git': GIT_INFO,
                'date': datetime.now().isoformat(),
            }
            torch.save(ckpt, last)
            if best_fitness == fi:
                torch.save(ckpt, best)
            del ckpt

        if stop:
            break

    torch.cuda.empty_cache()
    return results


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / 'yolov5s.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--rect', action='store_true')
    parser.add_argument('--resume', nargs='?', const=True, default=False)
    parser.add_argument('--nosave', action='store_true')
    parser.add_argument('--noval', action='store_true')
    parser.add_argument('--device', default='')
    parser.add_argument('--multi-scale', action='store_true')
    parser.add_argument('--single-cls', action='store_true')
    parser.add_argument('--optimizer', type=str, default='SGD')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--project', default=ROOT / 'runs/train')
    parser.add_argument('--name', default='exp')
    parser.add_argument('--exist-ok', action='store_true')
    parser.add_argument('--cos-lr', action='store_true')
    parser.add_argument('--label-smoothing', type=float, default=0.0)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--freeze', nargs='+', type=int, default=[0])
    parser.add_argument('--save-period', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dp', action='store_true', help='Enable Differential Privacy')
    parser.add_argument('--noise-multiplier', type=float, default=1.0)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--delta', type=float, default=1e-5)
    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(opt, callbacks=Callbacks()):
    if RANK in {-1, 0}:
        print_args(vars(opt))
        check_git_status()
        check_requirements(ROOT / 'requirements.txt')

    opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        assert torch.cuda.device_count() > LOCAL_RANK
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend='nccl', timeout=timedelta(seconds=10800))

    train(opt.hyp, opt, device, callbacks)



if __name__ == '__main__':
    opt = parse_opt()
    main(opt)