import argparse
import copy
import csv
import gc
import hashlib
import json
import math
import os
import random
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import ultralytics
from ultralytics import YOLO
from ultralytics.cfg import get_cfg

from .data import HeadDataset
from .losses import CrossingKD, raw_output


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def state_digest(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def collate(items):
    arrays, labels, paths = zip(*items)
    boxes = np.concatenate(labels, axis=0)
    indexes = np.concatenate([np.full(len(box), i, dtype=np.float32) for i, box in enumerate(labels)])
    return dict(img=torch.from_numpy(np.stack(arrays)),
                bboxes=torch.from_numpy(boxes),
                cls=torch.zeros(len(boxes), 1),
                batch_idx=torch.from_numpy(indexes), im_file=list(paths))


def move(batch, device):
    return {k:v.to(device) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}


def finite(value, label):
    if not torch.isfinite(value).all().item():
        raise RuntimeError(f'Non-finite {label}; stop and inspect, do not continue silently.')


def norm(parameters):
    total = 0.
    for p in parameters:
        if p.grad is not None:
            finite(p.grad, 'gradient')
            total += float(p.grad.detach().abs().sum().item())
    return total


def export_student(student, path, args, epoch):
    """Standard DetectionModel only: no custom class, teacher, or adapter dependency."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    network = copy.deepcopy(student).cpu().float().eval()
    if hasattr(network, 'criterion'):
        delattr(network, 'criterion')
    network.names = {0: 'head'}
    settings = dict(model=str(args.student), data=str(args.data), imgsz=args.imgsz,
                    task='detect', epochs=args.epochs, batch=1, seed=args.seed,
                    optimizer='AdamW', lr0=args.lr, amp=False, workers=0)
    network.args = settings
    payload = dict(model=network, ema=None, optimizer=None, epoch=epoch,
                   train_args=settings, version=ultralytics.__version__,
                   date=datetime.now(timezone.utc).isoformat(),
                   headlab_note='Custom matched continuation; see adjacent run.json for exact protocol.')
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)
    del network, payload


def setup(args):
    if ultralytics.__version__ != '8.4.176':
        raise RuntimeError('This build targets the inspected Ultralytics 8.4.176. Do not upgrade for this experiment.')
    for path in [args.student, args.data] + ([args.teacher] if args.command == 'smoke' or args.mode == 'kd' else []):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable. This configuration was prepared for the user\'s M1; report before switching device.')
    device = torch.device('mps')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    dataset = HeadDataset(args.data, args.imgsz, args.seed)
    wrapper = YOLO(str(args.student))
    student = wrapper.model.float().to(device)
    student.requires_grad_(True)
    for name, p in student.named_parameters():
        if name.endswith('dfl.conv.weight'):
            p.requires_grad_(False)
    student.args = get_cfg(overrides=dict(data=str(args.data), imgsz=args.imgsz,
        task='detect', box=7.5, cls=.5, dfl=1.5, single_cls=False))
    student.names = {0:'head'}
    student.train()
    if student.model[-1].nc != 1:
        raise RuntimeError('Expected one head class')
    teacher, crossing = None, None
    if args.command == 'smoke' or args.mode == 'kd':
        teacher = YOLO(str(args.teacher)).model.float().to(device).eval()
        teacher.requires_grad_(False)
        sh, th = student.model[-1], teacher.model[-1]
        if sh.end2end or th.end2end or sh.nc != th.nc or sh.reg_max != th.reg_max:
            raise RuntimeError('Unsupported head configuration')
        if sh.stride.tolist() != th.stride.tolist() or sh.stride.tolist() != [4,8,16,32]:
            raise RuntimeError('Expected matching P2/P3/P4/P5 grids')
        crossing = CrossingKD(sh, th).to(device)
    # Teacher/adapters must not change the subsequent global training RNG stream.
    torch.manual_seed(args.seed)
    parameters = [p for p in student.parameters() if p.requires_grad]
    if crossing is not None:
        parameters += list(crossing.parameters())
    groups = [dict(params=[p for p in parameters if p.ndim > 1], weight_decay=.0005),
              dict(params=[p for p in parameters if p.ndim <= 1], weight_decay=0.)]
    optimizer = torch.optim.AdamW(groups, lr=args.lr, betas=(.937,.999))
    criterion = student.init_criterion()
    return device, dataset, student, teacher, crossing, optimizer, criterion, parameters


def compute(student, teacher, crossing, criterion, batch, args):
    predictions = raw_output(student(batch['img']))
    detection, _ = criterion(predictions, batch)
    detection = detection.sum()/batch['img'].shape[0]
    finite(detection, 'supervised detection loss')
    if crossing is None:
        return detection, detection*0, detection*0, [0]*4, predictions
    if teacher.training:
        raise RuntimeError('Teacher must remain in eval mode')
    with torch.no_grad():
        targets = raw_output(teacher(batch['img']))
    classification, regression, counts = crossing(predictions, targets, teacher.model[-1], batch, args.quality)
    finite(classification, 'classification KD')
    finite(regression, 'regression KD')
    return detection, classification, regression, counts, predictions


def smoke(args, components):
    device, dataset, student, teacher, crossing, optimizer, criterion, parameters = components
    teacher_before = state_digest(teacher)
    source_hashes = sha(args.student), sha(args.teacher)
    batch = move(collate([dataset[0]]), device)
    optimizer.zero_grad(set_to_none=True)
    det, cls, reg, counts, predictions = compute(student, teacher, crossing, criterion, batch, args)
    kd = cls+reg
    if not sum(counts):
        raise RuntimeError('No reliable teacher positions in smoke image; inspect masks before training.')
    gradients = torch.autograd.grad(kd, predictions['feats'], retain_graph=True, allow_unused=True)
    feature_gradients = [0. if g is None else float(g.detach().abs().sum().item()) for g in gradients]
    if not all(math.isfinite(g) for g in feature_gradients) or sum(feature_gradients) <= 0:
        raise RuntimeError('No finite nonzero KD gradient to student features')
    branch_gradients = []
    for branch in [cls, reg]:
        values = torch.autograd.grad(branch, predictions['feats'], retain_graph=True, allow_unused=True)
        magnitude = sum(float(v.detach().abs().sum().item()) for v in values if v is not None)
        if not math.isfinite(magnitude) or magnitude <= 0:
            raise RuntimeError('Classification or regression KD has no finite nonzero feature gradient')
        branch_gradients.append(magnitude)
        del values
    total = det + args.lambda_cls*cls + args.lambda_reg*reg
    loss_values = [float(v.detach().item()) for v in [det,cls,reg,total]]
    finite(total, 'combined loss')
    total.backward()
    head_gradient = norm(student.model[-1].parameters())
    adapter_gradients = [norm(adapter.parameters()) for adapter in crossing.adapters]
    if head_gradient <= 0 or sum(adapter_gradients) <= 0:
        raise RuntimeError('Missing supervised-head or adapter gradients')
    if any(p.grad is not None for p in teacher.parameters()):
        raise RuntimeError('Teacher gradients found')
    adapter_before = crossing.adapters[0].weight.detach().clone()
    total_norm = torch.nn.utils.clip_grad_norm_(parameters, 10.)
    finite(total_norm, 'gradient norm')
    optimizer.step()  # In-memory step only; source checkpoints are not overwritten.
    if torch.equal(adapter_before, crossing.adapters[0].weight.detach()):
        raise RuntimeError('Optimizer did not update first adapter')
    if teacher_before != state_digest(teacher):
        raise RuntimeError('Teacher parameters or buffers changed')
    del gradients, predictions, total, det, cls, reg, kd
    optimizer.zero_grad(set_to_none=True)
    student.eval().requires_grad_(False)

    with tempfile.TemporaryDirectory(prefix='headlab_kd_smoke_') as temporary:
        path = Path(temporary)/'student.pt'
        export_student(student, path, args, 0)

        restored = YOLO(str(path)).model.cpu().float().eval()
        restored.requires_grad_(False)

        left_state = student.state_dict()
        right_state = restored.state_dict()

        if set(left_state) != set(right_state):
            raise RuntimeError('Export changed state keys')

        for key in left_state:
            first = left_state[key].detach().cpu()
            second = right_state[key].detach().cpu()
            if not torch.equal(first, second):
                raise RuntimeError(f'Export changed state tensor: {key}')

        print('Exact checkpoint state parity: PASS', flush=True)

        cpu_reference = copy.deepcopy(student).cpu().float().eval()
        cpu_images = batch['img'].detach().cpu()

        with torch.no_grad():
            cpu_expected = cpu_reference(cpu_images)[0]
            cpu_actual = restored(cpu_images)[0]

        if not torch.allclose(
            cpu_expected, cpu_actual, atol=1e-4, rtol=1e-4
        ):
            raise RuntimeError('CPU export/reload prediction parity failed')

        print(
            'CPU prediction parity: PASS; max difference:',
            (cpu_expected-cpu_actual).abs().max().item(),
            flush=True
        )

        del cpu_reference, cpu_expected, cpu_actual, cpu_images

        restored.to(device).eval()

        with torch.no_grad():
            for _ in range(3):
                student(batch['img'])
                restored(batch['img'])

            torch.mps.synchronize()

            original_a = student(batch['img'])[0].detach().cpu()
            restored_a = restored(batch['img'])[0].detach().cpu()
            original_b = student(batch['img'])[0].detach().cpu()
            restored_b = restored(batch['img'])[0].detach().cpu()

        comparisons = [
            ('Original MPS repeat', original_a, original_b),
            ('Reloaded MPS repeat', restored_a, restored_b),
            ('Warmed MPS export/reload', original_b, restored_b),
        ]

        for label, first, second in comparisons:
            box_error = (
                first[:, :4]-second[:, :4]
            ).abs().max().item()

            score_error = (
                first[:, 4:]-second[:, 4:]
            ).abs().max().item()

            print(
                label, 'box max:', box_error,
                'score max:', score_error,
                flush=True
            )

            if not torch.allclose(
                first, second, atol=1e-4, rtol=1e-4
            ):
                raise RuntimeError(
                    label + ' failed unchanged tolerance; send the output.'
                )

        del restored, original_a, original_b, restored_a, restored_b

    if source_hashes != (sha(args.student), sha(args.teacher)):
        raise RuntimeError('Source checkpoint changed')
    print('\n========== SUPERVISED + KD SMOKE PASS ==========')
    print('Supervised / classKD / boxKD / total losses:', loss_values)
    print('Reliable positions by stride [4,8,16,32]:', counts)
    print('KD feature gradients by level:', feature_gradients)
    print('Separate classKD / boxKD feature gradients:', branch_gradients)
    print('Adapter gradients:', adapter_gradients)
    print('Supervised student-head gradient:', head_gradient)
    print('Teacher parameters/buffers unchanged; teacher gradients NONE.')
    print('One optimizer step in memory passed. Standard student checkpoint reload passed.')
    print('Source checkpoints unchanged; no training run created.')


def metadata(args, dataset):
    return dict(protocol='headlab_crossing_v1', mode=args.mode, epochs=args.epochs,
        imgsz=args.imgsz, batch=1, accumulate=args.accumulate, lr=args.lr,
        seed=args.seed, optimizer='AdamW', weight_decay=.0005,
        beta=1, gamma=2, lambda_cls=args.lambda_cls, lambda_reg=args.lambda_reg,
        quality=args.quality, ramp_epochs=3, student=str(args.student),
        student_sha256=sha(args.student), teacher=str(args.teacher) if args.mode=='kd' else None,
        teacher_sha256=sha(args.teacher) if args.mode=='kd' else None,
        data=str(args.data), train_labels_fingerprint=dataset.fingerprint,
        train_images=len(dataset), augmentation='square letterbox, horizontal flip .5; no mosaic',
        initialization='same clean P2 best.pt for both continuation arms',
        ultralytics=ultralytics.__version__, torch=torch.__version__,
        implementation_sha256={p.name:sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
        validation='combined val, conf=.001, nms_iou=.7, max_det=1000, batch=1',
        deviations='YOLO11 P2, GT/teacher-IoU mask, equal-level reduction, adapters, KD ramp, continuation schedule; not exact S2 reproduction')


def write_history(path, history):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    os.replace(temporary, path)


def train(args, components):
    device, dataset, student, teacher, crossing, optimizer, criterion, parameters = components
    output = args.root / 'results/kd_continuation_v1' / args.mode
    state_path = output / 'training_state.pt'
    meta = metadata(args, dataset)
    start, best, history = 0, -1., []
    if args.resume:
        if not state_path.is_file() or not (output/'run.json').is_file():
            raise RuntimeError('No resumable custom checkpoint/run.json found')
        if json.loads((output/'run.json').read_text()) != meta:
            raise RuntimeError('Protocol/source changed. Do not silently change a resumed run.')
        # Only load your own local custom training_state.pt, never an untrusted file.
        state = torch.load(state_path, map_location='cpu', weights_only=False)
        student.load_state_dict(state['student'])
        if crossing is not None:
            crossing.load_state_dict(state['adapters'])
        optimizer.load_state_dict(state['optimizer'])
        start, best, history = state['epoch']+1, state['best'], state['history']
        torch.set_rng_state(state['torch_rng'])
        torch.mps.set_rng_state(state['mps_rng'])
        del state
        if start >= args.epochs:
            print('Run already completed. No training repeated.')
            return
    else:
        if output.exists():
            raise RuntimeError(f'Run already exists: {output}; use --resume with identical settings.')
        output.mkdir(parents=True)
        (output/'run.json').write_text(json.dumps(meta, indent=2))
    print('Run:', output, flush=True)
    print('Train images:', len(dataset), 'Batch: 1; accumulation:', args.accumulate, flush=True)
    teacher_before = state_digest(teacher) if teacher is not None else None
    for epoch in range(start, args.epochs):
        dataset.epoch = epoch
        generator = torch.Generator().manual_seed(args.seed+epoch)
        loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0,
                            collate_fn=collate, generator=generator)
        # Fixed continuation schedule, identical in control and KD.
        lr = args.lr*(.1+.9*(1+math.cos(math.pi*epoch/max(1,args.epochs-1)))/2)
        for group in optimizer.param_groups:
            group['lr'] = lr
        student.train()
        if teacher is not None:
            teacher.eval()
        optimizer.zero_grad(set_to_none=True)
        totals = [0.,0.,0.]
        level_counts = [0]*4
        weight = min(1., (epoch+1)/3) if crossing is not None else 0.
        for index, batch in enumerate(loader):
            batch = move(batch, device)
            det, cls, reg, counts, predictions = compute(student, teacher, crossing, criterion, batch, args)
            total = det + weight*(args.lambda_cls*cls+args.lambda_reg*reg)
            finite(total, 'total loss')
            window_start = (index//args.accumulate)*args.accumulate
            denominator = min(args.accumulate, len(loader)-window_start)
            (total/denominator).backward()
            totals = [a+float(b.detach().item()) for a,b in zip(totals,[det,cls,reg])]
            level_counts = [a+b for a,b in zip(level_counts,counts)]
            if (index+1)%args.accumulate == 0 or index+1 == len(loader):
                total_norm = torch.nn.utils.clip_grad_norm_(parameters, 10.)
                finite(total_norm, 'gradient norm')
                if teacher is not None and any(p.grad is not None for p in teacher.parameters()):
                    raise RuntimeError('Teacher unexpectedly has gradients')
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            del batch, det, cls, reg, total, predictions
            if (index+1)%100 == 0 or index+1 == len(loader):
                print(f'Epoch {epoch+1}/{args.epochs}: {index+1}/{len(loader)} images; mean det/clsKD/regKD {[round(v/(index+1),5) for v in totals]}', flush=True)
        if crossing is not None and not sum(level_counts):
            raise RuntimeError('Entire epoch had no reliable KD targets; inspect masking.')
        if teacher is not None and state_digest(teacher) != teacher_before:
            raise RuntimeError('Teacher state changed')
        last = output/'weights/last.pt'
        export_student(student, last, args, epoch)
        gc.collect()
        torch.mps.empty_cache()
        evaluator = YOLO(str(last))
        metrics = evaluator.val(data=str(args.data), split='val', imgsz=args.imgsz,
            batch=1, device='mps', workers=0, conf=.001, iou=.7, max_det=1000,
            augment=False, plots=False, project=str(output/'validation'),
            name=f'epoch_{epoch+1:03d}', exist_ok=True)
        score = float(metrics.box.map)
        if not math.isfinite(score):
            raise RuntimeError('Non-finite validation score')
        row = dict(epoch=epoch+1, lr=lr, kd_weight=weight,
                   detection=totals[0]/len(loader), kd_cls=totals[1]/len(loader),
                   kd_reg=totals[2]/len(loader), precision=float(metrics.box.mp),
                   recall=float(metrics.box.mr), map50=float(metrics.box.map50), map50_95=score,
                   eligible_p2=level_counts[0], eligible_p3=level_counts[1],
                   eligible_p4=level_counts[2], eligible_p5=level_counts[3])
        history.append(row)
        if score > best:
            best = score
            # Export exactly the state just evaluated. Teacher/adapters are excluded.
            export_student(student, output/'weights/best.pt', args, epoch)
        del metrics, evaluator
        state = dict(epoch=epoch, best=best, history=history,
            student={k:v.detach().cpu().clone() for k,v in student.state_dict().items()},
            adapters=({k:v.detach().cpu().clone() for k,v in crossing.state_dict().items()} if crossing is not None else None),
            optimizer=optimizer.state_dict(), torch_rng=torch.get_rng_state(),
            mps_rng=torch.mps.get_rng_state())
        temporary = state_path.with_suffix('.tmp')
        torch.save(state, temporary)
        os.replace(temporary, state_path)
        del state
        write_history(output/'results.csv', history)
        print('EPOCH RESULT:', json.dumps(row), flush=True)
        gc.collect()
        torch.mps.empty_cache()
    print('Completed:', output)
    print('Best strict validation mAP:', best)
    print('Inference-only student:', output/'weights/best.pt')
    print('Resumable model/adapters/optimizer state:', state_path)


def main():
    parser = argparse.ArgumentParser(description='Experimental YOLO11 feature-crossing focal-KD continuation, not exact paper reproduction.')
    parser.add_argument('command', choices=['smoke','train'])
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--mode', choices=['control','kd'], default='kd')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--lr', type=float, default=.0002)
    parser.add_argument('--accumulate', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--quality', type=float, default=.25)
    parser.add_argument('--lambda-cls', type=float, default=1.)
    parser.add_argument('--lambda-reg', type=float, default=1.)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.epochs < 1 or args.accumulate < 1 or args.imgsz < 64 or args.imgsz%32 or args.lr <= 0:
        parser.error('Invalid epochs, accumulation, image size, or learning rate')
    if not 0 < args.quality <= 1 or args.lambda_cls < 0 or args.lambda_reg < 0:
        parser.error('Invalid KD weights/quality')
    args.root = args.root.expanduser().resolve()
    runs = args.root/'results/clean_experiments'
    args.student = runs/'p2_yolo11n_50ep_seed42/weights/best.pt'
    args.teacher = runs/'teacher_yolo11s_p2_50ep_seed42/weights/best.pt'
    args.data = args.root/'configs/scut_dedup_v1/scut_head.yaml'
    if args.command == 'train' and not args.resume:
        output = args.root/'results/kd_continuation_v1'/args.mode
        if output.exists():
            raise SystemExit(f'Run already exists: {output}; do not restart it.')
    components = setup(args)
    if args.command == 'smoke':
        smoke(args, components)
    else:
        train(args, components)


if __name__ == '__main__':
    main()
