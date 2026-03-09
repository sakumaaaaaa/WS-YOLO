import os
import random
import warnings
import argparse
from pathlib import Path

import yaml

warnings.filterwarnings('ignore')
from ultralytics import YOLO

IMG_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def _img2label(p):
    """images/xxx.jpg -> labels/xxx.txt, works on both posix and nt paths."""
    s = str(p)
    # ultralytics convention: swap the last /images/ segment with /labels/
    for sep in ('/' , os.sep):
        s = s.replace(f'{sep}images{sep}', f'{sep}labels{sep}', 1)
    return Path(s).with_suffix('.txt')


def collect_images(data_cfg):
    """Walk every split dir listed in *data_cfg* and return deduplicated image paths."""
    root = Path(data_cfg['path'])
    imgs = set()
    for key in ('train', 'val', 'test'):
        d = data_cfg.get(key)
        if d is None:
            continue
        d = root / d
        if d.is_dir():
            imgs |= {p for p in d.iterdir() if p.suffix.lower() in IMG_SUFFIXES}
    return sorted(imgs)


def resplit(data_yaml, ratios=(0.7, 0.2, 0.1), seed=None):
    """
    Re-shuffle all images under *data_yaml* and write ``train.txt / val.txt /
    test.txt`` so YOLO can consume them directly.  Returns path to the
    generated ``data_random.yaml``.
    """
    with open(data_yaml, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    root = Path(cfg['path'])
    images = collect_images(cfg)
    assert images, f'no images found under {root}'

    # drop samples whose label file is missing
    paired = [p for p in images if _img2label(p).exists()]
    n_drop = len(images) - len(paired)
    if n_drop:
        print(f'[warn] {n_drop} image(s) have no label file, skipped')
    images = paired

    seed = seed if seed is not None else random.randrange(10 ** 6)
    rng = random.Random(seed)
    rng.shuffle(images)

    n = len(images)
    i = int(n * ratios[0])
    j = i + int(n * ratios[1])
    parts = {'train': images[:i], 'val': images[i:j], 'test': images[j:]}

    out = root / 'random_splits'
    out.mkdir(exist_ok=True)
    for name, lst in parts.items():
        (out / f'{name}.txt').write_text('\n'.join(str(p) for p in lst))

    new_cfg = {
        'path':  str(root),
        'train': 'random_splits/train.txt',
        'val':   'random_splits/val.txt',
        'test':  'random_splits/test.txt',
        'nc':    cfg['nc'],
        'names': cfg['names'],
    }
    dst = root / 'data_random.yaml'
    with open(dst, 'w', encoding='utf-8') as f:
        yaml.safe_dump(new_cfg, f, sort_keys=False, allow_unicode=True)

    tr, va, te = len(parts['train']), len(parts['val']), len(parts['test'])
    print(f'split {n} images -> train {tr} / val {va} / test {te}  (seed={seed})')
    return str(dst)


def parse_args():
    p = argparse.ArgumentParser(description='WS-YOLO training script')
    p.add_argument('--cfg',    default='cfg/models/11/yolo11-wsyolo.yaml', help='model yaml')
    p.add_argument('--data',   default='/root/autodl-tmp/data/wafer.yaml', help='dataset yaml')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch',  type=int, default=16)
    p.add_argument('--imgsz',  type=int, default=640)
    p.add_argument('--device', default='0')
    p.add_argument('--project', default='runs/train')
    p.add_argument('--name',   default='exp')
    p.add_argument('--weights', default='', help='pretrained weights path')
    # random split options
    p.add_argument('--no-resplit', action='store_true', help='use original fixed splits instead of random re-shuffle')
    p.add_argument('--split-ratio', nargs=3, type=float, default=[0.7, 0.3, 0.0],
                   metavar=('TRAIN', 'VAL', 'TEST'))
    p.add_argument('--seed', type=int, default=None, help='split seed (omit for random)')
    return p.parse_args()


def main():
    args = parse_args()

    data = args.data
    if not args.no_resplit:
        data = resplit(data, tuple(args.split_ratio), args.seed)

    model = YOLO(args.cfg)
    if args.weights:
        model.load(args.weights)

    model.train(
        data=data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        optimizer='SGD',
        patience=100,
        cache=False,
        workers=2,
        project=args.project,
        name=args.name,
        verbose=True,
        plots=True,
    )


if __name__ == '__main__':
    main()
