"""
C1 benchmark: Sinhala glyph classifier (EfficientNetV2-S, Work/Janindu).

Rebuilds the exact validation split the trainer used (ImageFolder order,
stratified 80/20, random_state 42) and scores best_model.pth on it.

Reports accuracy, top-5 accuracy, macro / weighted precision, recall and F1,
a breakdown by font difficulty (Basic / Intermediate / Hard, from the file
name), the worst classes and the most common confusions.

Run with Janindu's model venv (it has torch + CUDA):
    Work/Janindu/0_Data/Model/venv/Scripts/python.exe system/tools/bench/bench_c1_glyph.py
Output: system/docs/eval/c1_glyph.json and c1_glyph_per_class.csv
"""
import csv
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict

import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
DATA = os.path.join(ROOT, 'Work', 'Janindu', '0_Data', 'SkeletonImages', 'Skeletons')
MODEL = os.path.join(ROOT, 'Work', 'Janindu', '0_Data', 'Model')
OUT = os.path.join(ROOT, 'system', 'docs', 'eval')

# Same transform and split as Work/Janindu/0_Data/Model/trainer.py
TF = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def loader(path):
    try:
        return Image.open(path).convert('RGB')
    except Exception:
        return Image.new('RGB', (384, 384), color=(255, 255, 255))


def tier(path):
    name = os.path.basename(path)
    for t in ('Basic', 'Intermediate', 'Hard'):
        if f'_{t}_' in name:
            return t
    return 'other'


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = datasets.ImageFolder(root=DATA, loader=loader, transform=TF)
    _, val_idx = train_test_split(list(range(len(ds))), test_size=0.2,
                                  stratify=ds.targets, random_state=42)
    val = Subset(ds, val_idx)
    names = json.load(open(os.path.join(MODEL, 'class_mapping.json'), encoding='utf-8'))
    names = [names[str(i)] for i in range(len(names))]

    ck = torch.load(os.path.join(MODEL, 'best_model.pth'), map_location='cpu', weights_only=False)
    m = models.efficientnet_v2_s(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, ck['num_classes'])
    m.load_state_dict(ck['model_state_dict'])
    m.to(dev).eval()

    y, p, top5 = [], [], 0
    t0 = time.time()
    dl = DataLoader(val, batch_size=64, shuffle=False, num_workers=4)
    with torch.no_grad(), torch.autocast(dev.type, enabled=dev.type == 'cuda'):
        for i, (x, t) in enumerate(dl):
            out = m(x.to(dev))
            p += out.argmax(1).cpu().tolist()
            top5 += (out.topk(5, 1).indices.cpu() == t[:, None]).any(1).sum().item()
            y += t.tolist()
            if i % 50 == 0:
                print(f'{len(y)}/{len(val)}', flush=True)
    secs = time.time() - t0

    n = len(y)
    acc = sum(a == b for a, b in zip(y, p)) / n
    P, R, F, S = precision_recall_fscore_support(y, p, labels=range(len(names)), zero_division=0)
    mp, mr, mf, _ = precision_recall_fscore_support(y, p, average='macro', zero_division=0)
    wp, wr, wf, _ = precision_recall_fscore_support(y, p, average='weighted', zero_division=0)

    paths = [ds.samples[i][0] for i in val_idx]
    by_tier = defaultdict(lambda: [0, 0])
    for path, a, b in zip(paths, y, p):
        by_tier[tier(path)][0] += 1
        by_tier[tier(path)][1] += a == b

    # The same pili (vowel sign) exists under two class ids, e.g.
    # pili_kombuDouble_0006 and pili_kombuDouble_0303. Merge by name without
    # the numeric suffix: predicting either copy is the same character.
    base = [re.sub(r'_\d{4}$', '', c) for c in names]
    dup = sorted({b for b in base if base.count(b) > 1})
    yb, pb = [base[a] for a in y], [base[b] for b in p]
    bl = sorted(set(base))
    _, _, Fm, _ = precision_recall_fscore_support(yb, pb, labels=bl, zero_division=0)
    merged = {'n_classes': len(bl), 'duplicated_labels': dup,
              'accuracy': round(sum(a == b for a, b in zip(yb, pb)) / n, 4),
              'macro_f1': round(float(Fm.mean()), 4),
              'classes_f1_below_0.95': int((Fm < 0.95).sum())}
    conf = Counter((names[a], names[b]) for a, b in zip(y, p) if a != b)
    res = {
        'model': 'EfficientNetV2-S, best_model.pth (epoch %s)' % ck.get('epoch'),
        'split': 'stratified 80/20, random_state 42 (same as trainer.py)',
        'n_val': n, 'n_classes': len(names),
        'accuracy': round(acc, 4), 'top5_accuracy': round(top5 / n, 4),
        'macro': {'precision': round(mp, 4), 'recall': round(mr, 4), 'f1': round(mf, 4)},
        'weighted': {'precision': round(wp, 4), 'recall': round(wr, 4), 'f1': round(wf, 4)},
        'merged_duplicate_labels': merged,
        'by_difficulty': {k: {'n': v[0], 'accuracy': round(v[1] / v[0], 4)}
                          for k, v in sorted(by_tier.items())},
        'classes_f1_below_0.95': sum(f < 0.95 for f in F),
        'worst_classes': sorted(({'class': names[i], 'n': int(S[i]), 'f1': round(F[i], 3)}
                                 for i in range(len(names))), key=lambda r: r['f1'])[:10],
        'top_confusions': [{'true': a, 'pred': b, 'count': c} for (a, b), c in conf.most_common(10)],
        'inference': {'device': str(dev), 'images_per_sec': round(n / secs, 1)},
    }
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, 'c1_glyph.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2, default=lambda o: o.item())
    with open(os.path.join(OUT, 'c1_glyph_per_class.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['class', 'support', 'precision', 'recall', 'f1'])
        for i, c in enumerate(names):
            w.writerow([c, int(S[i]), round(P[i], 4), round(R[i], 4), round(F[i], 4)])
    print(json.dumps(res, ensure_ascii=False, indent=2, default=lambda o: o.item()))


if __name__ == '__main__':
    main()
