# Included for provenance only: this is the exact script that produced
# emb_full_A144.npy / emb_full_index.csv (pinned by SHA-256 in config.yaml).
# Not run or verified by anything in this package -- needs a GPU and an HF
# token, and model/CUDA nondeterminism means it isn't byte-reproducible.
#
# Paste into a Kaggle notebook (GPU T4 x1, ~15 min). Upload work/kaggle_pkg_full/
# as a dataset first, enable the HF_TOKEN secret, and edit PKG below.
import collections
import csv
import os

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoImageProcessor

PKG = '/kaggle/input/<your-dataset-slug>/kaggle_pkg_full'
OUT = '/kaggle/working'
LOW = 144
MODEL = 'facebook/dinov3-vitb16-pretrain-lvd1689m'

try:
    from kaggle_secrets import UserSecretsClient
    HF = UserSecretsClient().get_secret("HF_TOKEN")
    from huggingface_hub import login
    login(token=HF, add_to_git_credential=False)
except Exception as e:
    raise SystemExit(f'HF_TOKEN not attached to this notebook: {e}')

rows = list(csv.DictReader(open(f'{PKG}/manifest_full.csv')))
missing = [r['file'] for r in rows if not os.path.exists(f'{PKG}/images/{r["file"]}')]
empty = [r['file'] for r in rows if os.path.exists(f'{PKG}/images/{r["file"]}')
         and os.path.getsize(f'{PKG}/images/{r["file"]}') == 0]
assert not missing and not empty, f'package incomplete: {(missing + empty)[:5]}'
assert len(rows) == 26527, f'expected 26,527, got {len(rows)}'
print(dict(collections.Counter(r['cls'] for r in rows)))

BAD_FILES = []


class Imgs(Dataset):
    def __init__(self, rows, proc, size):
        self.rows, self.proc, self.size = rows, proc, size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        f = f'{PKG}/images/{self.rows[i]["file"]}'
        try:
            im = ImageOps.exif_transpose(Image.open(f)).convert('RGB')
        except Exception as e:
            print(f'   !! unreadable, substituting grey: {os.path.basename(f)} ({type(e).__name__})', flush=True)
            BAD_FILES.append(self.rows[i]['file'])
            im = Image.new('RGB', (self.size, self.size), (128, 128, 128))
        im = im.resize((self.size, self.size), Image.BICUBIC)
        return self.proc(images=im, return_tensors='pt',
                          do_resize=False, do_center_crop=False)['pixel_values'][0]


proc = AutoImageProcessor.from_pretrained(MODEL, token=HF, use_fast=True)
try:
    model = AutoModel.from_pretrained(MODEL, dtype=torch.float16, token=HF)
except TypeError:
    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float16, token=HF)
model = model.to('cuda').eval()


@torch.no_grad()
def run(rows, bs=64):
    dl = DataLoader(Imgs(rows, proc, LOW), batch_size=bs, num_workers=2, shuffle=False)
    out, pooled = [], None
    for i, px in enumerate(dl):
        o = model(pixel_values=px.to('cuda', torch.float16), interpolate_pos_encoding=True)
        if pooled is None:
            pooled = 'pooler' if getattr(o, 'pooler_output', None) is not None else 'cls'
            print(f'   pooling={pooled}  tokens={o.last_hidden_state.shape[1]}', flush=True)
        v = o.pooler_output if pooled == 'pooler' else o.last_hidden_state[:, 0]
        out.append(torch.nn.functional.normalize(v, dim=-1).float().cpu().numpy())
        if i % 50 == 0:
            print(f'   batch {i}/{len(dl)}', flush=True)
    return np.concatenate(out).astype(np.float16)


emb = run(rows)
np.save(f'{OUT}/emb_full_A144.npy', emb)

with open(f'{OUT}/emb_full_index.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print(f'\nemb_full_A144.npy  shape={emb.shape}  {os.path.getsize(f"{OUT}/emb_full_A144.npy") / 1e6:.1f} MB')
print(f'finite={np.isfinite(emb).all()}  |row0|={np.linalg.norm(emb[0].astype(np.float32)):.3f}')

e = np.array([r['cls'] == '1_Electronic' for r in rows])
print(f'Electronic rows in this run: {e.sum()} (expect 3961)')
if BAD_FILES:
    print(f'\n!! {len(BAD_FILES)} files substituted with grey: {BAD_FILES[:10]}')
else:
    print('\nall 26,527 images decoded cleanly')
print('\nDONE -- download emb_full_A144.npy + emb_full_index.csv into work/')
