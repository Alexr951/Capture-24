"""Fit the Context model on P001-P100 and save it for predict.py:  python train_context.py

The configuration scored on the held-out participants in capture24_new_model.ipynb: the frozen encoder, chunks of
64 windows, seed 42, early stopping on a fifth of the training participants. The scored fit's weights were not kept,
so this refits it and compares the refit's top class with the scored fit's on the held-out windows, without labels.
Needs the caches that notebook builds in data/.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import deep
import features as F
import har

SEED, LENGTH = 42, deep.CONTEXT_LENGTH
DATA, MODEL = Path("data"), Path("models/context_v1.pt")

windows = pd.read_parquet(DATA / f"features_w{F.WINSEC}s_v{F.FEATURE_VERSION}.parquet", columns=["participant", "window_start", "10-class"])
y, groups = windows["10-class"].to_numpy(), windows["participant"].to_numpy()
seq = har.sequence_ids(windows["window_start"].to_numpy(), groups)
E = np.load(deep.EMBEDDINGS, mmap_mode="r")
dev = np.flatnonzero(groups <= "P100")
labels = y[dev][y[dev] >= 0]
fit, val = har.early_stopping_split(y, groups, dev, SEED)

model = deep.fit_context(E, y, seq, fit, val, LENGTH, SEED, **deep.CONTEXT)
MODEL.parent.mkdir(exist_ok=True)
torch.save({"state": model.state_dict(), "classes": har.CODES, "encoder": deep.REPO,
            "prior": (np.bincount(labels) / len(labels)).tolist()}, MODEL)
print("saved", MODEL)

test = np.flatnonzero(groups > "P100")
for scored in (DATA / "fits").glob(f"context_encoder_{LENGTH}_holdout_{SEED}_*.npy"):
    same = deep.context_proba(model, E, seq, test, LENGTH).argmax(axis=1) == np.load(scored).argmax(axis=1)
    print(f"top class agrees with the scored fit on {same.mean():.1%} of held-out windows")
