"""The pretrained ssl-wearables encoder (Yuan et al., 2024) and the networks trained on its output."""

from itertools import pairwise
from pathlib import Path

import numpy as np
import torch
from joblib import Parallel, delayed
from scipy.interpolate import interp1d
from torch.nn.functional import cross_entropy

from . import features as F

# harnet10, a ResNet for 10 s windows at 30 Hz in g, shape (3, 300). The commit pins the code and the weights,
# which ship inside the repository.
REPO = "OxWearables/ssl-wearables:150550ea5d41800229c95e36f88f5bf0d2e7cf04"
SAMPLES = 300
EMBEDDING_SIZE = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HALF = DEVICE == "cuda"  # mixed precision on the GPU only

EMBEDDINGS = Path("data") / f"embeddings_w{F.WINSEC}s_v{F.FEATURE_VERSION}_{REPO.split(':')[1][:8]}.npy"

# The training settings of part 2, fixed before any score was seen.
FINE_TUNING = {"lr": 1e-4, "batch": 1000, "patience": 5}
CONTEXT = {"hidden": 128, "lr": 1e-3, "windows_per_batch": 16384, "patience": 5}
CONTEXT_LENGTH = 64  # the chunk length the context model trains on, about 10 min


def load_encoder(n_classes):
    # torch.hub validates branch and tag names but not commits, hence skip_validation.
    return torch.hub.load(REPO, "harnet10", class_num=n_classes, pretrained=True, trust_repo=True,
                          skip_validation=True, verbose=False).to(DEVICE)


def to_30hz(X):
    """Windows (n, samples, 3) at 100 Hz -> (n, 3, 300) float16 by linear interpolation, as the encoder's authors do."""
    X = interp1d(np.linspace(0, 1, X.shape[1]), X, axis=1, assume_sorted=True)(np.linspace(0, 1, SAMPLES))
    return X.transpose(0, 2, 1).astype(np.float16)


def resample_participant(path):
    """One participant's 30 Hz windows and their start times: the windows features.make_windows keeps."""
    raw = F.load_participant(path)
    X, start, _ = F.cut_windows(raw["time"].to_numpy(), raw[["x", "y", "z"]].to_numpy())
    return to_30hz(X), start


def build_store(files, n_windows, path, start_path, jobs=4):
    """Every participant's 30 Hz windows in one memory-mappable .npy (2.5 GB), in the order of `files`."""
    store = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(n_windows, 3, SAMPLES))
    starts, row = [], 0
    for X, start in Parallel(n_jobs=jobs, return_as="generator")(delayed(resample_participant)(p) for p in files):
        store[row:row + len(X)] = X
        starts.append(start)
        row += len(X)
    store.flush()
    np.save(start_path, np.concatenate(starts))


def batches(X, rows, size):
    """(sorted row indices, tensor on the device) for consecutive slices of `rows`."""
    for i in range(0, len(rows), size):
        idx = np.sort(rows[i:i + size])
        yield idx, torch.from_numpy(X[idx]).to(DEVICE, torch.float32)


def embed(X, out, size=4096):
    """Fill `out` (n, 1024) with the frozen encoder's output for the 30 Hz windows X."""
    encoder = load_encoder(2).feature_extractor.eval()
    with torch.no_grad(), torch.autocast(DEVICE, dtype=torch.float16, enabled=HALF):
        for idx, xb in batches(X, np.arange(len(X)), size):
            out[idx] = encoder(xb).flatten(1).float().cpu().numpy()
    return out


def build_embeddings(X, path):
    """The frozen encoder's output for every window, in one memory-mappable .npy (2.9 GB)."""
    embed(X, np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(len(X), EMBEDDING_SIZE))).flush()


def early_stopping(model, train_one_epoch, validation_loss, patience, max_epochs):
    """Train until the validation loss has not improved for `patience` epochs; return the model from the best epoch."""
    best, best_state, stale = np.inf, None, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_one_epoch()
        loss = validation_loss()
        print(f"epoch {epoch}: validation loss {loss:.4f}")
        if loss < best:
            best, stale = loss, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
            if stale == patience:
                break
    model.load_state_dict(best_state)
    return model


def fine_tune(X, y, train, val, seed, lr, batch, patience):
    """Fine-tune every layer of the encoder on rows `train`, stopping early on rows `val`."""
    k = int(y[train].max()) + 1
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = load_encoder(k)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    scaler = torch.amp.GradScaler(enabled=HALF)

    def train_one_epoch():
        order = rng.permutation(train)
        for idx, xb in batches(X, order[:len(order) // batch * batch], batch):
            with torch.autocast(DEVICE, dtype=torch.float16, enabled=HALF):
                loss = cross_entropy(model(xb), torch.from_numpy(y[idx]).long().to(DEVICE))
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    def validation_loss():
        return cross_entropy(encoder_logits(model, X, val), torch.from_numpy(y[val]).long()).item()

    return early_stopping(model, train_one_epoch, validation_loss, patience, max_epochs=50)


def encoder_logits(model, X, rows, size=4096):
    """Logits (float32, CPU) for `rows`, which must be sorted."""
    model.eval()
    with torch.no_grad(), torch.autocast(DEVICE, dtype=torch.float16, enabled=HALF):
        return torch.cat([model(xb).float().cpu() for _, xb in batches(X, rows, size)])


def encoder_proba(model, X, rows):
    return torch.softmax(encoder_logits(model, X, rows), dim=1).numpy()


class Context(torch.nn.Module):
    """A 2-layer bidirectional GRU over the inputs of consecutive windows: (sequences, windows, width) -> logits."""

    def __init__(self, n_classes, width=EMBEDDING_SIZE, hidden=128):
        super().__init__()
        self.drop = torch.nn.Dropout(0.5)
        self.gru = torch.nn.GRU(width, hidden, num_layers=2, bidirectional=True, batch_first=True, dropout=0.5)
        self.out = torch.nn.Linear(2 * hidden, n_classes)

    def forward(self, x):
        return self.out(self.gru(self.drop(x))[0])


def runs(seq, rows):
    """(first row, last row + 1) of each run of consecutive windows in `rows`, which must hold whole runs."""
    edges = np.flatnonzero(np.r_[True, seq[rows][1:] != seq[rows][:-1], True])
    return [(rows[a], rows[b - 1] + 1) for a, b in pairwise(edges)]


def context_logits(model, E, seq, rows, length):
    """Logits for every window of `rows`. A context model (length > 1) reads each whole run of consecutive windows;
    a single-window model (length 1) reads every window on its own."""
    model.eval()
    out = []
    with torch.no_grad():
        for a, b in runs(seq, rows):
            x = torch.from_numpy(np.array(E[a:b])).to(DEVICE, torch.float32)
            out.append(model(x[None] if length > 1 else x[:, None]).reshape(b - a, -1).cpu())
    return torch.cat(out)


def context_proba(model, E, seq, rows, length):
    return torch.softmax(context_logits(model, E, seq, rows, length), dim=1).numpy()


def fit_context(E, y, seq, train, val, length, seed, hidden, lr, windows_per_batch, patience):
    """Train Context on chunks of `length` consecutive windows of rows `train`, a new chunk every length // 2 windows,
    stopping early on rows `val`. Unlabelled windows (y = -1) are read but carry no loss. Runs shorter than `length`
    are left out of training."""
    k = int(y[train].max()) + 1
    offsets = np.arange(length)
    starts = np.concatenate([np.unique(np.r_[np.arange(a, b - length + 1, max(length // 2, 1)), b - length])
                             for a, b in runs(seq, train) if b - a >= length])
    starts = starts[(y[starts[:, None] + offsets] >= 0).any(axis=1)]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Context(k, E.shape[1], hidden).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    per_batch = windows_per_batch // length

    def train_one_epoch():
        order = rng.permutation(starts)
        for i in range(0, len(order), per_batch):
            idx = np.sort(order[i:i + per_batch])[:, None] + offsets
            logits = model(torch.from_numpy(E[idx]).to(DEVICE, torch.float32))
            loss = cross_entropy(logits.reshape(-1, k), torch.from_numpy(y[idx]).long().to(DEVICE).reshape(-1), ignore_index=-1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def validation_loss():
        return cross_entropy(context_logits(model, E, seq, val, length), torch.from_numpy(y[val]).long(), ignore_index=-1).item()

    return early_stopping(model, train_one_epoch, validation_loss, patience, max_epochs=100)
