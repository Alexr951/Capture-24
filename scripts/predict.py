"""Activity timeline from a raw wrist-accelerometer recording.

    python scripts/predict.py recording.csv.gz --out results/

Input: a CSV (gzipped or not) with columns time, x, y, z: 100 Hz, in g, from a wrist-worn Axivity AX3, the format of
CAPTURE-24's participant files. Anything else is untested.
Output: <name>_timeline.csv (one row per 10 s window: start, activity, a probability per activity) and
<name>_timeline.png (the activities as a coloured band over time).
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import deep
import features as F
import har

MODEL = Path(__file__).parent.parent / "models" / "context_v1.pt"
# How much of the training class frequencies to remove from the model's output (prior ** -alpha), with the
# 5-fold cross-validation scores of each rule on CAPTURE-24 (capture24_new_model.ipynb).
RULES = {"agreement": 0.0,   # kappa 0.814, macro-F1 0.580; the rule scored on the held-out participants (kappa 0.790)
         "balanced": 0.75}   # kappa 0.790, macro-F1 0.593; finds more of the rare activities


def load_recording(path):
    """time (datetime64) and xyz (float, g) of a recording; stops if it is not the input the model was validated on."""
    raw = pd.read_csv(path, usecols=["time", "x", "y", "z"], parse_dates=["time"])
    time, xyz = raw["time"].to_numpy(), raw[["x", "y", "z"]].to_numpy(dtype=np.float32)
    step = np.median(np.diff(time)) / np.timedelta64(1, "ms")
    if abs(step - 1000 / F.SAMPLE_RATE) > 0.5:
        sys.exit(f"{path}: samples are {step:g} ms apart. The model was validated on {F.SAMPLE_RATE} Hz recordings only; resample first.")
    magnitude = np.nanmedian(np.linalg.norm(xyz, axis=1))
    if not 0.5 < magnitude < 1.5:
        print(f"warning: median vector magnitude is {magnitude:.2f}, expected about 1 g. Are the units g?", file=sys.stderr)
    if np.nanmax(np.abs(xyz)) > 8.5:
        print("warning: values beyond ±8 g; the model was validated on a ±8 g sensor.", file=sys.stderr)
    return time, xyz


def predict(time, xyz, alpha):
    """Window starts and a probability per activity (columns in the order of the model's classes)."""
    X, start, counts = F.cut_windows(time, xyz)
    if not len(X):
        sys.exit("no complete 10 s window in this recording")
    saved = torch.load(MODEL, map_location=deep.DEVICE)
    assert saved["encoder"] == deep.REPO, "the saved model was trained on another encoder version"
    model = deep.Context(len(saved["classes"]), hidden=deep.CONTEXT["hidden"]).to(deep.DEVICE)
    model.load_state_dict(saved["state"])
    E = deep.embed(deep.to_30hz(X), np.empty((len(X), deep.EMBEDDING_SIZE), dtype=np.float16))
    proba = deep.context_proba(model, E, har.sequence_ids(start, np.zeros(len(start))), np.arange(len(X)), deep.CONTEXT_LENGTH)
    proba = proba / np.array(saved["prior"]) ** alpha
    print(f"{len(X):,} windows; {counts['n_short'] + counts['n_nan']} dropped as incomplete")
    return start, proba / proba.sum(axis=1, keepdims=True), saved["classes"]


def plot_timeline(start, codes, path):
    days = np.unique(start.astype("datetime64[D]"))
    fig, axes = plt.subplots(len(days), 1, figsize=(12, 0.7 * len(days) + 1.2), layout="constrained", squeeze=False)
    for ax, day in zip(axes[:, 0], days):
        today = start.astype("datetime64[D]") == day
        har.draw_band(ax, start[today], codes[today], str(day))
        ax.set_xlim(day, day + np.timedelta64(1, "D"))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    har.class_legend(fig)
    fig.savefig(path, dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--out", type=Path, default=Path("."), help="output folder (default: here)")
    parser.add_argument("--rule", choices=RULES, default="agreement",
                        help="agreement (default): the scored model, CV kappa 0.814 and macro-F1 0.580. "
                             "balanced: finds more rare activities, CV kappa 0.790 and macro-F1 0.593.")
    args = parser.parse_args()
    if not MODEL.exists() or MODEL.read_bytes().startswith(b"version https://git-lfs"):
        sys.exit(f"{MODEL} is missing or a Git LFS pointer: run git lfs pull, or scripts/train_context.py")
    har.set_style()
    start, proba, classes = predict(*load_recording(args.recording), RULES[args.rule])
    name = args.recording.name.split(".")[0]
    args.out.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(proba, columns=classes).round(4)
    table.insert(0, "activity", np.array(classes)[proba.argmax(axis=1)])
    table.insert(0, "window_start", start)
    table.to_csv(args.out / f"{name}_timeline.csv", index=False)
    plot_timeline(start, proba.argmax(axis=1), args.out / f"{name}_timeline.png")
    print("wrote", args.out / f"{name}_timeline.csv", "and", args.out / f"{name}_timeline.png")


if __name__ == "__main__":
    main()
