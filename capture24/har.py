# Shared by both notebooks: folds, the forest, the HMM, metrics, bootstraps and the shared plots.
from itertools import pairwise

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from matplotlib.patches import Patch
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from . import features as F

CLASSES = ["sleep", "sitting", "standing", "household-chores", "manual-work",
           "walking", "mixed-activity", "vehicle", "sports", "bicycling"]  # the 10 activities, display order
CODES = sorted(CLASSES)  # class codes are alphabetical
SHOWN = {"macro_f1": "macro-F1", "kappa": "κ"}  # the metrics reported; MCC stays within 0.005 of κ
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
BLUE, ORANGE, GREEN, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
CLASS_COLORS = dict(zip(CLASSES, ["#52514e", "#c3c2b7", BLUE, ORANGE, GREEN, "#eda100", "#e87ba4", "#008300", PURPLE, "#e34948"]))
TINY = 1e-16  # keeps log(0) finite


def set_style():
    plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": "#c3c2b7",
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                         "axes.axisbelow": True, "text.color": INK, "axes.labelcolor": "#52514e",
                         "xtick.color": MUTED, "ytick.color": MUTED, "legend.frameon": False, "figure.dpi": 110})


def draw_band(ax, start, codes, name):
    """A row of 10 s blocks coloured by 10-class code."""
    for label in CLASSES:
        t = start[codes == CODES.index(label)]
        if len(t):
            ax.bar(t, 1, width=pd.Timedelta(seconds=F.WINSEC), align="edge", color=CLASS_COLORS[label])
    ax.set(yticks=[], ylim=(0, 1))
    ax.set_ylabel(name, rotation=0, ha="right", va="center")
    ax.grid(False)


def class_legend(fig):
    fig.legend(handles=[Patch(color=c, label=n) for n, c in CLASS_COLORS.items()],
               loc="outside lower center", ncols=10, handlelength=1, columnspacing=1)


def plot_bands(start, groups, bands, pid="P101", first_hour=8, hours=6):
    """Rows of activity bands (name -> codes, one per window) for `hours` of participant `pid`, from `first_hour` on
    their first day. `start`, `groups` and the codes cover the same windows."""
    begin = start[groups == pid].min().astype("datetime64[D]") + np.timedelta64(first_hour, "h")
    shown = (groups == pid) & (start >= begin) & (start < begin + np.timedelta64(hours, "h"))
    fig, axes = plt.subplots(len(bands), 1, sharex=True, figsize=(12, 0.6 + 0.65 * len(bands)), layout="constrained")
    for ax, (name, codes) in zip(axes, bands.items()):
        draw_band(ax, start[shown], codes[shown], name)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    class_legend(fig)
    return fig


def save_for_paper(fig, path):
    """Save `fig` for the paper, without the titles its caption replaces and on white, then close it."""
    if fig._suptitle is not None:
        fig._suptitle.remove()
        fig._suptitle = None
    fig.set_facecolor("white")
    for ax in fig.axes:
        ax.set_title("")
        ax.set_facecolor("white")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def label_bars(ax, values, whiskers, fmt):
    """Write each horizontal bar's value just past its whisker."""
    for i, (value, whisker) in enumerate(zip(values, whiskers)):
        side = 1 if value >= 0 else -1
        ax.annotate(format(value, fmt), (value + side * whisker, i), xytext=(6 * side, 0), textcoords="offset points",
                    va="center", ha="left" if side > 0 else "right", fontsize=9)


def plot_holdout(models, published, title):
    """Held-out macro-F1 and κ: bars with 95% intervals and diamonds at the cross-validation means, then published
    numbers in grey. models: name -> (held-out scores, cross-validation means, colour); published: name -> (macro-F1, κ)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True, layout="constrained")
    for ax, (i, (metric, label)) in zip(axes, enumerate(SHOWN.items())):
        value = np.array([held[metric]["value"] for held, _, _ in models.values()])
        ci = np.array([held[metric]["ci"] for held, _, _ in models.values()]).T
        values = [*value, *(numbers[i] for numbers in published.values())]
        ax.barh([*models, *published], values, color=[c for *_, c in models.values()] + [MUTED] * len(published), height=0.6)
        ax.errorbar(value, range(len(models)), xerr=np.abs(ci - value), fmt="none", ecolor=INK, capsize=3)
        ax.scatter([cv[metric] for _, cv, _ in models.values()], range(len(models)), marker="D", s=28, color="white",
                   edgecolor=INK, zorder=3, label="5-fold cross-validation mean")
        label_bars(ax, values, [*(ci[1] - value), *[0] * len(published)], ".3f")
        ax.set(xlabel=label, xlim=(0, 0.9))
        ax.grid(axis="y", visible=False)
    axes[0].invert_yaxis()
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower right")
    fig.suptitle(title)
    return fig


def rows_of(groups, participants):
    """Every window of `participants` (np.unique first: np.isin on strings is slow against repeats)."""
    return np.flatnonzero(np.isin(groups, np.unique(participants)))


def participant_folds(y, groups, rows, n_folds, seed):
    """Participant-grouped folds over `rows`, stratified on y: yields (training rows, validation rows), each holding
    every window of its participants."""
    for split in StratifiedGroupKFold(n_folds, shuffle=True, random_state=seed).split(rows, y[rows], groups[rows]):
        yield tuple(rows_of(groups, groups[rows[part]]) for part in split)


def early_stopping_split(y, groups, train, seed):
    """(fit rows, validation rows): every window of four fifths of the training participants, and of the other fifth,
    held out for early stopping."""
    labelled = train[y[train] >= 0]
    split = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed).split(labelled, groups=groups[labelled]))
    return tuple(rows_of(groups, groups[labelled[part]]) for part in split)


def balanced_forest(n_trees, seed, **kwargs):
    """imblearn's balanced random forest as models A and B use it; `kwargs` override."""
    return BalancedRandomForestClassifier(**{"n_estimators": n_trees, "sampling_strategy": "not minority", "replacement": True,
                                             "bootstrap": True, "n_jobs": -1, "random_state": seed, **kwargs})


def viterbi(lik, prior, trans, floor=TINY):
    """Most likely state path. lik[t, state] is proportional to P(observation t | state); trans[from, to].
    `floor` is what a transition that was never counted gets instead of zero."""
    log_lik, log_trans = np.log(lik + TINY), np.log(trans + floor)
    score = np.log(prior + floor) + log_lik[0]
    back = np.zeros(lik.shape, dtype=int)
    for t in range(1, len(lik)):
        step = score[:, None] + log_trans  # step[i, j]: best path ending in i, then moving to j
        back[t] = step.argmax(axis=0)
        score = step.max(axis=0) + log_lik[t]
    path = np.empty(len(lik), dtype=int)
    path[-1] = score.argmax()
    for t in range(len(lik) - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def sequence_ids(start, groups, keep=None):
    """Sequence id of each kept window (0 for the others): a new sequence at each participant and wherever kept
    windows are not 10 s apart. Windows must be sorted by participant, then time."""
    keep = np.ones(len(start), dtype=bool) if keep is None else keep
    s, g = start[keep], groups[keep]
    new = np.r_[True, (np.diff(s) != np.timedelta64(F.WINSEC, "s")) | (g[1:] != g[:-1])]
    ids = np.zeros(len(start), dtype=int)
    ids[keep] = np.cumsum(new)
    return ids


def fit_hmm(y, seq, k):
    """Class prior and transition matrix from consecutive training windows `y` (-1 = unlabelled).
    Transitions are counted between adjacent labelled windows of a sequence."""
    pair = (seq[1:] == seq[:-1]) & (y[:-1] >= 0) & (y[1:] >= 0)
    trans = np.zeros((k, k))
    np.add.at(trans, (y[:-1][pair], y[1:][pair]), 1)
    count = np.bincount(y[y >= 0], minlength=k)
    assert count.all() and trans.sum(axis=1).all(), "a class is missing from training"
    return {"prior": count / count.sum(), "trans": trans / trans.sum(axis=1, keepdims=True)}


def smooth(lik, seq, hmm):
    """Viterbi path of each sequence on its own."""
    edges = np.flatnonzero(np.r_[True, seq[1:] != seq[:-1], True])
    return np.concatenate([viterbi(lik[a:b], **hmm) for a, b in pairwise(edges)])


def metrics_from_cm(cm):
    """Confusion matrix (rows = truth) -> macro-F1, Cohen's kappa, multiclass MCC and per-class F1."""
    cm = np.asarray(cm, dtype=float)
    n, true, pred, hit = cm.sum(), cm.sum(axis=1), cm.sum(axis=0), np.diag(cm)
    seen = true + pred > 0  # like sklearn, average F1 over classes that occur in truth or prediction
    f1 = np.where(seen, 2 * hit / np.where(seen, true + pred, 1), np.nan)
    chance = true @ pred
    mcc_den = np.sqrt((n**2 - pred @ pred) * (n**2 - true @ true))
    return {"macro_f1": float(np.nanmean(f1)),
            "kappa": float((n * hit.sum() - chance) / (n**2 - chance)),
            "mcc": float((n * hit.sum() - chance) / mcc_den) if mcc_den else 0.0,
            "per_class_f1": f1.tolist()}


def participant_cms(y_true, y_pred, groups, k):
    """One confusion matrix per participant, shape (participants, k, k)."""
    g = pd.factorize(groups)[0]
    cms = np.zeros((g.max() + 1, k, k))
    np.add.at(cms, (g, y_true, y_pred), 1)
    return cms


def interval(value, draws):
    """A value with the 95% percentile interval of its bootstrap draws."""
    return {"value": value, "ci": [draws.quantile(0.025), draws.quantile(0.975)]}


def bootstrap_metrics(cms, n_boot, seed):
    """Metrics of each bootstrap draw of participants, from their confusion matrices. The draws depend only on the
    seed and the number of participants, so two models' draws can be subtracted."""
    draws = np.random.default_rng(seed).integers(0, len(cms), (n_boot, len(cms)))
    return pd.DataFrame([metrics_from_cm(cms[d].sum(axis=0)) for d in draws]).drop(columns="per_class_f1")


def paired_gain(cms, base_cms, metrics, n_boot, seed):
    """Metrics of one model minus another's, from their per-participant confusion matrices on the same windows, with
    a 95% interval from the same bootstrap draws of participants."""
    point, base = metrics_from_cm(cms.sum(axis=0)), metrics_from_cm(base_cms.sum(axis=0))
    gain = bootstrap_metrics(cms, n_boot, seed) - bootstrap_metrics(base_cms, n_boot, seed)
    return {m: interval(point[m] - base[m], gain[m]) for m in metrics}


def windows_to_change(y, seq):
    """For each window, how many windows lie between it and the nearest change of label in its sequence
    (0 = next to one; a large number where its sequence has no change)."""
    edge = np.flatnonzero((seq[1:] == seq[:-1]) & (y[1:] != y[:-1]))  # a change between window i and i + 1
    idx, far = np.arange(len(y)), 10**9
    pos = np.searchsorted(edge, idx)
    after, before = edge[np.minimum(pos, len(edge) - 1)], edge[np.maximum(pos - 1, 0)]
    right = np.where((pos < len(edge)) & (seq[after] == seq), after - idx, far)
    left = np.where((pos > 0) & (seq[before] == seq), idx - before - 1, far)
    return np.minimum(left, right)
