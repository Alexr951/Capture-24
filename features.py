# Windowing and window features for CAPTURE-24. A module because joblib workers import it and later rounds reuse it.
# Feature set follows https://github.com/OxWearables/capture24/blob/master/feature-list.md (own implementation)
# without the third dominant frequency, plus per-axis means/stds and inter-axis correlations.
import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.ndimage import median_filter

SAMPLE_RATE = 100  # Hz
WINSEC = 10
FEATURE_VERSION = 2  # bump when windowing or features change; it is part of the cache file name


def load_participant(path):
    """One participant's CSV -> DataFrame[time, x, y, z, annotation]."""
    return pd.read_csv(path, engine="pyarrow", parse_dates=["time"],
                       dtype={"x": "f4", "y": "f4", "z": "f4", "annotation": "category"})


def make_windows(time, xyz, annotation, class_of, winsec=WINSEC, rate=SAMPLE_RATE):
    """Cut one recording into non-overlapping windows.

    time: sorted datetime64 (N,); xyz: float (N, 3); annotation: int (N,) annotation id per sample,
    -1 = unlabelled; class_of: {scheme: int (A,)} class code of each annotation id.
    Rule: bins of `winsec` from the first sample; keep a bin if it has exactly winsec*rate samples and no NaN.
    Its label is the most frequent class among its labelled samples (ties go to the lowest code), or -1 if
    "unlabelled" is its single most frequent annotation. Unlabelled windows stay so that a sequence model sees
    the whole timeline; the labelled ones are prepare_data.py's windows minus the unlabelled-majority ones,
    which reproduces the paper's window counts.
    Returns X (W, n, 3), window_start (W,), {scheme: label (W,)}, bin counts.
    """
    n = winsec * rate
    bins = (time - time[0]) // np.timedelta64(winsec, "s")
    assert (np.diff(bins) >= 0).all(), "timestamps must be sorted"
    _, start, count = np.unique(bins, return_index=True, return_counts=True)
    full = count == n
    idx = start[full][:, None] + np.arange(n)
    X = xyz[idx]
    has_nan = np.isnan(X).any(axis=(1, 2))

    # Samples per bin for each annotation, in one pass; column 0 counts the unlabelled samples.
    k = len(next(iter(class_of.values()))) + 1
    cell = np.arange(len(idx))[:, None] * k + annotation[idx] + 1
    counts = np.bincount(cell.ravel(), minlength=len(idx) * k).reshape(-1, k)
    votes = counts[:, 1:]
    labelled = votes.max(axis=1) >= counts[:, 0]
    labels = {scheme: (votes @ np.eye(c.max() + 1, dtype=int)[c]).argmax(axis=1) for scheme, c in class_of.items()}
    keep = ~has_nan
    counts = {"n_bins": len(count), "n_short": int((~full).sum()), "n_nan": int(has_nan.sum()),
              "n_unlabelled": int((keep & ~labelled).sum()), "n_labelled": int((keep & labelled).sum())}
    return X[keep], time[start[full]][keep], {s: np.where(labelled, y, -1)[keep] for s, y in labels.items()}, counts


def _first_peak(u, rate):
    """Height and location (s) of the first peak of u with prominence >= 0.1, else zeros."""
    peaks, _ = signal.find_peaks(u, prominence=0.1)
    return (u[peaks[0]], peaks[0] / rate) if len(peaks) else (0.0, 0.0)


def extract_features(X, rate=SAMPLE_RATE):
    """X: (W, n, 3) windows in g -> DataFrame (W, 39). Vectorised across windows except peak finding."""
    X = np.asarray(X, dtype=np.float64)
    W, n, _ = X.shape
    f = {}

    # Vector magnitude, as in the reference: median filter, remove 1 g, clip.
    v = median_filter(np.linalg.norm(X, axis=2), size=(1, 5), mode="nearest")
    v = np.clip(v - 1, -2, 2)

    # Moments and quantiles of the vector magnitude.
    f["avg"], f["std"] = v.mean(axis=1), v.std(axis=1)
    vc = v - v.mean(axis=1, keepdims=True)
    z = vc / np.where(f["std"] > 0.01, f["std"], np.inf)[:, None]  # still windows get skew = kurt = 0
    f["skew"] = (z ** 3).mean(axis=1)
    f["kurt"] = np.where(f["std"] > 0.01, (z ** 4).mean(axis=1) - 3, 0)
    f["min"], f["q25"], f["med"], f["q75"], f["max"] = np.quantile(v, (0, 0.25, 0.5, 0.75, 1), axis=1)

    # Per-axis mean (gravity direction, so wrist angle), per-axis std, and correlations between axes.
    mean, sd = X.mean(axis=1), X.std(axis=1)
    for i, a in enumerate("xyz"):
        f[f"{a}_mean"], f[f"{a}_std"] = mean[:, i], sd[:, i]
    Xc = X - mean[:, None, :]
    for i, j in ((0, 1), (0, 2), (1, 2)):
        with np.errstate(invalid="ignore", divide="ignore"):
            r = (Xc[:, :, i] * Xc[:, :, j]).mean(axis=1) / (sd[:, i] * sd[:, j])
        f[f"corr_{'xyz'[i]}{'xyz'[j]}"] = np.nan_to_num(r)

    # Autocorrelation up to a 2 s lag (via FFT), normalised by lag 0.
    acf = np.fft.irfft(np.abs(np.fft.rfft(vc, 2 * n, axis=1)) ** 2, axis=1)[:, : 2 * rate + 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        acf = np.nan_to_num(acf / acf[:, :1])
    f["acf_zeros"] = np.diff(np.signbit(acf), axis=1).sum(axis=1)

    # Periodogram: spectral entropy, total power, two dominant frequencies and their power.
    freqs, P = signal.periodogram(v, fs=rate, detrend="constant", scaling="density", axis=1)
    P /= n / rate
    f["pentropy"] = np.nan_to_num(stats.entropy(P, axis=1))
    f["power"] = P.sum(axis=1)
    is_peak = np.zeros_like(P, dtype=bool)
    is_peak[:, 1:-1] = (P[:, 1:-1] > P[:, :-2]) & (P[:, 1:-1] > P[:, 2:])
    peak_power = np.where(is_peak, P, 0)
    top = np.argsort(-peak_power, axis=1)[:, :2]
    for k in range(2):
        p = np.take_along_axis(peak_power, top[:, k : k + 1], axis=1)[:, 0]
        f[f"f{k + 1}"], f[f"p{k + 1}"] = np.where(p > 0, freqs[top[:, k]], 0), p

    # Welch power at 0, 1, ..., 5 Hz. fft0 is near zero because each segment has its mean removed; it adds nothing.
    _, Pw = signal.welch(v, fs=rate, nperseg=rate, noverlap=rate // 2, detrend="constant",
                         scaling="density", average="median", axis=1)
    for k in range(6):
        f[f"fft{k}"] = Pw[:, k]

    # Peaks of the 5 Hz low-passed signal and of the autocorrelation. scipy has no batched find_peaks, so loop.
    u = signal.sosfiltfilt(signal.butter(4, 5 / (rate / 2), btype="low", output="sos"), v, axis=1)
    peak_cols = ["acf_1st_max", "acf_1st_max_loc", "acf_1st_min", "acf_1st_min_loc",
                 "npeaks", "peaks_avg_promin", "peaks_min_promin", "peaks_max_promin"]
    peak = np.zeros((W, len(peak_cols)))
    for w in range(W):
        peak[w, 0:2] = _first_peak(acf[w], rate)
        lo, loc = _first_peak(-acf[w], rate)
        peak[w, 2:4] = -lo, loc
        prom = signal.find_peaks(u[w], distance=0.2 * rate, prominence=0.25)[1]["prominences"]
        if len(prom):
            peak[w, 4:] = len(prom) / (n / rate), prom.mean(), prom.min(), prom.max()
    f.update(zip(peak_cols, peak.T))

    return pd.DataFrame(f, dtype=np.float32)


FEATURE_NAMES = list(extract_features(np.zeros((1, WINSEC * SAMPLE_RATE, 3))).columns)
