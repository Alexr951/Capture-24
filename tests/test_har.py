# pytest from the repository root; the data are synthetic.
import numpy as np
import pytest
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score, matthews_corrcoef

import features as F
import har


def test_hmm_decoders():
    # Sticky 2-state chain. One odd symbol costs 1/4 (0.2 vs 0.8); switching there and back costs 1/81, so stay.
    # Three odd symbols cost 1/64; one switch costs 1/9, so switch.
    hmm = {"prior": np.array([0.5, 0.5]), "trans": np.array([[0.9, 0.1], [0.1, 0.9]])}
    lik = lambda symbols: np.array([[0.8, 0.2], [0.2, 0.8]])[symbols]  # P(symbol | state), symmetric
    assert har.viterbi(lik([0, 0, 1, 0, 0]), **hmm).tolist() == [0, 0, 0, 0, 0]
    assert har.viterbi(lik([0, 0, 1, 1, 1]), **hmm).tolist() == [0, 0, 1, 1, 1]
    post = har.posterior(lik([0, 0, 1, 1, 1]), **hmm)
    assert np.allclose(post.sum(axis=1), 1) and post.argmax(axis=1).tolist() == [0, 0, 1, 1, 1]
    # Single observation: Bayes' rule gives 0.8 / (0.8 + 0.2).
    assert np.allclose(har.posterior(lik([0]), **hmm), [[0.8, 0.2]])


def test_viterbi_floor_sets_the_cost_of_an_unseen_switch():
    # State 1 was never entered from state 0 in training. Three windows of 4:1 evidence for state 1 (64:1) beat a
    # floor of 1e-1 but not a floor of 1e-16.
    hmm = {"prior": np.array([1.0, 0.0]), "trans": np.array([[1.0, 0.0], [0.0, 1.0]])}
    lik = np.array([[0.8, 0.2]] + [[0.2, 0.8]] * 3)
    assert har.viterbi(lik, **hmm, floor=1e-1).tolist() == [0, 1, 1, 1]
    assert har.viterbi(lik, **hmm).tolist() == [0, 0, 0, 0]


def test_sequences_break_at_gaps_and_participants():
    t0 = np.datetime64("2020-01-01T00:00:00")
    start = t0 + np.array([0, 10, 20, 40, 50, 0, 10]) * np.timedelta64(1, "s")  # a 20 s gap after the third window
    groups = np.array(["P1"] * 5 + ["P2"] * 2)
    assert har.sequence_ids(start, groups).tolist() == [1, 1, 1, 2, 2, 3, 3]
    # Keeping only some windows: the others get 0, and a dropped window breaks the sequence.
    keep = np.array([True, False, True, True, True, True, True])
    assert har.sequence_ids(start, groups, keep).tolist() == [1, 0, 2, 3, 3, 4, 4]


def test_fit_hmm_counts_adjacent_labelled_pairs_only():
    y = np.array([0, 0, 1, -1, 1, 1, 0])
    seq = np.array([1, 1, 1, 1, 1, 2, 2])  # the last two windows are another sequence
    hmm = har.fit_hmm(y, seq, 2)
    # Pairs: 0->0, 0->1, (1->unlabelled and unlabelled->1 skipped), (1->1 across sequences skipped), 1->0.
    assert np.allclose(hmm["trans"], [[0.5, 0.5], [1.0, 0.0]])
    assert np.allclose(hmm["prior"], [3 / 6, 3 / 6])
    with pytest.raises(AssertionError):
        har.fit_hmm(np.array([0, 0]), np.array([1, 1]), 2)  # class 1 is missing


def test_smooth_decodes_each_sequence_on_its_own():
    hmm = {"prior": np.array([0.5, 0.5]), "trans": np.array([[0.9, 0.1], [0.1, 0.9]])}
    lik = np.array([[0.8, 0.2], [0.8, 0.2], [0.2, 0.8], [0.8, 0.2], [0.8, 0.2]])
    assert har.smooth(lik, np.array([1, 1, 1, 1, 1]), hmm).tolist() == [0, 0, 0, 0, 0]  # the odd window is smoothed away
    assert har.smooth(lik, np.array([1, 1, 2, 3, 3]), hmm).tolist() == [0, 0, 1, 0, 0]  # alone in its sequence, it stands


def test_metrics_match_sklearn():
    # Hand-checked: F1 = 0.8 and 2/3; accuracy 0.75 vs chance 0.5 -> kappa 0.5; MCC = 2 / sqrt(12).
    m = har.metrics_from_cm(confusion_matrix([0, 0, 1, 1], [0, 0, 1, 0]))
    assert np.allclose([m["macro_f1"], m["kappa"], m["mcc"]], [(0.8 + 2 / 3) / 2, 0.5, 2 / 12**0.5])
    rng = np.random.default_rng(42)
    t, p = rng.integers(0, 10, 5000), rng.integers(0, 10, 5000)
    m = har.metrics_from_cm(confusion_matrix(t, p))
    assert np.allclose([m["macro_f1"], m["kappa"], m["mcc"]],
                       [f1_score(t, p, average="macro"), cohen_kappa_score(t, p), matthews_corrcoef(t, p)])


def test_paired_gain():
    # A model against itself: zero on every bootstrap draw, so only if both use the same draws.
    cms = np.random.default_rng(0).integers(0, 20, (6, 3, 3))
    assert har.paired_gain(cms, cms, ["kappa"], 50, 0) == {"kappa": {"value": 0, "ci": [0, 0]}}


def test_windows_to_change():
    # One change between windows 2 and 3; the last two windows are another sequence, without a change.
    y, seq = np.array([0, 0, 0, 1, 1, 0, 0]), np.array([1, 1, 1, 1, 1, 2, 2])
    assert har.windows_to_change(y, seq).tolist() == [2, 1, 0, 0, 1, 10**9, 10**9]


def test_windows_forest_and_hmm_end_to_end():
    # Two "participants", each lying still for 5 min and then shaking for 5 min, at 100 Hz.
    rng = np.random.default_rng(0)
    n = 600 * F.SAMPLE_RATE
    time = np.datetime64("2020-01-01T00:00:00") + np.arange(n) * np.timedelta64(10, "ms")
    annotation = np.repeat([0, 1], n // 2)
    X, y, seq = [], [], []
    for participant in range(2):
        still = np.array([0, 0, 1]) + rng.normal(0, 0.01, (n // 2, 3))
        shaking = np.array([0, 0, 1]) + rng.normal(0, 0.5, (n // 2, 3))
        windows, start, labels, counts = F.make_windows(time, np.vstack([still, shaking]), annotation, {"task": np.array([0, 1])})
        assert counts["n_labelled"] == 60 and windows.shape == (60, F.WINSEC * F.SAMPLE_RATE, 3)
        X.append(F.extract_features(windows).to_numpy())
        y.append(labels["task"])
        seq.append(har.sequence_ids(start, np.full(60, participant)) + participant)
    forest = har.balanced_forest(20, seed=0, n_jobs=1).fit(X[0], y[0])
    path = har.smooth(forest.predict_proba(X[1]), seq[1], har.fit_hmm(y[0], seq[0], 2))
    assert (path == y[1]).mean() > 0.95
