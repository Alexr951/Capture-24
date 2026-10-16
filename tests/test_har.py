import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score, matthews_corrcoef

from capture24.har import metrics_from_cm, viterbi, windows_to_change


def test_metrics_by_hand():
    m = metrics_from_cm([[2, 0], [1, 1]])
    assert np.allclose(m["per_class_f1"], [0.8, 2 / 3])
    assert np.isclose(m["kappa"], 0.5)
    assert np.isclose(m["mcc"], 2 / np.sqrt(12))


def test_metrics_match_sklearn():
    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 4, 500), rng.integers(0, 4, 500)
    m = metrics_from_cm(confusion_matrix(y, p))
    assert np.isclose(m["macro_f1"], f1_score(y, p, average="macro"))
    assert np.isclose(m["kappa"], cohen_kappa_score(y, p))
    assert np.isclose(m["mcc"], matthews_corrcoef(y, p))


def test_viterbi_sticky_chain():
    trans, emit, prior = np.array([[0.9, 0.1], [0.1, 0.9]]), np.array([[0.8, 0.2], [0.2, 0.8]]), np.full(2, 0.5)

    def decode(obs):
        return viterbi(emit[:, obs].T, prior, trans).tolist()

    assert decode([0, 0, 1, 0, 0]) == [0, 0, 0, 0, 0]
    assert decode([0, 0, 1, 1, 1]) == [0, 0, 1, 1, 1]


def test_windows_to_change():
    d = windows_to_change(np.array([0, 0, 0, 1, 1, 0, 0]), np.array([0, 0, 0, 0, 0, 1, 1]))
    assert d[:5].tolist() == [2, 1, 0, 0, 1]
    assert (d[5:] >= 10**9).all()
