# The networks' plumbing, on a synthetic recording small enough for a CPU.
import numpy as np
import pytest

torch = pytest.importorskip("torch")
import deep  # noqa: E402


def test_whole_and_chunked_prediction_agree_when_a_run_fits_one_chunk():
    # Two runs of 5 and 3 windows; with chunks of 8 each run is read in one piece both ways.
    seq = np.array([1] * 5 + [2] * 3)
    E = np.random.default_rng(0).normal(size=(8, 6)).astype(np.float32)
    torch.manual_seed(0)
    model = deep.Context(n_classes=3, width=6, hidden=4).to(deep.DEVICE)
    rows = np.arange(8)
    assert deep.runs(seq, rows) == [(0, 5), (5, 8)]
    assert np.allclose(deep.context_proba(model, E, seq, rows, 8), deep.chunked_proba(model, E, seq, rows, 8), atol=1e-6)


def test_resampling_keeps_the_signal():
    t = np.linspace(0, 10, 1000)
    X = np.stack([np.sin(t), np.cos(t), np.ones_like(t)], axis=1)[None]  # one 10 s window at 100 Hz
    X30 = deep.to_30hz(X)
    assert X30.shape == (1, 3, 300)
    assert np.allclose(X30[0, 0], np.sin(np.linspace(0, 10, 300)), atol=2e-3)
