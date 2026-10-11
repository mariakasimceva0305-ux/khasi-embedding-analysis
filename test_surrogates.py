import numpy as np
from analyse import iaaft, asymmetry, longest_known


def test_iaaft_preserves_marginal_and_seed():
    x = np.random.default_rng(12).normal(size=256)
    a, e = iaaft(x, np.random.default_rng(2))
    b, _ = iaaft(x, np.random.default_rng(2))
    np.testing.assert_array_equal(np.sort(a), np.sort(x))
    np.testing.assert_array_equal(a, b)
    assert e < 0.05


def test_time_reversal_changes_sign():
    x = np.array([0.0, 1.0, 2.0, 8.0, 1.0])
    assert np.isclose(asymmetry(x), -asymmetry(x[::-1]))


def test_oov_does_not_join_unrelated_segments():
    assert longest_known(["a", "b", "missing", "a", "b", "a"], {"a": 0, "b": 1}) == [
        0,
        1,
        0,
    ]
