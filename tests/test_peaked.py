import numpy as np

import peaked


def test_mirror_circuit_is_exactly_peaked():
    n = 8
    gates, peak = peaked.mirror_circuit(n, half_depth=4, seed=0)
    probs = np.abs(peaked.run_exact(n, gates)) ** 2
    assert np.isclose(probs[int(peak, 2)], 1)


def test_mps_with_full_bond_matches_exact_state():
    n, rng = 8, np.random.default_rng(1)
    gates, _ = peaked.mirror_circuit(n, half_depth=5, seed=1)
    gates = gates[: len(gates) // 2]                      # forward half only
    exact = np.abs(peaked.run_exact(n, gates)) ** 2
    mps = peaked.run_mps(n, gates, max_bond=2 ** (n // 2))
    assert np.isclose(mps.kept, 1)
    for idx in rng.integers(0, 2**n, size=20):
        assert np.isclose(mps.probability(format(idx, f"0{n}b")), exact[idx], atol=1e-10)


def test_truncation_loses_the_peak_and_full_bond_keeps_it():
    n = 10
    gates, peak = peaked.mirror_circuit(n, half_depth=8, seed=2)
    assert np.isclose(peaked.run_mps(n, gates, max_bond=32).probability(peak), 1)
    assert peaked.run_mps(n, gates, max_bond=2).probability(peak) < 0.5


def test_sampling_returns_the_peak_and_ghz_gives_two_outcomes():
    n = 8
    gates, peak = peaked.mirror_circuit(n, half_depth=3, seed=3)
    assert peaked.run_mps(n, gates, 16).sample(np.random.default_rng(0)) == peak
    h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    cnot = np.eye(4)[[0, 1, 3, 2]].astype(complex)
    ghz = [(h, (0,))] + [(cnot, (i, i + 1)) for i in range(n - 1)]
    mps = peaked.run_mps(n, ghz, max_bond=2)
    rng = np.random.default_rng(0)
    assert {mps.sample(rng) for _ in range(40)} == {"0" * n, "1" * n}
    assert np.isclose(mps.kept, 1) and mps.largest_bond == 2
