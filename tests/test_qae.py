import numpy as np

import qae


def test_simulator_matches_textbook_formula():
    marked = qae.make_catalogue(10)
    a = marked.mean()
    for k in [0, 1, 2, 5, 13]:
        assert np.isclose(qae.p_marked(marked, k), qae.p_theory(a, k), atol=1e-10)


def test_state_stays_normalized():
    marked = qae.make_catalogue(8)
    assert np.isclose(np.linalg.norm(qae.grover_state(marked, 7)), 1)


def test_mlae_recovers_fraction_and_beats_monte_carlo_at_equal_queries():
    marked = qae.make_catalogue(10)
    a = marked.mean()
    ks, shots = qae.schedule(64), 32
    probs = {k: qae.p_marked(marked, k) for k in ks}
    n = qae.queries(ks, shots)
    rng = np.random.default_rng(0)
    err_q = np.sqrt(np.mean([(qae.mlae(probs, shots, rng) - a) ** 2 for _ in range(100)]))
    err_c = np.sqrt(np.mean([(qae.monte_carlo(a, n, rng) - a) ** 2 for _ in range(100)]))
    assert err_q < 0.002 and err_q < err_c / 3


def test_query_count_and_schedule():
    assert qae.schedule(8) == [0, 1, 2, 4, 8]
    assert qae.schedule(64, cap=4) == [0, 1, 2, 4]
    assert qae.queries([0, 1, 2], 10) == 10 * (1 + 3 + 5)


def test_numpy_simulator_matches_qibo_gate_level_circuit():
    import qibo

    qibo.set_backend("numpy")
    n, marked_idx = 4, [3, 10, 13]
    marked = np.zeros(2**n, dtype=bool)
    marked[marked_idx] = True
    for k in [0, 1, 2, 3]:
        probs = np.abs(np.asarray(qae.grover_qibo(n, marked_idx, k)().state())) ** 2
        assert np.allclose(probs, qae.grover_state(marked, k) ** 2, atol=1e-10)


def test_mps_roundtrip_and_ghz_bond_dimension():
    rng = np.random.default_rng(0)
    psi = rng.normal(size=2**8)
    psi /= np.linalg.norm(psi)
    fid, _, _ = qae.mps_fidelity(psi, max_bond=2**8)
    assert np.isclose(fid, 1)
    ghz = np.zeros(2**8)
    ghz[[0, -1]] = 1 / np.sqrt(2)
    fid, _, bonds = qae.mps_fidelity(ghz, max_bond=64)
    assert np.isclose(fid, 1) and max(bonds) == 2  # GHZ has bond dim 2
