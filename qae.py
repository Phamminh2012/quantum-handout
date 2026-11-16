"""Amplitude estimation for a counting problem (NumPy).

N = 2^n items, a fraction `a` is relevant (similarity to the user above a threshold).
Goal: estimate a.

  Monte Carlo        error ~ 1/sqrt(queries)
  AE (full depth)    error ~ 1/queries
  AE (depth <= K)    error ~ 1/sqrt(queries * K), many shallow circuits

Estimator is MLAE from Suzuki et al. 2020 (no QFT, just Grover circuits of different depth).
"""
import numpy as np


# the problem
def make_catalogue(n_qubits, dim=32, target_fraction=0.1, seed=0):
    """Random unit embeddings + one user vector. Returns a bool mask of the relevant items."""
    rng = np.random.default_rng(seed)
    items = rng.normal(size=(2**n_qubits, dim))
    items /= np.linalg.norm(items, axis=1, keepdims=True)
    user = rng.normal(size=dim)
    sims = items @ (user / np.linalg.norm(user))
    return sims > np.quantile(sims, 1 - target_fraction)


# the circuit
def grover_state(marked, k):
    """State after k Grover iterations starting from the uniform state.

    oracle: flip sign of marked items. diffusion: reflect about the mean.
    """
    N = len(marked)
    psi = np.full(N, 1 / np.sqrt(N))
    sign = np.where(marked, -1.0, 1.0)
    for _ in range(k):
        psi = sign * psi
        psi = 2 * psi.mean() - psi
    return psi


def p_marked(marked, k):
    """P(measure a marked item) after k iterations."""
    return float((grover_state(marked, k)[marked] ** 2).sum())


def p_theory(a, k):
    """sin^2((2k+1) theta) with a = sin^2(theta). Used to check the simulator."""
    return np.sin((2 * k + 1) * np.arcsin(np.sqrt(a))) ** 2


# estimators
THETA_GRID = np.linspace(0, np.pi / 2, 50_001)[1:-1]


def queries(ks, shots):
    """Total oracle calls. A depth-k circuit counts as 2k+1, same as Suzuki et al."""
    return int(sum(shots * (2 * k + 1) for k in ks))


def mlae(probs, shots, rng):
    """MLAE. probs is {k: exact P(marked)} from the simulator.

    Draw `shots` binomial samples per depth, then grid search theta for the max likelihood.
    """
    loglik = np.zeros_like(THETA_GRID)
    for k, p in probs.items():
        hits = rng.binomial(shots, p)
        pk = np.clip(np.sin((2 * k + 1) * THETA_GRID) ** 2, 1e-12, 1 - 1e-12)
        loglik += hits * np.log(pk) + (shots - hits) * np.log(1 - pk)
    return float(np.sin(THETA_GRID[np.argmax(loglik)]) ** 2)


def monte_carlo(a, n_samples, rng):
    """Baseline: check n_samples random items."""
    return rng.binomial(n_samples, a) / n_samples


def schedule(max_k, cap=None):
    """Depths 0, 1, 2, 4, ... up to max_k (dropping anything above cap)."""
    ks = [0] + [2**j for j in range(int(np.log2(max_k)) + 1)] if max_k >= 1 else [0]
    return [k for k in ks if cap is None or k <= cap]


# the same circuit in Qibo
def grover_qibo(n_qubits, marked_indices, k):
    """Same Grover circuit built from gates in Qibo, to cross-check grover_state.

    oracle: one multi-controlled Z per marked item (X on the zero bits).
    diffusion: H X (MCZ) X H. Qibo is big-endian.
    """
    from qibo import Circuit, gates

    qs = list(range(n_qubits))
    mcz = lambda: gates.Z(qs[-1]).controlled_by(*qs[:-1])
    c = Circuit(n_qubits)
    c.add(gates.H(q) for q in qs)
    for _ in range(k):
        for idx in marked_indices:
            zeros = [q for q in qs if format(idx, f"0{n_qubits}b")[q] == "0"]
            c.add(gates.X(q) for q in zeros)
            c.add(mcz())
            c.add(gates.X(q) for q in zeros)
        c.add(gates.H(q) for q in qs)
        c.add(gates.X(q) for q in qs)
        c.add(mcz())
        c.add(gates.X(q) for q in qs)
        c.add(gates.H(q) for q in qs)
    return c


# tensor networks (MPS)
def to_mps(psi, max_bond):
    """Statevector -> MPS by SVD sweep, left to right. Returns (tensors, bond dims)."""
    n = int(np.log2(len(psi)))
    tensors, bonds, rest, left = [], [], psi.reshape(1, -1), 1
    for _ in range(n - 1):
        u, s, vh = np.linalg.svd(rest.reshape(left * 2, -1), full_matrices=False)
        chi = min(max_bond, int((s > 1e-12 * s[0]).sum()))
        tensors.append(u[:, :chi].reshape(left, 2, chi))
        rest, left = s[:chi, None] * vh[:chi], chi
        bonds.append(chi)
    tensors.append(rest.reshape(left, 2, 1))
    return tensors, bonds


def from_mps(tensors):
    """MPS -> statevector."""
    out = tensors[0]
    for t in tensors[1:]:
        out = np.tensordot(out, t, axes=1)
    return out.reshape(-1)


def mps_fidelity(psi, max_bond):
    """Fidelity of the truncated MPS with psi, number of parameters, bond dims."""
    tensors, bonds = to_mps(psi, max_bond)
    approx = from_mps(tensors)
    approx /= np.linalg.norm(approx)
    return float(abs(np.vdot(psi, approx)) ** 2), sum(t.size for t in tensors), bonds
