"""Mirror circuits + an MPS simulator in NumPy.

Peaked circuit = looks random, but one output bitstring has high probability
(Aaronson & Zhang, arXiv:2404.14493). I use the easiest version, a mirror circuit:
random brickwork U, then U^dagger, then some X gates to pick the peak string.

The MPS simulator just applies gates in order and truncates. It never uses the fact
that the second half undoes the first.

Convention: site 0 = leftmost bit.
"""
import numpy as np

X = np.array([[0, 1], [1, 0]], dtype=complex)


# circuits
def random_unitary(dim, rng):
    """Haar random unitary (QR of a complex Gaussian, with the usual phase fix)."""
    z = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def mirror_circuit(n, half_depth, seed):
    """gates = list of (matrix, sites), peak = bitstring the circuit outputs with prob 1.

    half_depth brickwork layers, then the inverse, then X on random qubits.
    """
    rng = np.random.default_rng(seed)
    forward = []
    for layer in range(half_depth):
        for i in range(layer % 2, n - 1, 2):
            forward.append((random_unitary(4, rng), (i, i + 1)))
    backward = [(u.conj().T, sites) for u, sites in reversed(forward)]
    peak = rng.integers(0, 2, size=n)
    flips = [(X, (i,)) for i in range(n) if peak[i]]
    return forward + backward + flips, "".join(map(str, peak))


# exact simulator
def run_exact(n, gates):
    """Plain statevector sim. Only for small n, used to check the MPS."""
    psi = np.zeros([2] * n, dtype=complex)
    psi[(0,) * n] = 1
    for u, sites in gates:
        k = len(sites)
        psi = np.tensordot(u.reshape([2] * 2 * k), psi, axes=(list(range(k, 2 * k)), list(sites)))
        psi = np.moveaxis(psi, list(range(k)), list(sites))
    return psi.reshape(-1)


# MPS simulator
class MPS:
    """One tensor per qubit, shape (left bond, 2, right bond).

    Bonds are capped at max_bond, so memory is n * max_bond^2 and not 2^n.
    Anything above log2(max_bond) bits of entanglement on a cut gets truncated.
    """

    def __init__(self, n, max_bond):
        self.n, self.max_bond = n, max_bond
        self.t = [np.array([1, 0], dtype=complex).reshape(1, 2, 1) for _ in range(n)]
        self.center = 0          # orthogonality centre (left of it: left-canonical, right: right-canonical)
        self.kept = 1.0          # product of kept SVD weights, rough fidelity estimate
        self.largest_bond = 1

    def _move_center(self, target):
        while self.center < target:    # QR, push R into the next site
            i = self.center
            l, _, r = self.t[i].shape
            q, rest = np.linalg.qr(self.t[i].reshape(l * 2, r))
            self.t[i] = q.reshape(l, 2, -1)
            self.t[i + 1] = np.tensordot(rest, self.t[i + 1], axes=1)
            self.center += 1
        while self.center > target:    # same thing going left
            i = self.center
            l, _, r = self.t[i].shape
            q, rest = np.linalg.qr(self.t[i].reshape(l, 2 * r).T)
            self.t[i] = q.T.reshape(-1, 2, r)
            self.t[i - 1] = np.tensordot(self.t[i - 1], rest.T, axes=1)
            self.center -= 1

    def apply(self, u, sites):
        if len(sites) == 1:
            (i,) = sites
            self.t[i] = np.einsum("ab,lbr->lar", u, self.t[i])
            return
        i = sites[0]
        self._move_center(i)
        theta = np.tensordot(self.t[i], self.t[i + 1], axes=1)             # (l, 2, 2, r)
        l, _, _, r = theta.shape
        theta = np.einsum("abcd,lcdr->labr", u.reshape(2, 2, 2, 2), theta)
        a, s, b = np.linalg.svd(theta.reshape(l * 2, 2 * r), full_matrices=False)
        chi = min(self.max_bond, int((s > 1e-12 * s[0]).sum()))
        self.kept *= float((s[:chi] ** 2).sum() / (s**2).sum())
        s = s[:chi] / np.linalg.norm(s[:chi])                              # keep norm 1 after cutting
        self.t[i] = a[:, :chi].reshape(l, 2, chi)
        self.t[i + 1] = (s[:, None] * b[:chi]).reshape(chi, 2, r)
        self.center = i + 1
        self.largest_bond = max(self.largest_bond, chi)

    def probability(self, bits):
        """|<bits|psi>|^2"""
        v = np.ones(1, dtype=complex)
        for tensor, b in zip(self.t, bits):
            v = v @ tensor[:, int(b), :]
        return float(abs(v[0]) ** 2)

    def sample(self, rng):
        """Sample one bitstring site by site (needs the centre at site 0)."""
        self._move_center(0)
        v, bits = np.ones(1, dtype=complex), []
        for tensor in self.t:
            branches = [v @ tensor[:, b, :] for b in (0, 1)]
            p0 = np.linalg.norm(branches[0]) ** 2
            p1 = np.linalg.norm(branches[1]) ** 2
            b = int(rng.random() < p1 / (p0 + p1))
            v = branches[b] / np.sqrt(p1 if b else p0)
            bits.append(b)
        return "".join(map(str, bits))


def run_mps(n, gates, max_bond):
    mps = MPS(n, max_bond)
    for u, sites in gates:
        mps.apply(u, sites)
    return mps
