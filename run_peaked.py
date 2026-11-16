"""Experiments for the mirror-circuit / MPS part. Run: python run_peaked.py  (~35 s)

1. peak probability from the truncated MPS vs depth and bond dimension (16 qubits)
2. run time vs bond dimension
3. a 48-qubit case that a statevector can't hold
"""
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import peaked

OUT = Path(__file__).resolve().parent / "results"
N, SEEDS = 16, range(5)
DEPTHS = [2, 4, 8, 12, 16, 20, 24, 28, 32]
BONDS = [2, 4, 16, 64]
COLORS = dict(zip(BONDS, ["#eb6834", "#eda100", "#1baf7a", "#2a78d6"]))
INK, MUTED = "#0b0b0b", "#52514e"


def tidy(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#e6e5e0", lw=0.8)
    ax.set_axisbelow(True)


def mid_entropy(gates):
    """Entropy in bits across the middle cut, after the forward half only (exact state)."""
    two = [g for g in gates if len(g[1]) == 2]
    psi = peaked.run_exact(N, two[: len(two) // 2])
    p = np.linalg.svd(psi.reshape(2 ** (N // 2), -1), compute_uv=False) ** 2
    p = p[p > 1e-15]
    return float(-(p * np.log2(p)).sum())


def recovery():
    """Mean P(peak) from the truncated MPS for each (bond, depth)."""
    p = {chi: [] for chi in BONDS}
    entropy = []
    for d in DEPTHS:
        circuits = [peaked.mirror_circuit(N, d, s) for s in SEEDS]
        for chi in BONDS:
            p[chi].append(np.mean([peaked.run_mps(N, g, chi).probability(peak) for g, peak in circuits]))
        entropy.append(np.mean([mid_entropy(g) for g, _ in circuits]))
        print(f"half-depth {d:2d}: entropy {entropy[-1]:.2f} bits | " + "  ".join(f"chi={c}: {p[c][-1]:.3f}" for c in BONDS))

    fig, ax = plt.subplots(figsize=(7, 4))
    tidy(ax)
    for chi in BONDS:
        ax.plot(DEPTHS, p[chi], color=COLORS[chi], lw=2, marker="o", ms=6, label=f"bond {chi}")
        ax.annotate(f"bond {chi}", (DEPTHS[-1], p[chi][-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", color=INK, fontsize=9)
    ax.set_xticks(DEPTHS); ax.set_xlim(right=DEPTHS[-1] + 5); ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("half-depth (layers before the mirror)", color=MUTED)
    ax.set_ylabel("probability assigned to the true peak", color=MUTED)
    ax.set_title(f"{N}-qubit mirror circuits: the simulator loses the peak as depth grows", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout(); fig.savefig(OUT / "fig_peak_recovery.png", dpi=200); plt.close(fig)

    lines = [f"Mirror circuits, {N} qubits, mean over {len(SEEDS)} random circuits per depth. The exact output puts probability 1 on the peak.", "",
             "| half-depth | two-qubit gates | entanglement at the mirror (bits, max 8) | " + " | ".join(f"peak prob, bond {c}" for c in BONDS) + " |",
             "|---" * (3 + len(BONDS)) + "|"]
    for j, d in enumerate(DEPTHS):
        n_gates = sum(1 for u, s in peaked.mirror_circuit(N, d, 0)[0] if len(s) == 2)
        lines.append(f"| {d} | {n_gates} | {entropy[j]:.2f} | " + " | ".join(f"{p[c][j]:.3f}" for c in BONDS) + " |")
    return lines


def cost():
    n, d = 40, 16                              # big enough that all bonds hit the cap
    gates, _ = peaked.mirror_circuit(n, d, 0)
    bonds = [8, 16, 32, 64, 128]
    secs = []
    for chi in bonds:
        t = time.perf_counter()
        peaked.run_mps(n, gates, chi)
        secs.append(time.perf_counter() - t)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    tidy(ax)
    ax.loglog(bonds, secs, color="#2a78d6", lw=2, marker="o", ms=6)
    ax.set_xscale("log", base=2); ax.set_xticks(bonds); ax.set_xticklabels(bonds)
    ax.set_xlabel("maximum bond dimension", color=MUTED); ax.set_ylabel("seconds", color=MUTED)
    ax.set_title(f"Cost of simulating one {n}-qubit, half-depth-{d} circuit", loc="left", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "fig_peak_cost.png", dpi=200); plt.close(fig)
    slope = np.polyfit(np.log(bonds[-3:]), np.log(secs[-3:]), 1)[0]
    lines = ["", f"Run time vs bond dimension ({n} qubits, half-depth {d}, {len(gates)} gates). Fitted exponent over the last three points: {slope:.2f} (theory: 3).", "",
             "| bond | seconds |", "|---|---|"] + [f"| {b} | {s:.3f} |" for b, s in zip(bonds, secs)]
    return lines


def beyond_exact(n=48, half_depth=8, chi=64):
    gates, peak = peaked.mirror_circuit(n, half_depth, seed=0)
    t = time.perf_counter()
    mps = peaked.run_mps(n, gates, chi)
    secs = time.perf_counter() - t
    found = mps.sample(np.random.default_rng(0)) == peak
    params = sum(x.size for x in mps.t)
    return ["", f"Beyond exact simulation: {n} qubits, half-depth {half_depth}, bond cap {chi}.", "",
            "| quantity | value |", "|---|---|",
            f"| state-vector memory | {2**n * 16 / 1e12:,.0f} TB |",
            f"| MPS memory | {params * 16 / 1e6:.1f} MB ({params:,} numbers, largest bond used {mps.largest_bond}) |",
            f"| MPS run time | {secs:.2f} s |",
            f"| probability assigned to the true peak | {mps.probability(peak):.6f} |",
            f"| one sample equals the peak | {found} |"]


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    lines = recovery() + cost() + beyond_exact()
    (OUT / "peaked.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
