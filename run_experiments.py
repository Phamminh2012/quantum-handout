"""Experiments for the amplitude estimation part. Run: python run_experiments.py  (~30 s)

1. error vs queries: Monte Carlo, full-depth AE, depth-limited AE
2. time per Grover iteration vs qubits, and multiprocessing speedup for many shallow circuits
3. MPS compression of the Grover state for two catalogue layouts
"""
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import qae

OUT = Path(__file__).resolve().parent / "results"
N_QUBITS, TRIALS, SEED, SHOTS = 10, 200, 0, 100
CAPS = [4, 32]  # max Grover iterations per circuit
BLUE, ORANGE, AQUA, YELLOW, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#0b0b0b", "#52514e"


def rmse(estimates, a):
    return float(np.sqrt(np.mean((np.asarray(estimates) - a) ** 2)))


def slope(x, y):
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def tidy(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#e6e5e0", lw=0.8)
    ax.set_axisbelow(True)


# 1. accuracy
def accuracy():
    marked = qae.make_catalogue(N_QUBITS, seed=SEED)
    a = marked.mean()
    rng = np.random.default_rng(SEED)
    probs = {k: qae.p_marked(marked, k) for k in qae.schedule(256)}
    series = {}

    # full depth: 100 shots per depth, more budget = deeper circuits
    pts = []
    for max_k in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        ks = qae.schedule(max_k)
        sub = {k: probs[k] for k in ks}
        pts.append((qae.queries(ks, SHOTS), rmse([qae.mlae(sub, SHOTS, rng) for _ in range(TRIALS)], a)))
    series["AE, full depth"] = pts

    # capped: depths stay <= cap, more budget = more shots
    for cap in CAPS:
        ks = qae.schedule(cap)
        sub = {k: probs[k] for k in ks}
        pts = []
        for shots in [50, 100, 200, 400, 800, 1600]:
            pts.append((qae.queries(ks, shots), rmse([qae.mlae(sub, shots, rng) for _ in range(TRIALS)], a)))
        series[f"AE, depth ≤ {cap}"] = pts

    budgets = np.unique(np.geomspace(100, 100_000, 12).astype(int))
    series["Monte Carlo"] = [(int(n), rmse([qae.monte_carlo(a, n, rng) for _ in range(TRIALS)], a)) for n in budgets]

    order = ["Monte Carlo"] + [f"AE, depth ≤ {c}" for c in CAPS] + ["AE, full depth"]
    colors = dict(zip(order, [ORANGE, YELLOW, AQUA, BLUE]))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    tidy(ax)
    rows = []
    for name in order:
        x, y = map(np.array, zip(*series[name]))
        ax.loglog(x, y, color=colors[name], lw=2, marker="o", ms=6, label=name)
        ax.annotate(name, (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points", va="center", color=INK, fontsize=9)
        s = slope(x, y)
        # queries for 1e-3 error, extrapolated from the fit
        c = np.exp(np.mean(np.log(y) - s * np.log(x)))
        depth = {"Monte Carlo": 0, "AE, full depth": 256}.get(name) if "≤" not in name else int(name.split()[-1])
        rows.append((name, s, (1e-3 / c) ** (1 / s), depth))
    ax.set_xlabel("oracle queries (total)", color=MUTED)
    ax.set_ylabel("RMSE of estimated fraction", color=MUTED)
    ax.set_title(f"Estimating the relevant fraction a = {a:.3f} of {2**N_QUBITS} items", loc="left", fontsize=11)
    ax.set_xlim(right=ax.get_xlim()[1] * 6)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_accuracy.png", dpi=200)
    plt.close(fig)

    lines = [f"Catalogue: {2**N_QUBITS} items, true relevant fraction a = {a:.4f}. RMSE over {TRIALS} trials, seed {SEED}.", "",
             "| Method | max Grover depth | fitted slope (error ∝ queries^slope) | queries for RMSE 1e-3 (from fit) |", "|---|---|---|---|"]
    for name, s, q, depth in rows:
        lines.append(f"| {name} | {depth} | {s:.2f} | {q:,.0f} |")
    with open(OUT / "accuracy.csv", "w") as f:
        f.write("method,queries,rmse\n")
        f.writelines(f"{name},{q},{e:.6g}\n" for name in order for q, e in series[name])
    for line in lines[2:]:
        print(line)
    return lines


# 2. HPC
def one_circuit(args):
    """One Grover circuit, run inside a worker process."""
    n_qubits, k, seed = args
    marked = np.random.default_rng(seed).random(2**n_qubits) < 0.1
    return qae.p_marked(marked, k)


def hpc(configs=((16, 4000), (22, 32)), n_circuits=32):
    """configs = (qubits, depth). 16 qubits is a 0.5 MB state, 22 qubits is 32 MB."""
    qubits = list(range(10, 25, 2))
    per_iter = []
    for n in qubits:
        marked = np.random.default_rng(0).random(2**n) < 0.1
        reps = 20
        t = time.perf_counter()
        qae.grover_state(marked, reps)
        per_iter.append((time.perf_counter() - t) / reps)

    workers = [w for w in (1, 2, 4, 8) if w <= (os.cpu_count() or 1)]
    runs = {}
    for n, k in configs:
        jobs = [(n, k, s) for s in range(n_circuits)]
        wall = []
        for w in workers:
            with ProcessPoolExecutor(w) as pool:
                list(pool.map(one_circuit, jobs[:w]))  # warm up
                t = time.perf_counter()
                list(pool.map(one_circuit, jobs))
                wall.append(time.perf_counter() - t)
        runs[(n, k)] = wall

    fig, (l, r) = plt.subplots(1, 2, figsize=(10, 3.8))
    tidy(l); tidy(r)
    l.semilogy(qubits, per_iter, color=BLUE, lw=2, marker="o", ms=6)
    l.set_xlabel("qubits", color=MUTED); l.set_ylabel("seconds per Grover iteration", color=MUTED)
    l.set_title("Simulation cost grows as 2^qubits", loc="left", fontsize=11)
    r.plot(workers, workers, color=MUTED, lw=1, ls="--", label="ideal")
    for ((n, k), wall), color in zip(runs.items(), [BLUE, ORANGE]):
        label = f"{n} qubits ({2**n * 8 / 2**20:.1f} MB state)"
        sp = [wall[0] / t for t in wall]
        r.plot(workers, sp, color=color, lw=2, marker="o", ms=6, label=label)
        r.annotate(label, (workers[-1], sp[-1]), xytext=(6, 0), textcoords="offset points", va="center", color=INK, fontsize=9)
    r.set_xticks(workers); r.set_xlim(right=workers[-1] * 1.45)
    r.set_xlabel("worker processes", color=MUTED); r.set_ylabel("speedup", color=MUTED)
    r.set_title(f"Parallel simulation of {n_circuits} independent circuits", loc="left", fontsize=11)
    r.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_hpc.png", dpi=200)
    plt.close(fig)

    lines = ["", f"Simulation cost (one Grover iteration, single process, {os.cpu_count()} logical CPUs on this machine):", "",
             "| qubits | amplitudes | seconds per iteration |", "|---|---|---|"]
    lines += [f"| {n} | {2**n:,} | {t:.2e} |" for n, t in zip(qubits, per_iter)]
    for (n, k), wall in runs.items():
        lines += ["", f"Parallel simulation of {n_circuits} independent circuits ({n} qubits, depth {k}, state {2**n * 8 / 2**20:.1f} MB):", "",
                  "| workers | wall time (s) | speedup | efficiency |", "|---|---|---|---|"]
        lines += [f"| {w} | {t:.2f} | {wall[0] / t:.2f}× | {wall[0] / t / w:.0%} |" for w, t in zip(workers, wall)]
    for line in lines:
        print(line)
    return lines


# 3. tensor networks
def tensor_network(n=14, k=1):
    """MPS bond dimension of the Grover state, same 10% relevant fraction, two layouts:
    relevant items at random indices vs. catalogue sorted by score (one block).
    """
    N = 2**n
    scattered = qae.make_catalogue(n, seed=SEED)
    block = np.arange(N) >= N - scattered.sum()
    bonds_list = [1, 2, 4, 8, 16, 32, 64, 128]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    tidy(ax)
    lines = ["", f"MPS compression of the Grover state ({n} qubits = {N:,} amplitudes, depth {k}, relevant fraction {scattered.mean():.3f}):", "",
             "| catalogue layout | bond dimension for exact MPS | MPS parameters (exact) | fidelity at bond 2 | fidelity at bond 16 |", "|---|---|---|---|---|"]
    for name, marked, color in [("unstructured (random indices)", scattered, ORANGE), ("sorted by score (one block)", block, BLUE)]:
        psi = qae.grover_state(marked, k)
        fids = [qae.mps_fidelity(psi, chi)[0] for chi in bonds_list]
        _, params, bonds = qae.mps_fidelity(psi, N)
        ax.plot(bonds_list, fids, color=color, lw=2, marker="o", ms=6, label=name)
        lines.append(f"| {name} | {max(bonds)} | {params:,} | {fids[1]:.4f} | {fids[4]:.4f} |")
    ax.set_xscale("log", base=2); ax.set_xticks(bonds_list); ax.set_xticklabels(bonds_list)
    ax.set_xlabel("maximum bond dimension", color=MUTED); ax.set_ylabel("fidelity with exact state", color=MUTED)
    ax.set_title("Structure decides whether a tensor network can simulate it", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_mps.png", dpi=200)
    plt.close(fig)
    for line in lines:
        print(line)
    return lines


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    lines = accuracy() + hpc() + tensor_network()
    (OUT / "results.md").write_text("\n".join(lines) + "\n")
