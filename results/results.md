Catalogue: 1024 items, true relevant fraction a = 0.1006. RMSE over 200 trials, seed 0.

| Method | max Grover depth | fitted slope (error ∝ queries^slope) | queries for RMSE 1e-3 (from fit) |
|---|---|---|---|
| Monte Carlo | 0 | -0.50 | 84,820 |
| AE, depth ≤ 4 | 4 | -0.58 | 14,170 |
| AE, depth ≤ 32 | 32 | -0.51 | 2,479 |
| AE, full depth | 256 | -0.98 | 5,220 |

Simulation cost (one Grover iteration, single process, 10 logical CPUs on this machine):

| qubits | amplitudes | seconds per iteration |
|---|---|---|
| 10 | 1,024 | 3.08e-06 |
| 12 | 4,096 | 3.42e-06 |
| 14 | 16,384 | 1.05e-05 |
| 16 | 65,536 | 3.56e-05 |
| 18 | 262,144 | 3.34e-04 |
| 20 | 1,048,576 | 1.48e-03 |
| 22 | 4,194,304 | 6.75e-03 |
| 24 | 16,777,216 | 2.77e-02 |

Parallel simulation of 32 independent circuits (16 qubits, depth 4000, state 0.5 MB):

| workers | wall time (s) | speedup | efficiency |
|---|---|---|---|
| 1 | 3.91 | 1.00× | 100% |
| 2 | 1.95 | 2.01× | 100% |
| 4 | 1.01 | 3.85× | 96% |
| 8 | 0.82 | 4.75× | 59% |

Parallel simulation of 32 independent circuits (22 qubits, depth 32, state 32.0 MB):

| workers | wall time (s) | speedup | efficiency |
|---|---|---|---|
| 1 | 6.50 | 1.00× | 100% |
| 2 | 4.61 | 1.41× | 71% |
| 4 | 3.71 | 1.75× | 44% |
| 8 | 3.33 | 1.95× | 24% |

MPS compression of the Grover state (14 qubits = 16,384 amplitudes, depth 1, relevant fraction 0.100):

| catalogue layout | bond dimension for exact MPS | MPS parameters (exact) | fidelity at bond 2 | fidelity at bond 16 |
|---|---|---|---|---|
| unstructured (random indices) | 128 | 43,688 | 0.6411 | 0.7053 |
| sorted by score (one block) | 2 | 104 | 1.0000 | 1.0000 |
