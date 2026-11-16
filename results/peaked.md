Mirror circuits, 16 qubits, mean over 5 random circuits per depth. The exact output puts probability 1 on the peak.

| half-depth | two-qubit gates | entanglement at the mirror (bits, max 8) | peak prob, bond 2 | peak prob, bond 4 | peak prob, bond 16 | peak prob, bond 64 |
|---|---|---|---|---|---|---|
| 2 | 30 | 0.76 | 0.883 | 1.000 | 1.000 | 1.000 |
| 4 | 60 | 1.68 | 0.518 | 0.923 | 1.000 | 1.000 |
| 8 | 120 | 3.20 | 0.118 | 0.444 | 0.974 | 1.000 |
| 12 | 180 | 4.47 | 0.040 | 0.195 | 0.789 | 0.996 |
| 16 | 240 | 5.64 | 0.008 | 0.086 | 0.529 | 0.954 |
| 20 | 300 | 6.49 | 0.006 | 0.032 | 0.331 | 0.841 |
| 24 | 360 | 6.95 | 0.000 | 0.007 | 0.210 | 0.703 |
| 28 | 420 | 7.13 | 0.000 | 0.003 | 0.134 | 0.578 |
| 32 | 480 | 7.22 | 0.000 | 0.002 | 0.083 | 0.464 |

Run time vs bond dimension (40 qubits, half-depth 16, 639 gates). Fitted exponent over the last three points: 2.01 (theory: 3).

| bond | seconds |
|---|---|
| 8 | 0.054 |
| 16 | 0.154 |
| 32 | 0.555 |
| 64 | 2.014 |
| 128 | 9.020 |

Beyond exact simulation: 48 qubits, half-depth 8, bond cap 64.

| quantity | value |
|---|---|
| state-vector memory | 4,504 TB |
| MPS memory | 4.9 MB (305,832 numbers, largest bond used 64) |
| MPS run time | 1.05 s |
| probability assigned to the true peak | 0.999872 |
| one sample equals the peak | True |
