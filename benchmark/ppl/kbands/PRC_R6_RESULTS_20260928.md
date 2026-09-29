# Round 6 results — 2026-09-28

Plan and pre-registration: `ROUNDS_6_8_PLAN_20260928.md` §3; manifests `prc_r6_attn_diag_20260928.json`,
`prc_r6_retarget_20260928.json`. Submitted 2026-09-28 with the user's OK: diagnostic array **62208439**
(tasks 0–2, throttle 2), re-target **62208440** (`--array=0`, 14B t32 only; 30B t64 dropped).

## Attention diagnostic (TRAIN windows, not citable, not iso-cost)

16 paired TRAIN windows per cell (30B: the round-4 confirmation windows; 4B: a fresh disjoint set).
Every INC re-evaluation reproduced the earlier NLLs exactly, and every arm passed the realized-length trace
audit. Stream lengths are HALVED code units. A = qk/av buckets pinned at the inherited top rung get a
per-bucket ladder whose top is 128, with thresholds unchanged. ATT128/LIN128 are attribution-only.
Source: Turbo `hpca/kbands/prc_r6_20260928/<cell>/diag_summary.json`.

| cell | arm | ΔPPL | SE | z | extra SC cycles | ΔPPL per 1% cycles |
|---|---|---:|---:|---:|---:|---:|
| 30B t32 | A | −1.981% | 0.375 | −5.2 | +7.00% | −0.283 |
| 30B t32 | ATT128 | −6.265% | 0.505 | −12.0 | +44.67% | −0.140 |
| 30B t40 | A | −2.356% | 0.324 | −7.2 | +8.66% | −0.272 |
| 30B t40 | ATT128 | −3.440% | 0.491 | −6.9 | +29.27% | −0.118 |
| 30B t40 | LIN128 | −5.061% | 0.354 | −13.9 | +172.97% | −0.029 |
| 4B t40 | A | −0.602% | 0.301 | −2.0 | +8.99% | −0.067 |
| 4B t40 | ATT128 | −0.676% | 0.313 | −2.2 | +11.32% | −0.060 |
| 4B t64 | A | −0.440% | 0.237 | −1.9 | +4.29% | −0.102 |
| 4B t64 | ATT128 | −0.543% | 0.297 | −1.8 | +4.92% | −0.111 |

Pre-registered Round-7 gate: ΔPPL(A) < −(0.6 × chord × extra%) − 1.5 SE, and gross ΔNLL(A) ≤ −0.01 (30B) /
≤ −0.004 (4B).

| cell | ΔPPL(A) | threshold | chord (%PPL per 1% cycles) | Round 7 |
|---|---:|---:|---:|---|
| 30B t32 | −1.981% | −1.526% | 0.2292 (t32→t40) | **proceed** |
| 30B t40 | −2.356% | −1.252% | 0.1475 (t40→t48) | **proceed** |
| 4B t40 | −0.602% | −1.222% | 0.1428 (t40→t48) | stop |
| 4B t64 | −0.440% | −0.419% | 0.0247 (t64→t96) | proceed (marginal: 0.02 pp inside, z −1.9) |

Reading:
- **30B:** the pinned attention buckets pay 0.27–0.28 %PPL per 1% cycles. That is 1.2× (t32) and 1.85× (t40) the
  budget chord, and about 9× what linears pay on t40 (LIN128 −0.029/%). The inherited ladder top starves 30B
  attention; an iso-cost joint re-solve that funds the 128 rung from low-value cycles has real room.
- **Q9 attribution (30B t40):** attention→128 recovers 0.035 nats, linears→128 0.052 nats (paired difference
  0.017 ± 0.0045). Both carry residual loss; attention is far cheaper per cycle.
- **4B:** the lever is weak (−0.07 to −0.11 %/%). t40 fails; t64 passes only because its chord is tiny.
- 30B t48 was not measured. Per the plan (§4) it follows 30B t40's gate.
- Framing: this is per-row attention reallocation, not a T1 granularity effect.

## 14B t32 budget re-target

Solved at scale ×1.0161 (held-out fixed point). Stage B: cost ratio 0.999337 vs parent (tol ±0.3%), held-out ΔNLL
−0.00262 vs incumbent (z −0.59), identity/audit/sanity checks all true → full test running. Flip threshold
≤1.05× fp16 = 9.070215 (incumbent 9.076412).

**Full test (job 62208440, COMPLETED 3h15m, protocol_ok, 298,862 tokens): PPL 9.084489** — +0.089% vs the incumbent,
1.0517× fp16 (**no flip**), −2.42% vs parent. Test cost +0.084% vs parent (+1.44% vs the incumbent).
Source: Turbo `hpca/kbands/prc_r6_retarget_20260928/14B_t32/full_test_result.json`.

**Verdict: budget re-targeting is refuted on its best case.** Spending the incumbent's 1.34% under-spend bought
nothing measurable (held-out predicted −0.26% at z −0.59; test +0.09%). Per plan §4, no further re-targets run.
Best(all) for 14B t32 stays the c7gfisla incumbent (9.0764); the r6 table is recorded as a comparable candidate.
