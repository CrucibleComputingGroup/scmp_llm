# T1 per-group allocation — complete round ledger (as of 2026-09-27 21:50 EDT)

Read-only investigation. All numbers below were re-read from artifacts; the source of each is given
inline or in §13. Units: code/trace costs are HALVED stream lengths (nominal = 2x). "dPPL" = relative
PPL change; paired held-out/search/confirmation deltas are exp(mean dNLL)−1.

## 0. Bottom line

* Best(all) vs the submitted per-row parents: mean **−2.53%** over 20 cells
  (`hpca_results/llm/ppl/prc2/BEST_ALL_VS_SUBMITTED_20260927.md`). Re-deriving it round by round from
  the full-test traces (script `inv/decomp.py`), the mean splits into:
  **c17 per-group recalibration −1.51%**, **global Fisher λ (round 1) −0.60%**,
  **joint linear+attention λ (round 2) −0.44%**, and **everything after round 2 (Claude's calib8
  "round 3/3b/3c", Codex round 3 `prc_local`, Codex round 4 `prc_adjacent` + 30B companion) −0.002%**
  (a single −0.030% cell, 30B t40).
* After round 2, 5 more search rounds produced **0 improvements that survive a paired z < −1.5
  on fresh windows and then win on test**. The ledger says why: (a) the post-round-2 proposals carried
  **no information about which direction helps** (fixed families ranked by size/diversity, one
  iteration); (b) the step size was wrong in both directions — round 3 moved 2% of total SC cycles
  by shifting whole threshold vectors (rung skips, −17…−20% of the affected operator's own budget) and
  lost 19/20; round 4 moved **0.005–0.069%** of total SC cycles, **10–20x below what 6- or 16-window
  paired NLL can resolve**, so its search was noise (candidate spread ≈ paired SE; |dPPL| uncorrelated
  with cycles moved); (c) the only allocation change that paid after round 1 was **putting more
  budget into attention** (18/20 round-2 cells: attention-up ⇒ test better, attention-down ⇒ worse),
  but in the dense tight cells attention is **pinned at its top rung (96/97 halved)**, and per-group
  dispatch never reaches attention at all (20.8–42.3% of SC cycles).

## 1. What "rounds 3, 4 and 5" are

No document, JSON, Slurm job name, or Codex session log uses "round 5"
(grep over `SCMP/` and `~/.codex/sessions/2026/09/{26,27}`; zero hits). The Codex session
`~/.codex/sessions/2026/09/26/rollout-2026-09-26T15-29-47-…jsonl` shows the user asking Codex, in order:
finish round 2 → "another round 3" → "code the round 4 implementation and queue (use 2 gpus)" →
"make it 4 gpus". Slurm (sacct since 2026-09-20) has exactly these Codex jobs after round 2:

| user's round (most plausible) | Codex label | Slurm | cells | ran (EDT) |
|---|---|---|---|---|
| (round 2 finish) | round 2 finish | 61982983 `p7finish_20260926` 0–13%4 | 14 tasks | 09-26 18:40 → 09-27 04:44 |
| **3** | round 3 `prc_local` | 62032412 `p3local20260926` 0–2 | 4B t32, llama8B t32, 30B t40 | 09-27 04:45 → 11:11 |
| **4** | round 4 `prc_adjacent` | 62104342 `p4adjacent20260927` 0–1 | 4B t32, llama8B t32 | 09-27 14:27 → 16:10 |
| **5** (?) | round-4 **companion** | 62105189 `p4adj30b20260927` 0–1 | 30B t32, 30B t40 | 09-27 14:28 → 17:36 |

So "round 5" is most likely the 30B companion array, which Codex itself files under round 4 (same
driver/generator/hashes; `prc_adjacent_30b_20260927.json` `scope_extension`). **Naming collision:**
PRC2_OVERNIGHT.md also calls Claude's 09-24/25 calib8 work "round 3" (+"3b", "3c"); Codex's
`prc_local` is a different "round 3" (ROUND3_INVESTIGATION_20260926.md acknowledges "Claude ran an
earlier round 3 on September 24-25"). Below I use R3-Claude and R3-Codex.
No GPU jobs are running now (`squeue` empty).

## 2. The ledger

| round | dates | hypothesis | what changed (allocation only) | cells | outcome vs incumbent | stated reason |
|---|---|---|---|---|---|---|
| d17 (calib2 first try) | 09-23 02:11–03:35 | per-(row,chunk) staircase fixes v7's 8 calibration defects | calib2 tables, budgets = parent TEST trace | 20 calibrated | **0 evaluated**: gate failed (4B/8B/14B) + env `ModuleNotFoundError` | budgets from test trace vs train windows; editable-install path dead |
| **R1a c17** (calib2) | 09-23 → 09-25 | per-(row,chunk) groups + correct calibration | per-(op, layer-quartile) monotone staircase on full-call-normalized chunk absmax, 17-rung dense ladder [8..128], raw squared error, budget = parent's own L on 6 calib windows; attention = parent | 20/20 (+c17e32 30B t32, c17h20 14B t64/96) | **19/20 better, mean −1.51%** vs submitted | — (the T1 result) |
| controls | 09-23/24 | isolate granularity; iso-PPL | r17/r17m (row-shared, same calib), s80/s85/s90 (reduced budget) | 4B t32/t48, llama8B t32; s80 on 5 cells | granularity alone −4.33% (4B t32, r17m), −2.05% (llama8B t32 r17) | not comparable (different budget) |
| R1b g5 (calib5) | 09-23 14:30–16:53 | one global λ removes bucket boundary | global σ (relative-error) currency across all linears | 4B t32 (held-out only), llama8B t32 | 4B: held-out +1.3% vs c17 (z +2.2) → stopped; llama8B: test 8.3710 (−0.23% vs c17) | σ currency misprices across operators |
| **R1c Fisher** (calib6 `c6*`) | 09-23 16:40 → 09-25 | loss-aware currency makes the global λ pay | diagonal Fisher (MC-label `gfis`; true-label `gfis_emp`) × chunk error, one λ over all linear groups; attention = parent | test: 14 dense cells gfis, 2 gfis_emp, 14B h20 t64/96 (`c6h20gfis`); 30B via calib7 linears-only (`c7gfis`) | **mean −0.60% on top of c17** (best-of arms) | — |
| **R2 joint** (calib7 `c7gfisla`) | 09-24 13:45 → 09-27 04:44 | put attention rows in the same λ | same Fisher currency; attention thresholds re-solved (ladder/escape fixed) | 20/20 tested (incl. round-2 finish array 61982983) | **9/20 better than r1 on test**, 5 ≥0.5%, 3 ≥1% (all 30B t32–t48); mean −0.44% after best-of | Fisher under-prices attention cuts; wins where λ moves budget INTO attention |
| R3-Claude (calib8, "round 3/3b") | 09-24 late → 09-25 03:00 | more windows / robust / per-block / re-linearize | 12 windows; fisclip, kraw; per-block keys; 2nd pass at the r1 trajectory | 4B t32 (+ llama8B t32 dump) | held-out only; **no test**. 12-window raw Fisher +32.5% vs parent; robust ≈ tie (+0.11…+0.45% NLL vs c6gfis); re-linearized +3.1% (raw) | heavy-tailed single-sample Fisher (one window 80–99% of mass) |
| R3c-Claude attention ceiling | 09-25 03:30–12:27 | attention is pinned at a <128 top rung | raise attention top to 112/128, linears cut to iso-cost | 4B t32, llama8B t32 | held-out vs its own reproduction: 4B worse (+0.31%/+1.22%), llama8B ≈0 (0.00/+0.12%) | gain paid back by the linear cut |
| **R3-Codex prc_local** | 09-27 04:45–11:11 | small measured-loss transfers fix Fisher's residual misallocation | common offset to ALL thresholds of a population, 2% of total SC cycles, 8 family directions (QK/AV/linears/proj/MLP) | 4B t32, llama8B t32, 30B t40 | **19/20 search candidates worse**; tests +0.212%, +0.342%, **−0.030% (30B t40 → best(all))** | coarse shifts: coincident attention thresholds sent rows 96→19 (halved), −20% of QK's own budget |
| **R4-Codex prc_adjacent** | 09-27 14:27–16:10 | one-rung, bounded-locality exchanges | one boundary in each of two buckets, ≤1 rung, ≤5% groups/MACs per bucket (planned at 80%), ≤2% of op budget | 4B t32, llama8B t32 | 4B: confirmation +0.138% (z +0.48), no test; llama8B: confirm −0.398% (z −1.98) → **test +0.077%** | "smaller exchanges did not generalize" |
| **R4b/"R5" 30B companion** | 09-27 14:28–17:36 | same | same | 30B t32 (incumbent c7gfisla), 30B t40 (incumbent R3-Codex cand07) | confirm −0.329% (z −0.68), −0.079% (z −0.27); no tests | failed z < −1.5 |

Sources: PRC2_OVERNIGHT.md (Log; §2026-09-24 evening; §round 3c), `prc_round2_status_20260927.json`
(`summary`), `prc_local_20260926_results.json`, Turbo `prc_adjacent_20260927/<cell>/selected.json`,
`…/llama8B_t32/full_test_result.json`, prc2 `*_heldout_nll.json`, full-test trace headers in
`kbands_20260801/ppl/<m>_t<T>_prc_<tag>_trace.json` (all `ppl_max_tokens=0`, 298,862 / 288,627 tokens).

**Coverage gap:** every post-round-2 search touched only t32/t40 cells of 4B, llama8B and 30B.
No 14B cell and no t48–t96 cell received any refinement round after round 2.

## 3. Per-cell ledger of full-test PPL (every comparable per-group arm)

Sequential decomposition (each step = best comparable arm so far; `inv/decomp.py` over the trace
headers). Parent = submitted per-row config (14B t64/t96 = h=20% parents `parents_h20`).

| cell | parent | c17 | r1 Fisher (arm) | r2 `c7gfisla` | R3/R4 tests | best(all) | total | c17 | +Fisher | +joint | +R3/R4 |
|---|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---:|
| 4B t32 | 11.9886 | 11.3305 | 11.1891 c6gfis | 11.1953 | R3 11.2128 | 11.1891 | −6.67 | −5.49 | −1.25 | 0 | 0 |
| 4B t40 | 11.1440 | 10.8766 | 10.8751 c6gfis | **10.7729** | — | 10.7729 | −3.33 | −2.40 | −0.01 | −0.94 | 0 |
| 4B t48 | 10.6491 | 10.5030 | 10.4665 c6gfis | 10.5532 | — | 10.4665 | −1.71 | −1.37 | −0.35 | 0 | 0 |
| 4B t64 | 10.4749 | 10.4085 | 10.3645 c6gfis | **10.3056** | — | 10.3056 | −1.62 | −0.63 | −0.42 | −0.57 | 0 |
| 4B t96 | 10.2225 | 10.1993 | 10.1995 c6gfis | **10.1827** | — | 10.1827 | −0.39 | −0.23 | 0 | −0.16 | 0 |
| llama8B t32 | 8.6758 | 8.3899 | 8.2795 c6gfis (g5 8.3710) | 8.3108 | R3 8.3078, R4 8.2859 | 8.2795 | −4.57 | −3.30 | −1.32 | 0 | 0 |
| llama8B t40 | 8.2093 | 8.0857 | 8.0266 c6gfis_emp (gfis 8.0300) | 8.1023 | — | 8.0266 | −2.23 | −1.51 | −0.73 | 0 | 0 |
| llama8B t48 | 7.9200 | 7.8151 | 7.7878 c6gfis | 7.7888 | — | 7.7878 | −1.67 | −1.32 | −0.35 | 0 | 0 |
| llama8B t64 | 7.7109 | 7.6584 | 7.6435 c6gfis | 7.6487 | — | 7.6435 | −0.87 | −0.68 | −0.19 | 0 | 0 |
| llama8B t96 | 7.6145 | 7.5987 | 7.5855 c6gfis | **7.5723** | — | 7.5723 | −0.55 | −0.21 | −0.17 | −0.17 | 0 |
| 14B t32 | 9.3097 | 9.1672 | 9.1041 c6gfis | **9.0764** | — | 9.0764 | −2.51 | −1.53 | −0.69 | −0.30 | 0 |
| 14B t40 | 9.0598 | 8.9478 | 8.9029 c6gfis (emp 8.9163) | 8.9076 | — | 8.9029 | −1.73 | −1.24 | −0.50 | 0 | 0 |
| 14B t48 | 8.9072 | 8.8142 | 8.7894 c6gfis | 8.8073 | — | 8.7894 | −1.32 | −1.04 | −0.28 | 0 | 0 |
| 14B t64 h20 | 8.7758 | 8.7110 | 8.6993 c6h20gfis | 8.7658 | — | 8.6993 | −0.87 | −0.74 | −0.13 | 0 | 0 |
| 14B t96 h20 | 8.6418 | 8.6441 | 8.6455 c6h20gfis | **8.6383** | — | 8.6383 | −0.04 | +0.03 | 0 | −0.07 | 0 |
| 30B t32 | 9.4647 | 8.9599 (e32) | 8.7163 c7gfis | **8.5812** | — | 8.5812 | −9.33 | −5.33 | −2.72 | −1.55 | 0 |
| 30B t40 | 8.5294 | 8.4123 | 8.3210 c7gfis | 8.0880 | R3 **8.0856** | 8.0856 | −5.20 | −1.37 | −1.09 | −2.80 | −0.03 |
| 30B t48 | 8.1784 | 8.1159 | 8.0414 c7gfis | **7.8571** | — | 7.8571 | −3.93 | −0.76 | −0.92 | −2.29 | 0 |
| 30B t64 | 7.7146 | 7.6601 | 7.6281 c7gfis | 7.6483 | — | 7.6281 | −1.12 | −0.71 | −0.42 | 0 | 0 |
| 30B t96 | 7.5985 | 7.5751 | 7.5336 c7gfis | 7.5898 | — | 7.5336 | −0.85 | −0.31 | −0.55 | 0 | 0 |
| **mean** | | | | | | | **−2.53** | **−1.51** | **−0.60** | **−0.44** | **−0.002** |

Per budget (mean total / c17 / Fisher / joint): t32 −5.77/−3.91/−1.49/−0.46; t40 −3.12/−1.63/−0.58/−0.93;
t48 −2.16/−1.13/−0.47/−0.57; t64 −1.12/−0.69/−0.29/−0.14; t96 −0.46/−0.18/−0.18/−0.10.

Distance to fp16 at best(all) (fp16 10.0445/7.2130/8.6383/7.2613): t32 4B 1.114, llama8B **1.148**,
14B 1.051, 30B **1.182**; t40 1.073/1.113/1.031/1.114. Fraction of the parent's fp16 excess removed:
10–56% (llama8B t64/t96 only 13.5%/10.5%).

Not comparable, therefore excluded from best(all) (correctly): h10 14B t64 c6gfis 8.6763 and h10 14B t96
c17 8.6102 — both lower than the h20 arms because the **h20 parents themselves are worse** (8.7758 vs
h10 8.7341; 8.6418 vs 8.6150); v7; s80/s85/s90; r17/r17m; `parent_swap_*`/`grid`/`mcmask` variants.

## 4. Lineage of each best(all) arm

| arm | cells where it is best(all) | what it is (code + recipe) |
|---|---|---|
| `c6gfis` | 4B t32/t48, llama8B t32/t48/t64, 14B t40/t48 | `mp_per_row_chunk_calib6.py` via `run_prc6.sbatch` (`P6_CUR` default `rel,fis`, then `fis,fis_emp`; `--grad-mode full`): on the loaded submitted per-row parent (NOT on c17), 6 calib + 2 held-out stratified TRAIN windows, 64 rows/call; per (row,chunk) error curves with AWQ `smooth_scales` on the SC trajectory; squared output gradients from a straight-through backward with **one MC target token per position** sampled from the SC model (MC-label diagonal Fisher); currency e_g(L)=Σ_j G[r,j]² δ_{g,j}(L)² − e_g(128); **one global λ** over all linear groups of all ops/layers, bisected to the parent's own linear cost on the calibration pairs; each (op, layer-quartile) keeps one ascending threshold vector on full-call min–max-normalized chunk absmax over the 17-rung ladder [8…128]; no linear escape gate; attention, escape, protected channels, INT mask = parent. All dense `c6_gfis` tables are timestamped 09-23 17:45 → 09-24 03:58, i.e. **before** the 09-24 13:20 bit-exact custom-autograd STE fix (PRC2_OVERNIGHT 13:20 entry; TWO_PHASE_DESCRIPTION: "Historical early c6 runs preceded the bit-exact custom STE correction"). |
| `c6gfis_emp` | llama8B t40 | same calib6 run, true-label (empirical) Fisher currency. 8.026594 vs MC 8.029962 — a 0.04% difference, inside test noise. |
| `c6h20gfis` | 14B t64 (h20) | calib6 recipe on `parents_h20/14B/target64` (`KB_PARENT`), job p6_14B_t64_c6h20 (61883659), post-STE-fix. |
| `c7gfis` | 30B t64, t96 | `mp_per_row_chunk_calib7.py` linear-only output (same currency as c6gfis) for 30B: block-sequential backward (`--grad-mode block`), 16 expert calls per block-window (PRC2_OVERNIGHT 09-24 21:10/13:20/19:50 entries), post-STE-fix. |
| `c7gfisla` | 4B t40/t64/t96, llama8B t96, 14B t32, 30B t32/t48 | calib7 **joint** λ over linear groups AND attention rows: attention keeps its per-row statistic, direction, ladder (the parent's own attention ladder — top 96/97 halved in the t32 cells and 30B; some higher-budget cells, e.g. llama8B t48/t64, already reach 128 — plus 128 via the μ+2τ escape) and escape rows (fixed cost); only attention thresholds are re-solved. Total cost held at the parent's. |
| `c7h20gfisla` | 14B t96 (h20) | calib7 joint on `parents_h20/14B/target96` (round-2 finish task). |
| `round3_local` = `prc_local_20260926/30B_t40/candidate07_mlp_from_projections` | 30B t40 | the 30B t40 `c7gfisla` table with a common threshold offset on the MLP (gate/up/down) and projection (q/k/v/o) populations moving 2.000% of total SC cycles from projections into MLP; search −0.052%, confirmation −0.045% (z −0.15), test 8.085611 vs 8.088038 (−0.030%). A no-signal difference kept only by the best(all) rule. |

Note on "round 1" identity: the round-1 incumbent is `c6_gfis`, but calib7 also emits a `c7_gfis`
table for the same cell. They are different tables (different sha256) and on identical 16 held-out
windows `c7_gfis − c6_gfis` = **+0.72% (4B t32, z +2.09), −0.74% (4B t40), +0.15%, −0.44%, −0.10%**
(4B t48/t64/t96), ±0.14–0.27% (llama8B), −0.28…0% (14B). I.e. **re-running the same Fisher recipe
moves held-out PPL by up to ±0.7%**, the same size as every gain sought after round 2.

## 5. Round 2 (joint λ) — the only post-round-1 lever that worked, and why it is selective

Joint-λ attention/linear moves are printed by calib7 (`[c7] JOINT lambda: linear L a -> b; attention L c -> d`,
Slurm logs `_kbands/p7_*_c7*.out`, `p7finish_20260926_61982983_*.out`); test deltas are
`prc_round2_status_20260927.json` rows (`dppl_pct` = r2 vs r1); held-out = paired 16-window TRAIN NLL
from prc2 `*_c7*_heldout_nll.json` (against the actual `c6_gfis` incumbent where that file has the same
windows, else against calib7's own `gfis`).

| cell | attention L (halved) | linear L | held-out r2−r1 | **test r2−r1** |
|---|---|---|---:|---:|
| 4B t32 | 83.49→72.90 (−12.7%) | 24.12→25.76 | −0.11% (z −0.22) | +0.055% |
| 4B t40 | 82.67→92.27 (+11.6%) | 34.15→32.65 | −1.34% (z −2.90) | **−0.940%** |
| 4B t48 | 98.06→95.54 (−2.6%) | 40.78→40.99 | +0.92% (z +2.31) | +0.829% |
| 4B t64 | 90.21→103.57 (+14.8%) | 59.35→57.27 | −0.95% (z −2.68) | **−0.568%** |
| 4B t96 | 122.33→127.30 (+4.1%) | 90.79→90.01 | −0.48% (z −3.01) | −0.164% |
| llama8B t32 | 96.64→94.33 (−2.4%) | 27.39→27.57 | +0.42% (z +1.60) | +0.378% |
| llama8B t40 | 98.61→94.77 (−3.9%) | 35.07→35.38 | +1.09% (z +3.04) | +0.901% |
| llama8B t48 | 128.00→127.17 (−0.6%) | 41.57→41.63 | −0.07% | +0.012% |
| llama8B t64 | 128.00→125.63 (−1.9%) | 58.09→58.28 | +0.08% | +0.068% |
| llama8B t96 | 121.48→122.69 (+1.0%) | 91.98→91.87 | −0.51% (z −3.75) | −0.174% |
| 14B t32 | 96.36→89.96 (−6.6%) | 27.97→28.35 | +0.33% | −0.304% |
| 14B t40 | 84.35→82.66 (−2.0%) | 36.96→37.06 | −0.02% | +0.052% |
| 14B t48 | 111.75→115.76 (+3.6%) | 43.31→43.07 | +0.27% | +0.203% |
| 14B t64 h20 | 87.54→84.04 (−4.0%) | 60.90→60.78 | +1.22% (z +3.13) | +0.764% |
| 14B t96 h20 | 123.83→127.31 (+2.8%) | 93.55→93.30 | −0.10% | −0.083% |
| 30B t32 | 61.75→62.85 (+1.8%) | 23.93→23.50 | −1.67% (z −3.26)* | **−1.550%** |
| 30B t40 | 64.68→74.87 (+15.8%) | 34.83→31.65 | −2.92% (z −7.51)* | **−2.800%** |
| 30B t48 | 70.19→86.44 (+23.2%) | 43.06→37.64 | −2.19% (z −6.54)* | **−2.293%** |
| 30B t64 | 117.13→110.16 (−6.0%) | 50.99→53.11 | +0.20%* | +0.266% |
| 30B t96 | 122.14→111.12 (−9.0%) | 86.15→89.54 | +1.04% (z +4.15)* | +0.746% |

\* vs calib7's own `gfis` (no 30B `c6` gfis exists: the c6 30B jobs lost all expert gradients).

* **Direction rule:** attention budget up ⇒ test better, attention down ⇒ test worse (or tie) in
  **18/20 cells**; Pearson r(attention change, test dPPL) = **−0.77**. Exceptions 14B t32/t48.
  The diagonal-Fisher currency therefore misprices attention in the "cut" direction; where the joint
  λ raised attention (30B, 4B t40/t64) the gains are the largest post-round-1 gains in the ledger.
* Net attention L understates what round 2 did inside attention: at 30B t32 the net change is only
  +1.8%, but the full traces show mean QK length 65.50→88.13 and AV 59.56→44.55 (halved;
  ROUND3_INVESTIGATION §Evidence 2). The −1.55% there came from a qk-up/av-down re-split.
* **Held-out tracks test:** across the 20 cells r(held-out, test) = **0.973**, slope 0.87, residual sd
  0.23%, 16/20 sign agreement. The 16-window paired NLL protocol is a reliable instrument for effects
  ≥ ~0.3–0.5%.
* Early narrative corrected: PRC2_OVERNIGHT's 4B t32 "gfisla vs gfis −0.82%, z −1.8" compares with
  the regenerated `c7_gfis`; against the real incumbent `c6_gfis` it is −0.11% (z −0.22) — verified
  here from both held-out files (same windows, bit-identical parent losses), as ROUND3_INVESTIGATION said.

## 6. R3-Claude (calib8 family, 09-24/25) — held-out only, no test evals

From prc2 held-out files (24 windows excluding both calibration sets):

| file | arm | paired dNLL vs c6gfis | z |
|---|---|---:|---:|
| `4B_t32_c8_heldout_nll.json` | c8gfis (12-window raw MC Fisher) | +0.3586 (PPL 16.999 vs parent 12.832: **+32.5% vs parent**) | +23.4 |
| same | c8gfisblk (per-block keys) | +0.1617 (+8.8% vs parent) | +20.8 |
| `4B_t32_c8o_select_heldout_nll.json` | fisclip10:q / fisclip100:q / kraw:q | +0.0044 / +0.0039 / +0.0045 | +1.2 / +1.0 / +1.1 |
| same | fisclip10:blk / kraw:blk | +0.0015 / +0.0011 | +0.42 / +0.25 |
| `4B_t32_c8xo_select_heldout_nll.json` (re-linearized at the r1 trajectory) | fis:q / fis:blk / fisclip100:q | +0.0306 / +0.0254 / +0.0044 | +7.5 / +8.4 / +1.1 |
| `4B_t32_c8w6o_select_heldout_nll.json` | r1 reproduction (same 6 windows) | +0.0056 | +2.14 |
| same | att112 (+linears x0.9399) / att128 (x0.8759) | +0.0087 / +0.0178 | +1.91 / +6.04 |
| `llama8B_t32_c8w6o_select_heldout_nll.json` | r1 reproduction / att112 / att128 | −0.0027 / −0.0027 / −0.0015 | −1.31 / −0.95 / −0.49 |

(dNLL x100 ≈ %PPL for these small values.)

Reasons recorded: the single-sample MC Fisher is heavy-tailed (one calibration window carries 80–99%
of each operator's Fisher mass; ~20 rows around a wikitext section break carry 83%); robust pricing
only restores round-1 quality; per-block keys ≈ −0.003 NLL vs quartile keys; raising the attention
ceiling is paid back by the iso-cost linear cut (dense t32 cells only; never tried on 30B).

## 7. R3-Codex `prc_local` (array 62032412) — per-candidate search results

Knob: `prc_local_proposals.py` adds one common offset to all thresholds of a recipient population and
the opposite to a donor population, targeting 2% of total SC cycles; 6 TRAIN search windows,
16 confirmation windows; best new candidate full-tested even if confirmation lost. Source: Turbo
`prc_local_20260926/<cell>/search_results.json` (`proposal`, `paired_vs_incumbent`), `selected.json`,
`full_test_result.json`.

| cell | recipient ← donor | moved (% total SC) | search dPPL | z | cost ratio |
|---|---|---:|---:|---:|---:|
| 4B t32 | qk ← linears | 2.000 | +1.395% | +2.12 | 1.0006 |
| | linears ← qk | 1.999 | +2.254% | +2.10 | 0.9994 |
| | av ← linears | 2.000 | **+0.545%** (sel) | +0.79 | 1.0011 |
| | linears ← av | 1.999 | +1.017% | +1.28 | 1.0007 |
| | qk ← av | 2.000 | +1.485% | +2.94 | 1.0035 |
| | av ← qk | 2.000 | +0.795% | +1.83 | 0.9996 |
| | projections ← mlp | 2.000 | +1.851% | +6.21 | 1.0006 |
| | mlp ← projections | 1.984 | +1.042% | +1.07 | 0.9970 |
| llama8B t32 | linears ← qk | 2.000 | +17.605% | +12.88 | 0.9965 |
| | linears ← av | 2.000 | +4.100% | +5.01 | 1.0054 |
| | projections ← mlp | 2.000 | **+0.665%** (sel) | +1.15 | 1.0002 |
| | mlp ← projections | 1.999 | +0.690% | +0.93 | 0.9987 |
| 30B t40 | qk ← linears | 0.914 (capacity) | +0.930% | +0.67 | 1.0000 |
| | linears ← qk | 2.000 | +0.396% | +0.41 | 0.9997 |
| | av ← linears | 1.992 | +0.821% | +1.05 | 1.0002 |
| | linears ← av | 2.000 | +0.845% | +0.64 | 1.0002 |
| | qk ← av | 0.914 | +1.662% | +1.50 | 1.0002 |
| | av ← qk | 1.992 | +0.829% | +0.81 | 1.0004 |
| | projections ← mlp | 2.000 | +0.612% | +0.75 | 1.0003 |
| | mlp ← projections | 2.000 | **−0.052%** (sel) | −0.07 | 0.9993 |

Selected → confirmation → test: 4B +0.545 → +0.431% (z +1.21) → **+0.212%** (11.212848 vs 11.189097);
llama8B +0.665 → +0.354% (z +1.02) → **+0.342%** (8.307774 vs 8.279485); 30B t40 −0.052 → −0.045%
(z −0.15) → **−0.030%** (8.085611 vs 8.088038; adopted into best(all)). All test costs within 0.06%.

Reading: at a 2%-of-total step, **both directions** of the proj↔MLP axis lose on 4B (+1.85 / +1.04%) and
llama8B (+0.67 / +0.69%), and every attention↔linear direction loses on 4B and 30B t40. The incumbents
are therefore at (or near) a local optimum of the operator-family budget split — the global Fisher λ
and joint λ already balanced it — so no family-level transfer of that size can help. The large QK
losses on llama8B are additionally inflated by the rung-skip defect (ROUND3_POSTMORTEM: five coincident
zero thresholds moved together, ~25% of QK MACs jumped 96→19 halved; −20% of QK's own budget).

## 8. R4-Codex `prc_adjacent` (62104342) + 30B companion (62105189) — every candidate

Knob (`prc_adjacent_proposals.py::propose`): move ONE threshold in each of two buckets (a recipient
gains cycles, a donor loses them), ≤1 rung per group, ≤5% of a bucket's groups/MACs and ≤5% of its
cost, ≤2% of an operator's budget, planned at 80% of these caps; requested transfer ceiling 0.25%
(0.5% also requested, see §11). Proposal choice: one candidate per family in a fixed priority order,
ranked by (bucket-reuse count, −cycles moved, cost-match error) — **no loss, Fisher or error signal
enters the ranking** (source lines 381–408). 12 candidates/cell out of 54,292–90,622 matched pairs.
Identity checks passed in all four cells (`identity.json`: `exact_match: true`, replayed = trace cost).
Columns: move = `bucket bBoundary rungs lo/hi (±cyc), % of bucket's groups changed, % change of bucket
cost`; "cycles moved" = `transferred_fraction`; search = 6 TRAIN windows paired vs incumbent.
Source: Turbo `prc_adjacent_20260927/<cell>/search_results.json`, `selected.json`.

**4B_t32** — incumbent search NLL 2.65358, trace cost 33.912; selected for confirmation: `candidate01_adj_linears_from_qk_01`

| # | family | move 1 (bucket, boundary, rungs, groups chg, bucket-cost chg) | move 2 | cycles moved (% total SC) | search dPPL | z | cost ratio |
|---:|---|---|---|---:|---:|---:|---:|
| 0 | qk_from_linears | qk:t0:l1 b0 rungs 97/64 (+cyc), 3.52% grp, 1.37% | down_proj:t0:l3 b13 rungs 80/96 (−cyc), 2.23% grp, 1.07% | 0.0667% | -0.231% | -0.43 | 1.0012 |
| 1 | linears_from_qk **←sel** | qk:t0:l2 b0 rungs 97/64 (−cyc), 2.91% grp, 1.22% | up_proj:t0:l3 b10 rungs 48/56 (+cyc), 3.88% grp, 1.05% | 0.0614% | -0.446% | -1.23 | 1.0003 |
| 2 | av_from_linears | av:t0:l3 b0 rungs 97/64 (+cyc), 3.59% grp, 1.38% | gate_proj:t0:l3 b8 rungs 32/40 (−cyc), 3.90% grp, 1.14% | 0.0614% | -0.388% | -2.72 | 1.0001 |
| 3 | linears_from_av | av:t0:l1 b0 rungs 97/64 (−cyc), 3.47% grp, 1.37% | down_proj:t0:l2 b10 rungs 48/56 (+cyc), 3.96% grp, 1.04% | 0.0593% | +0.686% | +1.50 | 1.0015 |
| 4 | qk_from_av | av:t0:l2 b0 rungs 97/64 (−cyc), 2.72% grp, 1.06% | qk:t0:l3 b0 rungs 97/64 (+cyc), 3.98% grp, 1.61% | 0.0578% | -0.193% | -0.54 | 0.9998 |
| 5 | av_from_qk | av:t0:l0 b0 rungs 97/64 (+cyc), 3.98% grp, 1.55% | qk:t0:l0 b0 rungs 97/64 (−cyc), 3.61% grp, 1.35% | 0.0261% | -0.321% | -0.69 | 1.0002 |
| 6 | projections_from_mlp | gate_proj:t0:l2 b5 rungs 20/24 (−cyc), 3.35% grp, 0.78% | o_proj:t0:l3 b10 rungs 48/56 (+cyc), 3.98% grp, 1.45% | 0.0264% | -0.440% | -1.05 | 0.9999 |
| 7 | mlp_from_projections | up_proj:t0:l1 b8 rungs 32/40 (+cyc), 2.15% grp, 0.79% | o_proj:t0:l2 b9 rungs 40/48 (−cyc), 3.97% grp, 1.05% | 0.0263% | -0.116% | -0.37 | 0.9990 |
| 8 | within_projections | o_proj:t0:l1 b10 rungs 48/56 (+cyc), 3.98% grp, 1.17% | q_proj:t0:l3 b9 rungs 40/48 (−cyc), 3.08% grp, 1.37% | 0.0205% | -0.312% | -0.78 | 1.0014 |
| 9 | within_mlp | down_proj:t0:l0 b10 rungs 48/56 (−cyc), 2.64% grp, 1.06% | up_proj:t0:l2 b6 rungs 24/28 (+cyc), 3.89% grp, 0.71% | 0.0307% | -0.097% | -0.26 | 1.0001 |
| 10 | within_av | av:t0:l1 b0 rungs 97/64 (−cyc), 4.00% grp, 1.58% | av:t0:l2 b0 rungs 97/64 (+cyc), 3.21% grp, 1.25% | 0.0685% | -0.024% | -0.05 | 1.0014 |
| 11 | within_down_proj | down_proj:t0:l1 b10 rungs 48/56 (+cyc), 3.98% grp, 1.13% | down_proj:t0:l3 b7 rungs 28/32 (−cyc), 3.55% grp, 0.43% | 0.0265% | -0.027% | -0.05 | 1.0001 |

**llama8B_t32** — incumbent search NLL 2.26435, trace cost 33.811; selected for confirmation: `candidate09_adj_within_o_proj_09`

| # | family | move 1 (bucket, boundary, rungs, groups chg, bucket-cost chg) | move 2 | cycles moved (% total SC) | search dPPL | z | cost ratio |
|---:|---|---|---|---:|---:|---:|---:|
| 0 | linears_from_qk | up_proj:t0:l2 b8 rungs 32/40 (+cyc), 1.70% grp, 0.55% | qk:t0:l3 b0 rungs 96/74 (−cyc), 3.99% grp, 0.92% | 0.0290% | -0.112% | -0.67 | 1.0018 |
| 1 | linears_from_av | av:t0:l3 b0 rungs 96/74 (−cyc), 4.00% grp, 0.92% | gate_proj:t0:l3 b10 rungs 48/56 (+cyc), 1.80% grp, 0.59% | 0.0306% | +0.050% | +0.33 | 0.9999 |
| 2 | projections_from_mlp | o_proj:t0:l2 b10 rungs 48/56 (+cyc), 3.98% grp, 1.33% | down_proj:t0:l2 b11 rungs 56/64 (−cyc), 1.91% grp, 0.46% | 0.0221% | +0.221% | +1.61 | 0.9995 |
| 3 | mlp_from_projections | down_proj:t0:l0 b11 rungs 56/64 (+cyc), 1.59% grp, 0.41% | o_proj:t0:l3 b9 rungs 40/48 (−cyc), 3.98% grp, 1.28% | 0.0220% | -0.147% | -0.44 | 1.0017 |
| 4 | within_projections | q_proj:t0:l0 b8 rungs 32/40 (+cyc), 4.00% grp, 1.17% | o_proj:t0:l0 b10 rungs 48/56 (−cyc), 3.97% grp, 1.00% | 0.0193% | -0.598% | -0.95 | 1.0030 |
| 5 | within_mlp | up_proj:t0:l3 b8 rungs 32/40 (−cyc), 3.30% grp, 0.91% | down_proj:t0:l3 b11 rungs 56/64 (+cyc), 3.98% grp, 0.98% | 0.0641% | +0.086% | +0.40 | 0.9997 |
| 6 | within_down_proj | down_proj:t0:l1 b10 rungs 48/56 (+cyc), 3.96% grp, 1.05% | down_proj:t0:l3 b10 rungs 48/56 (−cyc), 3.40% grp, 0.84% | 0.0547% | -0.294% | -0.60 | 1.0016 |
| 7 | within_gate_proj | gate_proj:t0:l0 b5 rungs 20/24 (−cyc), 3.90% grp, 0.81% | gate_proj:t0:l2 b8 rungs 32/40 (+cyc), 1.95% grp, 0.71% | 0.0236% | -0.372% | -0.94 | 1.0007 |
| 8 | within_k_proj | k_proj:t0:l2 b9 rungs 40/48 (−cyc), 3.99% grp, 0.98% | k_proj:t0:l3 b10 rungs 48/56 (+cyc), 3.46% grp, 0.77% | 0.0048% | -0.347% | -1.52 | 1.0004 |
| 9 | within_o_proj **←sel** | o_proj:t0:l0 b11 rungs 56/64 (+cyc), 3.97% grp, 1.00% | o_proj:t0:l1 b9 rungs 40/48 (−cyc), 3.98% grp, 0.87% | 0.0193% | -0.749% | -2.19 | 1.0023 |
| 10 | within_q_proj | q_proj:t0:l1 b8 rungs 32/40 (+cyc), 3.08% grp, 0.98% | q_proj:t0:l3 b6 rungs 24/28 (−cyc), 3.74% grp, 0.63% | 0.0102% | +0.021% | +0.07 | 1.0016 |
| 11 | within_up_proj | up_proj:t0:l0 b7 rungs 28/32 (−cyc), 3.88% grp, 0.65% | up_proj:t0:l1 b6 rungs 24/28 (+cyc), 3.88% grp, 0.70% | 0.0236% | -0.421% | -0.69 | 1.0027 |

**30B_t32** — incumbent search NLL 2.22159, trace cost 33.972; selected for confirmation: `candidate11_adj_within_down_proj_11`

| # | family | move 1 (bucket, boundary, rungs, groups chg, bucket-cost chg) | move 2 | cycles moved (% total SC) | search dPPL | z | cost ratio |
|---:|---|---|---|---:|---:|---:|---:|
| 0 | qk_from_linears | qk:t0:l2 b0 rungs 96/64 (+cyc), 1.78% grp, 0.66% | down_proj:t0:l3 b11 rungs 56/64 (−cyc), 3.93% grp, 0.94% | 0.0475% | -0.628% | -1.31 | 0.9998 |
| 1 | linears_from_qk | qk:t0:l3 b2 rungs 48/32 (−cyc), 3.20% grp, 0.60% | up_proj:t0:l3 b8 rungs 32/40 (+cyc), 3.96% grp, 1.18% | 0.0490% | +0.428% | +1.50 | 0.9997 |
| 2 | av_from_linears | av:t0:l2 b2 rungs 48/32 (+cyc), 1.98% grp, 0.70% | o_proj:t0:l3 b9 rungs 40/48 (−cyc), 3.98% grp, 1.03% | 0.0334% | -0.200% | -0.33 | 0.9989 |
| 3 | linears_from_av | av:t0:l3 b2 rungs 48/32 (−cyc), 2.91% grp, 0.76% | gate_proj:t0:l3 b8 rungs 32/40 (+cyc), 3.95% grp, 1.25% | 0.0490% | +0.052% | +0.20 | 1.0001 |
| 4 | qk_from_av | av:t0:l1 b2 rungs 48/32 (−cyc), 4.00% grp, 1.61% | qk:t0:l1 b0 rungs 96/64 (+cyc), 3.03% grp, 1.11% | 0.0674% | -0.713% | -1.59 | 0.9997 |
| 5 | av_from_qk | av:t0:l0 b4 rungs 24/16 (+cyc), 4.00% grp, 1.12% | qk:t0:l0 b0 rungs 96/64 (−cyc), 2.33% grp, 0.81% | 0.0253% | -0.511% | -0.47 | 0.9995 |
| 6 | projections_from_mlp | down_proj:t0:l2 b9 rungs 40/48 (−cyc), 2.33% grp, 0.78% | q_proj:t0:l3 b10 rungs 48/56 (+cyc), 3.15% grp, 0.83% | 0.0259% | -0.869% | -1.82 | 0.9998 |
| 7 | mlp_from_projections | down_proj:t0:l1 b8 rungs 32/40 (+cyc), 2.49% grp, 0.95% | o_proj:t0:l2 b9 rungs 40/48 (−cyc), 3.99% grp, 1.08% | 0.0250% | -0.601% | -1.48 | 0.9990 |
| 8 | within_projections | o_proj:t0:l0 b9 rungs 40/48 (−cyc), 3.11% grp, 1.09% | q_proj:t0:l2 b8 rungs 32/40 (+cyc), 2.28% grp, 0.83% | 0.0173% | -0.172% | -0.46 | 0.9991 |
| 9 | within_mlp | gate_proj:t0:l1 b5 rungs 20/24 (−cyc), 3.94% grp, 1.04% | up_proj:t0:l2 b6 rungs 24/28 (+cyc), 3.95% grp, 0.89% | 0.0225% | -0.589% | -0.84 | 1.0001 |
| 10 | within_av | av:t0:l1 b2 rungs 48/32 (+cyc), 4.00% grp, 1.61% | av:t0:l3 b2 rungs 48/32 (−cyc), 4.00% grp, 1.05% | 0.0674% | -0.388% | -0.52 | 0.9994 |
| 11 | within_down_proj **←sel** | down_proj:t0:l0 b9 rungs 40/48 (−cyc), 3.93% grp, 1.34% | down_proj:t0:l3 b13 rungs 80/96 (+cyc), 0.82% grp, 0.39% | 0.0198% | -1.138% | -2.53 | 0.9993 |

**30B_t40** — incumbent search NLL 2.15103, trace cost 42.509; selected for confirmation: `candidate07_adj_mlp_from_projections_07`

| # | family | move 1 (bucket, boundary, rungs, groups chg, bucket-cost chg) | move 2 | cycles moved (% total SC) | search dPPL | z | cost ratio |
|---:|---|---|---|---:|---:|---:|---:|
| 0 | qk_from_linears | qk:t0:l1 b0 rungs 96/65 (+cyc), 3.99% grp, 1.44% | down_proj:t0:l3 b12 rungs 64/80 (−cyc), 3.55% grp, 1.13% | 0.0684% | -0.296% | -0.53 | 1.0002 |
| 1 | linears_from_qk | qk:t0:l3 b0 rungs 96/65 (−cyc), 2.25% grp, 0.73% | o_proj:t0:l3 b12 rungs 64/80 (+cyc), 4.00% grp, 1.87% | 0.0533% | -0.122% | -0.70 | 0.9998 |
| 2 | av_from_linears | av:t0:l3 b0 rungs 96/65 (+cyc), 1.50% grp, 0.52% | up_proj:t0:l3 b9 rungs 40/48 (−cyc), 3.96% grp, 0.84% | 0.0392% | -0.168% | -0.69 | 0.9998 |
| 3 | linears_from_av | av:t0:l2 b3 rungs 47/32 (−cyc), 3.73% grp, 0.95% | down_proj:t0:l2 b12 rungs 64/80 (+cyc), 2.66% grp, 1.18% | 0.0472% | -0.357% | -1.02 | 0.9989 |
| 4 | qk_from_av | av:t0:l1 b3 rungs 47/32 (−cyc), 3.59% grp, 1.17% | qk:t0:l2 b0 rungs 96/65 (+cyc), 2.19% grp, 0.73% | 0.0453% | +0.538% | +0.90 | 0.9994 |
| 5 | av_from_qk | av:t0:l0 b3 rungs 47/32 (+cyc), 1.05% grp, 0.32% | qk:t0:l0 b1 rungs 65/49 (−cyc), 2.32% grp, 0.41% | 0.0100% | -0.033% | -0.08 | 1.0007 |
| 6 | projections_from_mlp | down_proj:t0:l1 b10 rungs 48/56 (−cyc), 3.29% grp, 0.84% | q_proj:t0:l3 b9 rungs 40/48 (+cyc), 3.98% grp, 0.89% | 0.0265% | +0.677% | +1.73 | 0.9989 |
| 7 | mlp_from_projections **←sel** | q_proj:t0:l2 b9 rungs 40/48 (−cyc), 4.00% grp, 1.17% | gate_proj:t0:l3 b12 rungs 64/80 (+cyc), 1.23% grp, 0.58% | 0.0243% | -0.644% | -2.00 | 0.9998 |
| 8 | within_projections | o_proj:t0:l0 b10 rungs 48/56 (+cyc), 1.99% grp, 0.71% | q_proj:t0:l1 b4 rungs 16/20 (−cyc), 3.97% grp, 0.73% | 0.0089% | -0.613% | -0.94 | 1.0000 |
| 9 | within_mlp | gate_proj:t0:l2 b8 rungs 32/40 (+cyc), 2.18% grp, 0.73% | up_proj:t0:l2 b9 rungs 40/48 (−cyc), 2.38% grp, 0.70% | 0.0217% | -0.496% | -1.24 | 0.9998 |
| 10 | within_av | av:t0:l2 b1 rungs 65/49 (+cyc), 4.00% grp, 1.09% | av:t0:l3 b0 rungs 96/65 (−cyc), 2.07% grp, 0.71% | 0.0539% | -0.254% | -0.76 | 1.0002 |
| 11 | within_down_proj | down_proj:t0:l0 b12 rungs 64/80 (+cyc), 3.92% grp, 1.82% | down_proj:t0:l3 b14 rungs 96/112 (−cyc), 1.63% grp, 0.52% | 0.0316% | -0.352% | -1.01 | 0.9997 |

**Confirmation (16 fresh windows) of the selected candidate** (`selected.json` → `confirmation.paired_vs_incumbent`):

| cell | selected | search dPPL | confirmation dPPL | z | gate (z<−1.5) | full test |
|---|---|---:|---:|---:|---|---|
| 4B t32 | #1 linears_from_qk | −0.446% | **+0.138%** | +0.48 | fail | not run |
| llama8B t32 | #9 within_o_proj (o_proj l0 56/64 up, l1 40/48 down; 3.97%/3.98% of groups; 0.019% of total cycles) | −0.749% | −0.398% | −1.98 | pass | **8.285860 vs 8.279485 = +0.077%**, cost 33.849 vs 33.863 (−0.04%) |
| 30B t32 | #11 within_down_proj | −1.138% | −0.329% | −0.68 | fail | not run |
| 30B t40 | #7 mlp_from_projections | −0.644% | −0.079% | −0.27 | fail | not run |

(Test from `llama8B_t32/full_test_result.json`, trace
`kbands_20260801/ppl/llama8B_t32_prc_p4adjacent20260927_trace.json`, 288,627 tokens, ctx 2048,
`ppl_max_tokens 0`, bitrev/64 masks. `candidate_beats_incumbent: false`; chosen wrapper stays `c6_gfis`.)

**Noise read-out of the round-4 search** (`inv/r4_noise.py`):

| cell | candidates with search dPPL<0 | mean / sd of candidate dPPL | median paired SE | cycles moved range | corr(\|dPPL\|, cycles moved) | mean pairwise corr of window-delta vectors |
|---|---:|---|---:|---|---:|---:|
| 4B t32 | 11/12 | −0.159% / 0.305% | 0.410% | 0.021–0.069% | +0.19 | +0.01 |
| llama8B t32 | 8/12 | −0.222% / 0.293% | 0.328% | 0.005–0.064% | −0.26 | +0.17 |
| 30B t32 | 10/12 | −0.444% / 0.422% | 0.466% | 0.017–0.067% | −0.20 | +0.32 |
| 30B t40 | 10/12 | −0.177% / 0.411% | 0.370% | 0.009–0.068% | −0.31 | +0.17 |

A second symptom: the realized total-cost change of a candidate on its own 6-window trajectory (|cost ratio − 1|, median 0.026/0.158/0.038/0.023% for 4B/llama8B/30B t32/30B t40) is as large as the planned transfer itself (median 0.044/0.023/0.030/0.035%); in 24/48 candidates it is larger (llama8B median 7x). The intended move is smaller than the model's own downstream response to it.

The spread of 12 candidates equals one paired SE and is unrelated to how much compute was moved, so
the search was measuring configuration noise, not allocation value. The shared negative offset (most
candidates "beat" the incumbent in the same windows, e.g. llama8B window 2 mean −0.0099) means the
incumbent's own 6-window score sat on the unlucky side of that noise. Selecting the minimum of 12
null draws gives the observed −0.45…−1.14% search "wins" (expected best-of-12 ≈ −1.6 SE ≈
−0.5…−0.75%), which regressed to −0.40…+0.14% on confirmation and +0.08% on test — a textbook
winner's curse. The one confirmation pass (llama8B, z −1.98) is a false positive: if its −0.40% were
real, the 141-window test (paired SE ≈ 0.20%×√(16/141) ≈ 0.07%) would have shown ≈ −0.4%, not +0.077%.

## 9. Power: how big must an allocation change be to be seen?

Budget elasticity measured inside single cells (c17 vs its own reduced-budget s80 table, same windows,
attention and mask; prc2 SUMMARY §2a costs, PAPER_METRICS PPLs):

| cell | c17 PPL @cost | s80 PPL @cost | dPPL / dcost | elasticity (%PPL per % total SC cycles) |
|---|---|---|---|---:|
| 4B t32 | 11.3305 @33.67 | 11.9836 @29.66 | +5.76% / −11.9% | **0.48** |
| llama8B t32 | 8.3899 @33.73 | 8.8363 @28.98 | +5.32% / −14.1% | 0.38 |
| 14B t32 | 9.1672 @33.19 | 9.4615 @28.19 | +3.21% / −15.1% | 0.21 |
| 4B t48 | 10.5030 @49.57 | 10.6761 @42.64 | +1.65% / −14.0% | 0.12 |

A transfer of Δ% of total SC cycles between populations whose marginal values differ by a fraction
m of the average buys ≈ Δ·e·m. Round 4's largest transfer (Δ = 0.0685%) at e = 0.48 buys **0.033%
per 100% mispricing**. The 16-window confirmation gate (paired SE 0.20–0.49%, z < −1.5) needs
≈ 0.3–0.7%, i.e. m ≈ 9–21; even the full test (≈0.07% noise) needs m ≈ 2. Round 4 could not have
confirmed a real improvement of any plausible size. Round 3's Δ = 2% (≈0.96% per unit m) was
detectable — and says m ≈ 0 or negative for every family direction tried. Round 2's successful moves
were +12–23% of attention's own budget (4B t40/t64, 30B t40/t48) or a large qk↑/av↓ re-split (30B t32). ⇒ A useful round needs total moves of order
**0.5–2% of total SC cycles**, spread as ≤1-rung changes over many buckets, in a direction chosen by
a signal (not a fixed family sweep), evaluated on ≥16 paired windows.

## 10. Suspected causes (ledger angle), ranked

1. **Post-round-2 rounds searched without directional information.** R3-Codex used 8 fixed family
   directions; R4 chose 12 of ~54k–91k pairs by bucket diversity and size (code lines 381–408), with
   no Fisher/error/loss marginal. One iteration each. Blind ±directions around an optimum mostly lose.
2. **Step size mis-sized both times.** R3: 2% of total cycles as a single common threshold offset
   (rung skips; −17…−20% of the affected operator's own budget) → 19/20 worse. R4: one boundary in each of only two buckets, with ≤4% of a bucket's groups/MACs
   crossing one rung (the 80%-planning cap binds in 40/48 proposals; the 2%-of-operator cap never binds,
   max 0.86%), made moves 0.005–0.069% of total cycles, 10–20x below detectability (§9) → pure noise
   + winner's curse.
3. **The operator-family budget split is already at its optimum after global Fisher (+ joint λ).**
   2% transfers lose in both directions on the proj↔MLP axis (4B +1.85/+1.04%, llama8B +0.67/+0.69%)
   and in every attention↔linear direction on 4B t32 and 30B t40. Further family re-splits cannot pay.
4. **Attention is the under-served population and the allocator cannot give it more.** Round 2 improved
   only when attention budget went UP (18/20 sign agreement, r = −0.77); cutting attention (which the
   diagonal-Fisher joint λ does in 11/20 cells) hurts. But in the incumbents, attention is pinned at its top
   rung: llama8B t32 qk/av **100%** of MACs at 96 (halved), 4B t32 qk 58.9% / av 68.6% at 97,
   30B t32 qk 77.3% at 96, 30B t40 qk 87.9% at 96 (profile replay, `inv/occupancy.py`, replay/actual
   = 1.000). The attention ladder tops out at 96/97 (nominal 192/194) while linears reach 128; only
   μ+2τ escape rows get 128. The ceiling raise was tested only on dense t32 at iso-cost with a
   uniform linear cut (R3c: neutral/worse), never on 30B where round 2 showed attention is under-funded.
5. **Per-group dispatch never touches attention.** T1's per-(row,chunk) path is only in `SCLinear`
   (`model/sc_common.py:904`); attention stays per-row (`attn_granularity: per_row` in every trace).
   Attention is 20.8% (llama8B t32), 31.4% (4B t32), 42.3% (30B t32), 40.2% (30B t40) of SC cycles in the
   round-4 incumbents, so the lever acts on at most 58–79% of the budget.
6. **Calibration variance is as large as the sought gains.** Re-running the same global-Fisher recipe
   (`c7_gfis` vs `c6_gfis`) shifts held-out PPL by −0.74…+0.72% on 4B; the single-sample MC Fisher is
   heavy-tailed (12-window draw → +32.5% vs parent). Best(all) already harvests some of this variance;
   it does not compound across rounds.
7. **Headroom shrinks with budget and with model.** Mean best(all) gain t32 −5.77% → t96 −0.46%; the
   parent's own linear error falls ~1/L² (PRC2_OVERNIGHT investigation); 14B t96 parent is already at
   1.0004×fp16; the requested h20 parents for 14B t64/t96 are themselves 0.48%/0.31% worse than the h10
   parents, and nothing allocation-only removes that.
8. **Coverage.** No refinement round after round 2 touched 14B or any t48–t96 cell; the only cells with
   real remaining distance to fp16 (llama8B t32 1.148×, 30B t32 1.182×, t40 1.11×) were piloted only
   with the two defective search designs.

## 11. Doc-vs-artifact discrepancies found

1. `scmp_llm/CLAUDE.md` STATUS (2026-09-27): "No round-4 PPL results yet" — stale. All four round-4
   tasks COMPLETED by 17:36 EDT (sacct); llama8B t32 test = 8.285860.
2. `SCMP/CLAUDE.md` calls `prc2/SUMMARY.md` "the source of truth for counts/means" and says 14B t64/t96
   parents are h=10%. SUMMARY.md (built 09-25 14:02) is stale vs `BEST_ALL_VS_SUBMITTED_20260927`:
   its §1b "final" lists 4B t40 10.8751, 14B t32 9.1041, llama8B t96 7.5855, llama8B t40 8.0300 and the
   h10 14B t64 row (parent 8.7341, r1 8.6763), whereas best(all) is 10.7729 / 9.0764 / 7.5723 / 8.0266 and
   h20 (8.7758 → 8.6993). Its rule text ("round 2 is test-evaluated only where it beat round 1 on
   held-out") is superseded by the round-2 finish (all 20 tested).
3. PRC2_OVERNIGHT 4B t32 round 2 "held-out gfisla vs gfis −0.82% (z −1.8)": that comparator is the
   regenerated `c7_gfis`; vs the actual incumbent `c6_gfis` it is −0.11% (z −0.22). (Already flagged by
   ROUND3_INVESTIGATION; confirmed.)
4. PRC_ADJACENT_20260927.md: "Requested transfers of 0.25% and 0.5%". Every one of the 48 emitted
   candidates has `requested_transfer_fraction` 0.0025; the 0.5% variants were all de-duplicated
   (`duplicate_pair` = `unique_matched_pairs` in every cell: 90,622 / 54,292 / 64,740 / 68,558) because
   the locality caps bind first. Realized transfers were 2–27% of the 0.25% ceiling.
5. `model/sc_common.py:480–483` docstring: "THE STATISTIC IS FREE … exactly the group quantization
   scale". The metric is the raw residual chunk absmax computed before `smooth_scales` is applied
   (TWO_PHASE_DESCRIPTION item 6); under AWQ it is not the quantization scale, and the paper rule forbids
   "free".
6. "Round 3" names two different things (R3-Claude calib8 family; R3-Codex prc_local). No artifact is
   labelled "round 5".
7. Minor oddity, not an error: calib7-vs-calib6 `gfis` held-out mean differences for llama8B t32/t40/t48
   are +0.001423/+0.001420/+0.001421 (t64 −0.001446) with different per-window vectors — coincidence,
   but worth knowing if anyone re-derives them.

## 12. What the ledger implies for any new round (constraints, not yet a plan)

* Do not repeat: family-level threshold offsets (R3-Codex), blind adjacent swaps (R4), raw 12-window MC
  Fisher, per-block keys, re-linearization at the r1 trajectory, dense-t32 attention-ceiling with a
  uniform linear cut, σ-currency global λ.
* Effect-size floor: aim for moves worth ≥0.5% PPL (≈0.5–2% of total SC cycles, one rung at a time,
  many buckets), selected by a signal; score on ≥16 paired windows (r = 0.97 with test at that scale);
  never pick the best of many candidates on 6 windows.
* Evidence-backed directions still untested at a detectable size: (a) attention budget UP where
  attention is pinned (raise its top rung inside the joint λ, not with a uniform linear cut), first on
  30B t32–t48 where round 2 proved attention under-funded and qk is now 77–88% pinned; (b) a joint λ
  that is never allowed to CUT attention (the 11/20 cells where the joint λ cut attention lost or tied, except 14B t32); (c) lower-
  variance Fisher (average several MC-label draws / windows) so the currency stops moving ±0.7% per re-run;
  (d) extend refinement to 14B and t48–t96, which no post-round-2 round touched.
* Whether (a) counts as "same QK behavior" (it changes qk stream lengths, not qk operands) and whether a
  static per-chunk class map (ROUND3_INVESTIGATION backup) counts as new runtime metadata are user calls.

## 13. Sources and scripts

* Docs: `scmp_llm/benchmark/ppl/kbands/{PRC2_OVERNIGHT.md, ROUND3_INVESTIGATION_20260926.md,
  PRC_LOCAL_20260926.md, ROUND3_POSTMORTEM_20260927.md, PRC_ADJACENT_20260927.md,
  TWO_PHASE_DESCRIPTION_20260927.md}`; JSON `prc_round2_status_20260927.json`,
  `prc_local_20260926{,_results,_submission}.json`, `prc_adjacent_20260927{,_submission}.json`,
  `prc_adjacent_30b_20260927{,_submission}.json`, `prc7_finish_20260926.tsv`.
* Archive: `hpca_results/llm/ppl/prc2/{BEST_ALL_VS_SUBMITTED_20260927.{md,csv,json}, PAPER_METRICS_20260927.md, SUMMARY.md}`.
* Turbo: `/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/{prc2/*_heldout_nll.json, prc2/*_table.json,
  prc_local_20260926/<cell>/, prc_adjacent_20260927/<cell>/, kbands_20260801/ppl/*_trace.json}`.
* Logs: `/scratch/nbleier_owned_root/nbleier_owned1/shared_data/allenjin/hpca/logs/_kbands/{p6_*,p7_*,p7finish_*,p3local*,p4adj*}.out`; sacct.
* Code read: `benchmark/ppl/prc_adjacent_proposals.py` (propose, 316–419), `kbands/run_prc6.sbatch`,
  `kbands/run_prc7.sbatch`, `model/sc_common.py` (462–530, 860–915).
* Scripts written for this ledger (all in `…/scratchpad/inv/`): `tab_r4.py`, `r4_md.py`, `r4_noise.py`,
  `occupancy.py`, `ledger_traces.py`, `decomp.py`, `r2_heldout_vs_test.py`.
