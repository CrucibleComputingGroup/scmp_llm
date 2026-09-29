# T1 stall investigation: mapper reports (2026-09-27/28)

This directory holds the five read-only mapper reports behind
[`../T1_STALL_INVESTIGATION_20260928.md`](../T1_STALL_INVESTIGATION_20260928.md), plus the round-6 critic report.
On 2026-09-28 the mapper reports were copied byte-for-byte from the session scratch directory
`/tmp/claude-114365137/-home-allenjin-Projects-SCMP/aad3505a-096e-4395-aba0-71871b9d6077/scratchpad/inv/`.
The critic report was copied from `…/scratchpad/r6/`. None of these reports ran GPU work, loaded a model, or changed
a repo file.

| file | scope | sha256 |
|---|---|---|
| `rounds.md` | Complete round ledger (c17 → round 1 → round 2 → R3 `prc_local` → R4/R5 `prc_adjacent`) and the per-round mean decomposition | `47772ddd35f5205745d7470e46e79c1bcd5e7ac7927f865ec5ec7440a15359da` |
| `gap.md` | Size of the gap for all 20 LLM cells vs fp16, the SC ceiling, `_3`/`_4` | `5302c14b6afe88423dcc24b51b2c1855ad00f911a33897a01aa1e4eda307b4a4` |
| `algo.md` | How the deployed per-group allocation works in code; the allocation degrees of freedom and which were tried | `a1f4db875f7af3a4ea6744ddda05ebda85459e0c969971968214d74faf71fc69` |
| `attrib.md` | Where the remaining PPL loss sits (linears vs attention vs ceiling) and how much allocation can still recover | `baa095c9a12d383318db25eccc599170cde8edd97e4349f06f3996782dcdb511` |
| `protocol.md` | Search/selection protocol audit of rounds 3–5 (power, MDE, held-out vs test agreement) | `6a2dbaa974c73f8fd96ba46a23c098ca203daf826e3d9a8fd15c4a92cb844792` |
| `r6_critic_20260928.md` | Round-6 critic, written after the investigation: the family-currency (κ_lin vs κ_att) finding on 30B, re-ranked levers L1–L8, and an audit of the investigation's Rounds A/B/C and §6. Copied from scratch `…/scratchpad/r6/critic.md`, where its `r6/` paths resolve. Used by [`../../ROUNDS_6_8_PLAN_20260928.md`](../../ROUNDS_6_8_PLAN_20260928.md). | `a168f4bfe1a5c129d2e1f9df6b858553f93c95e08ab634db9cc8e925e680fc42` |

## Path resolution inside the reports

- "This directory", `inv/`, `gap_work/`, and bare script/JSON names (`trace_occ2.py`, `occ_all.json`,
  `attrib_*.py`, `heldout_vs_test.py`, …) refer to the scratch directory above. Those helper scripts and
  outputs were **not** copied and are not durable.
- Turbo paths (`/nfs/turbo/coe-nbleier/allenjin/hpca/kbands/…`), `hpca_results/…`, and repo paths are
  durable.
- Units: code/trace stream lengths are HALVED (nominal = 2×); the code maximum is 128.
