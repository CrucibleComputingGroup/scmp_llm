"""OCS stationarity + headroom gate (offline, GPU, cheap).

Go/no-go for Outlier-Channel Split BEFORE writing the runtime. OCS assumes the
dangerous per-row-abs-max-collapsing outliers live at a FIXED, small set of
input channels C_hi (the LLM 'massive activation' premise). If they instead
wander per-token, a static C_hi misses them and OCS gains ~nothing.

Measures, on the SMOOTHED activation x/smooth (what the SC kernel actually
quantizes — apply_smoothing gives a/s), per operator (q/k/v/o/gate/up/down):
  * channel-consistency = fraction of token rows whose argmax_d |x_sm[row,d]|
    falls in the top-|C_hi| channels by global abs-max. (>=0.9 => GO)
  * global collapse ratio = max_d act_scale / max_{d not in C_hi} act_scale
    (how much the per-row scale is dominated by C_hi => gain proxy).
  * per-row collapse ratio (mean over rows of amax_full / amax_bulk) — the
    quantity that directly drives the SC quant-floor; the real gain proxy.
Reports both the SmoothQuant-ON view (deploy reality) so the "SmoothQuant
already fixed it" concern is answered with data.
"""
import os, sys, math
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))  # repo root
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                    # benchmark/ppl
from loader import load_sc_model                                    # noqa: E402
from model.smoothquant_apply import apply_smoothquant_from_env      # noqa: E402
from model.sc_common import SCLinear                                # noqa: E402
from datasets import load_dataset                                   # noqa: E402
from transformers import AutoTokenizer                              # noqa: E402

MODEL_PATH = os.environ["MODEL_PATH"]
CTX = int(os.environ.get("CTX", "1024"))
NWIN = int(os.environ.get("OCS_WINDOWS", "4"))
KHI = int(os.environ.get("OCS_KHI", "8"))          # |C_hi|
TAU = float(os.environ.get("OCS_TAU", "10.0"))     # outlier threshold x median

device = "cuda"
model = load_sc_model(MODEL_PATH, dtype=torch.float16, device_map="auto")
n_sq = apply_smoothquant_from_env(model)
print(f"[ocs_gate] smoothquant applied to {n_sq} modules "
      f"(USE_SMOOTHQUANT={os.environ.get('USE_SMOOTHQUANT')})", flush=True)
model.eval()

# per-(op) accumulators aggregated over all blocks
from collections import defaultdict
chan_amax = {}          # op -> (D,) running max |x_sm|
argmax_cnt = {}         # op -> (D,) count of rows whose argmax==d
n_rows = defaultdict(int)
# per-row collapse ratio running sum (needs C_hi; do it in a 2nd micro-pass per
# batch using the CURRENT global top-k, good enough as we accumulate)
collapse_num = defaultdict(float)   # sum over rows of amax_full/amax_bulk
collapse_cnt = defaultdict(int)

def make_hook(op):
    def hook(mod, inp):
        x = inp[0]
        D = x.shape[-1]
        xf = x.reshape(-1, D).to(torch.float32)
        s = getattr(mod, "smooth_scales", None)
        if s is not None:
            xf = xf / s.to(torch.float32).reshape(1, D)   # apply_smoothing: a/s
        a = xf.abs()
        cmax = a.amax(dim=0)                               # (D,)
        if op not in chan_amax:
            chan_amax[op] = cmax.clone()
            argmax_cnt[op] = torch.zeros(D, device=cmax.device)
        else:
            chan_amax[op] = torch.maximum(chan_amax[op], cmax)
        am = a.argmax(dim=1)                               # (rows,)
        argmax_cnt[op].index_add_(0, am, torch.ones_like(am, dtype=torch.float32))
        n_rows[op] += xf.shape[0]
        # per-row collapse vs CURRENT global top-k channels
        khi = min(KHI, D)
        chi = torch.topk(chan_amax[op], khi).indices
        full = a.amax(dim=1)                               # (rows,)
        a_bulk = a.clone()
        a_bulk[:, chi] = 0.0
        bulk = a_bulk.amax(dim=1).clamp(min=1e-6)
        ratio = (full / bulk)
        collapse_num[op] += float(ratio.sum().item())
        collapse_cnt[op] += ratio.numel()
    return hook

hooks = []
for m in model.modules():
    if isinstance(m, SCLinear):
        op = getattr(m, "_sc_op_name", None)
        if op is None:
            continue
        # normalize op name to base (strip block idx if embedded)
        hooks.append(m.register_forward_pre_hook(make_hook(op)))

# wikitext-2 windows (same corpus/ctx as calibration)
tok = AutoTokenizer.from_pretrained(MODEL_PATH)
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
text = "\n\n".join(ds["text"])
enc = tok(text, return_tensors="pt").input_ids[0]
print(f"[ocs_gate] {enc.numel()} tokens; running {NWIN} windows of ctx={CTX}", flush=True)
with torch.no_grad():
    for w in range(NWIN):
        s = w * CTX
        e = s + CTX
        if e > enc.numel():
            break
        ids = enc[s:e].unsqueeze(0).to(device)
        model(ids)
        print(f"  window {w} done", flush=True)

for h in hooks:
    h.remove()

print("\n================ OCS GATE RESULTS (smoothed activation x/s) ================")
print(f"{'op':<12}{'D':>6}{'|C_hi|':>7}{'consistency':>13}{'glob_collapse':>15}{'row_collapse':>14}")
ops_order = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
seen = set()
def op_key(op):
    for o in ops_order:
        if o in op: return o
    return op
# fold per-module op names into base op
agg_amax=defaultdict(lambda:None); agg_arg=defaultdict(lambda:None); agg_nr=defaultdict(int)
agg_cn=defaultdict(float); agg_cc=defaultdict(int)
for op in chan_amax:
    b=op_key(op)
    agg_amax[b]= chan_amax[op] if agg_amax[b] is None else torch.maximum(agg_amax[b],chan_amax[op])
    agg_arg[b]= argmax_cnt[op].clone() if agg_arg[b] is None else (agg_arg[b]+argmax_cnt[op] if agg_arg[b].shape==argmax_cnt[op].shape else agg_arg[b])
    agg_nr[b]+= n_rows[op]; agg_cn[b]+=collapse_num[op]; agg_cc[b]+=collapse_cnt[op]

results={}
for b in ops_order:
    if agg_amax[b] is None: continue
    amax=agg_amax[b]; D=amax.numel(); khi=min(KHI,D)
    chi=torch.topk(amax,khi).indices
    cons=float(agg_arg[b][chi].sum().item())/max(agg_nr[b],1)
    bulk_amax=amax.clone(); bulk_amax[chi]=0.0
    glob=float(amax.max().item()/max(bulk_amax.max().item(),1e-6))
    rowc=agg_cn[b]/max(agg_cc[b],1)
    med=float(amax.median().item())
    n_over_tau=int((amax>TAU*med).sum().item())
    results[b]=dict(D=D,khi=khi,consistency=cons,glob_collapse=glob,row_collapse=rowc,n_over_tau=n_over_tau)
    print(f"{b:<12}{D:>6}{khi:>7}{cons:>13.3f}{glob:>15.2f}{rowc:>14.2f}  (chans>{TAU:.0f}x median: {n_over_tau})")

print("\n---- VERDICT ----")
for b in ("down_proj","k_proj"):
    if b in results:
        r=results[b]
        go = r["consistency"]>=0.9
        print(f"{b}: consistency={r['consistency']:.3f} -> {'GO (stationary)' if go else 'NO-GO (wandering outliers -> pivot to per-row masked-amax)'}; "
              f"row_collapse={r['row_collapse']:.2f} (gain proxy; >~3 means real quant-floor headroom)")
print("Note: high row_collapse + SmoothQuant ON = SmoothQuant did NOT fully remove the per-row outlier -> OCS headroom exists.")
