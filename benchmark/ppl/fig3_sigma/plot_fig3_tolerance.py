"""Fig. 3 (error tolerance, LLM): 2x2 3D-surface panel from the raw sigma dump.

Same visual style as the original ViT companion figure
(scmp_vit/cls/sensitivity/plot_heatmap_3d_surface.py): smooth viridis_r
surfaces, orthographic projection, quiet rows in front / loud ridge at the
back. Columns = NOMINAL stream length 256 / 128 (halved 128 / 64):

  row (a)  per-OPERATOR: surface over (block index x operator class),
           z = mean per-group sigma (relative-L2 SC reconstruction error vs
           the FP16 teacher). Shared z/color scale across both columns.
  row (b)  per-GROUP, one operator (auto: largest mean sigma at the shorter
           L, or --focus-op): surface over (block index x group), where each
           block's groups are sorted by sigma so the axis is the group's
           sorted position (%); the worst groups form the back ridge. Shows
           that within a single operator most groups are nearly free while a
           small tail concentrates the error.

  python benchmark/ppl/fig3_sigma/plot_fig3_tolerance.py \
      --npz /nfs/turbo/.../qwen4b_raw_sigma_L128_64.npz \
      --out /nfs/turbo/.../fig_tolerance_llm
"""
from __future__ import annotations

import argparse
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402
from scipy.ndimage import zoom  # noqa: E402

CMAP = "viridis_r"
UP_OP = 8    # upsample factors, as in the ViT script (op x block: 9x36 -> 72x144)
UP_BLK = 4
N_Q = 140    # sorted-group sample points per block in row (b)

PRETTY = {
    "q_proj": r"$\mathbf{W_{Q}}$",
    "k_proj": r"$\mathbf{W_{K}}$",
    "v_proj": r"$\mathbf{W_{V}}$",
    "o_proj": r"$\mathbf{W_{O}}$",
    "gate_proj": r"$\mathbf{W_{gate}}$",
    "up_proj": r"$\mathbf{W_{up}}$",
    "down_proj": r"$\mathbf{W_{down}}$",
    "qk": r"$\mathbf{Q}\!\cdot\!\mathbf{K}^{\!\top}$",
    "av": r"$\mathbf{A}\!\cdot\!\mathbf{V}$",
}
# Compact tick labels for the single-column row-(a) y axis (9 rows collide
# with the full W_x forms under the 3D foreshortening).
COMPACT = {
    "q_proj": r"$\mathbf{Q}$",
    "k_proj": r"$\mathbf{K}$",
    "v_proj": r"$\mathbf{V}$",
    "o_proj": r"$\mathbf{O}$",
    "gate_proj": "gate",
    "up_proj": "up",
    "down_proj": "down",
    "qk": r"$\mathbf{Q}\!\cdot\!\mathbf{K}$",
    "av": r"$\mathbf{A}\!\cdot\!\mathbf{V}$",
}
KEY_RE = re.compile(r"^(?P<op>.+)\.t(?P<t>\d+)\.l(?P<layer>\d+)\.errors$")


def load_sigma(npz_path):
    """-> levels_nominal [K] (descending), sigma[op][layer] = float32 [n, K]."""
    z = np.load(npz_path)
    levels_halved = z["_levels_halved"]
    data: dict[str, dict[int, np.ndarray]] = {}
    for key in z.files:
        m = KEY_RE.match(key)
        if not m:
            continue
        data.setdefault(m["op"], {})[int(m["layer"])] = z[key]
    levels_nom = 2 * levels_halved
    order = np.argsort(-levels_nom)
    for per in data.values():
        for layer in per:
            per[layer] = per[layer][:, order]
    return levels_nom[order], data


def op_block_matrix(data, ops, n_layers, col):
    M = np.full((len(ops), n_layers), np.nan)
    for i, op in enumerate(ops):
        for layer, err in data[op].items():
            M[i, layer] = float(err[:, col].mean())
    return M


def group_block_matrix(data, focus, n_layers, col):
    """[N_Q, n_layers]; row 0 = quietest group (front), last row = loudest."""
    M = np.zeros((N_Q, n_layers))
    frac = np.linspace(1.0, 0.0, N_Q)          # front -> back = quiet -> loud
    for layer in sorted(data[focus]):
        s = np.sort(data[focus][layer][:, col])[::-1]   # descending
        idx = np.minimum((frac * (s.size - 1)).astype(int), s.size - 1)
        M[:, layer] = s[idx]
    return M


def style_axis(ax, n_x, vmax, y_aspect=1.7):
    ax.set_xlim(0, n_x - 1)
    ax.set_zlim(0, vmax * 1.02)
    ax.set_xticks(list(range(0, n_x, 10)))
    ax.set_xlabel("block index", labelpad=6, fontweight="bold")
    ax.set_zticklabels([])
    ax.set_proj_type("ortho")
    for t in ax.get_xticklabels():
        t.set_fontweight("bold")
    ax.tick_params(axis="x", pad=0)
    ax.tick_params(axis="y", pad=1)
    ax.view_init(elev=22, azim=-44)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_alpha(0.0)
    ax.grid(True, linewidth=0.3, alpha=0.3)
    ax.set_box_aspect((2.5, y_aspect, 1.1))


def add_cbar(fig, norm, cmap, rect, label):
    cax = fig.add_axes(rect)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(label, rotation=90, labelpad=5, fontsize=9.5, fontweight="bold")
    cb.ax.tick_params(labelsize=9)
    for t in cb.ax.get_yticklabels():
        t.set_fontweight("bold")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True, help="output stem (.pdf/.png/.stats.json)")
    ap.add_argument("--focus-op", default=None,
                    help="operator for row (b); default = largest mean sigma at the shorter L")
    args = ap.parse_args()

    levels_nom, data = load_sigma(args.npz)
    n_cols = len(levels_nom)
    short_col = n_cols - 1                     # levels are descending
    n_layers = 1 + max(l for per in data.values() for l in per)

    ops_all = [op for op in PRETTY if op in data]
    loud = {op: np.nanmean(op_block_matrix(data, [op], n_layers, short_col)[0])
            for op in ops_all}
    focus = args.focus_op or max(loud, key=loud.get)
    if focus not in data:
        raise SystemExit(f"--focus-op {focus} not in npz (have {sorted(data)})")

    # Quiet ops in front (y=0), loudest at the back — as in the ViT figure.
    ops_plot = sorted(ops_all, key=lambda o: loud[o])
    Ma = [op_block_matrix(data, ops_plot, n_layers, c) for c in range(n_cols)]
    Mb = [group_block_matrix(data, focus, n_layers, c) for c in range(n_cols)]
    n_ops = len(ops_plot)

    cmap = mpl.colormaps[CMAP]
    norm_a = mpl.colors.Normalize(vmin=0.0, vmax=max(np.nanmax(M) for M in Ma))
    norm_b = mpl.colors.Normalize(vmin=0.0, vmax=max(np.nanmax(M) for M in Mb))

    # Sized for a SINGLE IEEE column: rendered at ~6.6 in wide and scaled to
    # \linewidth (~3.5 in) in the paper, so print sizes are ~0.53x these.
    # Serif/Times to match IEEEtran body text; STIX for Times-like mathtext.
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman",
                       "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.titlesize": 12.5,
        "axes.labelsize": 11.5,
        "axes.labelweight": "bold",
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "font.weight": "normal",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(6.6, 6.2))

    # ---- row (a): (block x operator) surfaces ------------------------------
    big_y = np.linspace(0.0, n_ops - 1.0, n_ops * UP_OP)
    big_x = np.linspace(0.0, n_layers - 1.0, n_layers * UP_BLK)
    XXa, YYa = np.meshgrid(big_x, big_y)
    for c in range(n_cols):
        ax = fig.add_subplot(2, n_cols, c + 1, projection="3d")
        Z = np.clip(zoom(Ma[c], zoom=(UP_OP, UP_BLK), order=3, mode="nearest"),
                    0.0, None)
        ax.plot_surface(XXa, YYa, Z, cmap=cmap, norm=norm_a,
                        rcount=80, ccount=160,
                        linewidth=0, antialiased=True, shade=True)
        ax.set_title(rf"Nominal $L = {levels_nom[c]}$",
                     fontsize=15, fontweight="bold", y=0.92, pad=0)
        ax.set_ylim(0, n_ops - 1)
        ax.set_yticks(range(n_ops))
        ax.set_yticklabels([COMPACT[o] for o in ops_plot], fontweight="bold",
                           fontsize=8, rotation=-22, ha="left", va="center")
        style_axis(ax, n_layers, norm_a.vmax, y_aspect=2.35)

    # ---- row (b): (block x sorted group) surfaces for the focus op ---------
    big_g = np.linspace(0.0, N_Q - 1.0, N_Q * 2)
    XXb, YYb = np.meshgrid(big_x, big_g)
    for c in range(n_cols):
        ax = fig.add_subplot(2, n_cols, n_cols + c + 1, projection="3d")
        Z = np.clip(zoom(Mb[c], zoom=(2, UP_BLK), order=3, mode="nearest"),
                    0.0, None)
        ax.plot_surface(XXb, YYb, Z, cmap=cmap, norm=norm_b,
                        rcount=80, ccount=160,
                        linewidth=0, antialiased=True, shade=True)
        # Front (y=0) = quietest group = sorted position 100%; back = worst.
        ax.set_ylim(0, N_Q - 1)
        tick_frac = [1.0, 0.5, 0.0]
        ax.set_yticks([(1.0 - f) * (N_Q - 1) for f in tick_frac])
        ax.set_yticklabels([f"{int(f * 100)}" for f in tick_frac],
                           fontweight="bold")
        ax.set_ylabel("group (%)", labelpad=2, fontweight="bold")
        style_axis(ax, n_layers, norm_b.vmax)

    fig.subplots_adjust(left=0.02, right=0.86, top=0.99, bottom=0.02,
                        wspace=0.14, hspace=-0.06)
    add_cbar(fig, norm_a, cmap, [0.92, 0.56, 0.015, 0.34],
             r"mean per-group $\sigma$ (rel. $\ell_2$ vs FP16)")
    add_cbar(fig, norm_b, cmap, [0.92, 0.10, 0.015, 0.34],
             r"per-group $\sigma$ (rel. $\ell_2$ vs FP16)")
    fig.text(0.012, 0.94, "(a)", fontsize=14, fontweight="bold")
    fig.text(0.012, 0.47, rf"(b) {PRETTY[focus]} groups",
             fontsize=14, fontweight="bold")

    fig.savefig(args.out + ".pdf", bbox_inches="tight")
    fig.savefig(args.out + ".png", dpi=200, bbox_inches="tight")

    # ---- caption-ready stats ------------------------------------------------
    stats = {"levels_nominal": levels_nom.tolist(), "focus_op": focus,
             "per_level": {}}
    for c, L in enumerate(levels_nom):
        m = Ma[c]
        flat = [(ops_plot[i], l, m[i, l]) for i in range(n_ops)
                for l in range(n_layers) if np.isfinite(m[i, l])]
        top = sorted(flat, key=lambda r: -r[2])[:5]
        pooled = np.concatenate([data[focus][l][:, c] for l in sorted(data[focus])])
        p = np.percentile(pooled, [50, 90, 99])
        share = np.sort(pooled ** 2)[::-1]
        top1_share = share[: max(1, share.size // 100)].sum() / share.sum()
        stats["per_level"][int(L)] = {
            "top_sites": [{"op": o, "layer": l, "mean_sigma": round(v, 5)}
                          for o, l, v in top],
            "focus_pooled_p50_p90_p99": [round(float(x), 5) for x in p],
            "focus_top1pct_share_of_sq_error": round(float(top1_share), 4),
            "op_mean_sigma": {o: round(float(np.nanmean(m[i])), 5)
                              for i, o in enumerate(ops_plot)},
        }
    with open(args.out + ".stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[plot_fig3] focus_op={focus}; wrote {args.out}.pdf/.png/.stats.json")


if __name__ == "__main__":
    main()
