"""
1205_visualize_results.py
=========================
SCRIPT 4 of 4 -- publication-quality figures.

Figures produced
----------------
1. signal_level.png           -- clean vs noisy signals for both protocols
2. parameter_recovery.png     -- one panel per param, HCP vs Novel lines
3. stage6_T2_spotlight.png    -- posterior densities of T2_t and T2_s
4. stage6_corner_*.png        -- 7-param corner plots
5. degeneracy_widths.png      -- 68% CI widths across stages
6. bias_heatmap.png           -- Stage 6 bias, compact HCP vs Novel
7. width_ratio.png            -- Novel/HCP posterior-width ratio
8. convergence_diagnostics.png -- trace plots + Rhat for a representative case
9. classification_roc.png     -- pseudo-ROC from existing posteriors

Output: results_1205/figures/
"""

from __future__ import annotations

import importlib
import os

import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend("Agg")

try:
    import corner
    HAS_CORNER = True
except ImportError:
    HAS_CORNER = False

cfg = importlib.import_module("1205_config")

CHAINS_DIR        = cfg.CHAINS_DIR
SIGNALS_DIR       = cfg.SIGNALS_DIR
FIGURES_DIR       = cfg.FIGURES_DIR
ensure_dirs       = cfg.ensure_dirs
PROTOCOLS         = cfg.PROTOCOLS
GLIA_FRACS        = cfg.GLIA_FRACS
STAGE_NAMES       = cfg.STAGE_NAMES
STAGE_FREE_PARAMS = cfg.STAGE_FREE_PARAMS
PARAM_LATEX       = cfg.PARAM_LATEX
V_IC_TRUE         = cfg.V_IC_TRUE
ODI_TRUE          = cfg.ODI_TRUE
T2_TISSUE_TRUE    = cfg.T2_TISSUE_TRUE
T2_SPHERE_TRUE    = cfg.T2_SPHERE_TRUE
THETA_TRUE        = cfg.THETA_TRUE
PHI_TRUE          = cfg.PHI_TRUE

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

PROTO_COLOUR = {"hcp": "#444444", "novel": "#c9302c"}
PROTO_STYLE  = {"hcp": "--",      "novel": "-"}
PARAM_COLOUR = {
    "v_ic": "#1f77b4", "ODI": "#2ca02c", "v_glia": "#9467bd",
    "T2_t": "#ff7f0e", "T2_s": "#8c564b",
    "theta": "#17becf", "phi": "#e377c2",
}
_MISSPEC_STAGES = {"stage2", "stage4"}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def _vg_str(vg):
    return f"vg{vg:.2f}".replace(".", "p")

def _chain_path(proto, stage, vg):
    return os.path.join(CHAINS_DIR, f"{proto}_{stage}_{_vg_str(vg)}.npz")

def _signal_path(proto, stage, vg, kind):
    return os.path.join(SIGNALS_DIR, f"{proto}_{stage}_{_vg_str(vg)}_{kind}.npy")

def _load_chain(proto, stage, vg):
    path = _chain_path(proto, stage, vg)
    if not os.path.exists(path):
        return None
    with np.load(path, allow_pickle=True) as d:
        return (d["flat"].copy(), d["truths"].copy(),
                list(d["param_names"]),
                d.get("split_rhat", None),
                d.get("autocorr_time", None))

def _save(fig, name):
    out = os.path.join(FIGURES_DIR, name)
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


# ===================================================================
# Figure 1 -- Signal-level
# ===================================================================
def plot_signal_level():
    """Clean vs noisy signals for both protocols at vg=0 and vg=0.30,
    showing multi-TE overlay for Novel."""

    vg_show = [0.0, 0.30]
    fig, axes = plt.subplots(len(vg_show), 2, figsize=(12, 4.5 * len(vg_show)),
                              sharey="row")
    fig.patch.set_facecolor("white")

    for row, vg in enumerate(vg_show):
        for col, proto_name in enumerate(PROTOCOLS):
            ax = axes[row, col]
            protocol = PROTOCOLS[proto_name]
            stage = "stage6"

            # Load clean signal
            clean_path = _signal_path(proto_name, stage, vg, "clean")
            noisy_path = _signal_path(proto_name, stage, vg, "noisy")
            if not os.path.exists(clean_path):
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                        ha="center", va="center", color="grey")
                continue

            S_clean = np.load(clean_path)   # (N_TE, N_B, N_GRAD)
            S_noisy = np.load(noisy_path)

            if S_clean.ndim == 2:
                S_clean = S_clean[np.newaxis, ...]
                S_noisy = S_noisy[np.newaxis, ...]

            n_te = S_clean.shape[0]
            te_vals = protocol.te_multi

            # Plot mean signal across gradient directions vs b-value
            for i_te in range(n_te):
                clean_mean = S_clean[i_te].mean(axis=1)   # (N_B,)
                noisy_mean = S_noisy[i_te].mean(axis=1)
                b_vals = protocol.b_values * 1000   # s/mm^2

                color = plt.cm.viridis(i_te / max(n_te - 1, 1))
                ax.plot(b_vals, clean_mean, "o-", color=color, lw=2,
                        label=f"TE={te_vals[i_te]:.0f} ms (clean)")
                ax.plot(b_vals, noisy_mean, "x", color=color, ms=8,
                        alpha=0.6, label=f"TE={te_vals[i_te]:.0f} ms (noisy)")

            ax.set_xlabel("b [s/mm$^2$]")
            if col == 0:
                ax.set_ylabel("Signal (mean over directions)")
            ax.set_title(f"{protocol.label}   |   "
                         rf"$v_{{glia}}^{{\rm true}}={vg:.2f}$")
            ax.legend(fontsize=7, ncol=1)
            ax.grid(True, alpha=0.25)

    fig.suptitle("Synthetic signals: clean vs noisy, mean over gradient directions",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    _save(fig, "signal_level.png")


# ===================================================================
# Figure 2 -- Parameter recovery (replaces messy master_recovery_grid)
# ===================================================================
def plot_parameter_recovery():
    """One subplot per parameter.  Each subplot shows relative bias vs v_glia,
    with HCP and Novel as distinct lines.  Only the *key* stages are shown
    (1, 3, 6) to keep it readable; mis-spec stages go to supplementary."""

    # Which stages to show per parameter (skip mis-spec for clarity)
    show_stages = {
        "v_ic":   ["stage1", "stage3", "stage6"],
        "ODI":    ["stage1", "stage3", "stage6"],
        "v_glia": ["stage3", "stage6"],
        "T2_t":   ["stage6"],
        "T2_s":   ["stage6"],
        "theta":  ["stage1", "stage3", "stage6"],
        "phi":    ["stage1", "stage3", "stage6"],
    }

    all_params = list(show_stages.keys())
    n = len(all_params)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.8 * nrows))
    axes = axes.ravel()
    fig.patch.set_facecolor("white")

    stage_markers = {"stage1": "o", "stage3": "s", "stage6": "D"}

    for idx, param in enumerate(all_params):
        ax = axes[idx]
        ax.axhline(0, color="black", lw=0.8, alpha=0.5)

        for stage in show_stages[param]:
            if param not in STAGE_FREE_PARAMS[stage]:
                continue
            marker = stage_markers.get(stage, "o")

            for proto_name in PROTOCOLS:
                xs, meds = [], []
                for vg in GLIA_FRACS:
                    if stage == "stage1" and vg != GLIA_FRACS[0]:
                        continue
                    res = _load_chain(proto_name, stage, vg)
                    if res is None:
                        continue
                    flat, truths, names, _, _ = res
                    j = names.index(param)
                    truth = float(truths[j])
                    median = float(np.percentile(flat[:, j], 50))
                    scale = abs(truth) if abs(truth) > 1e-3 else 1.0
                    xs.append(vg)
                    meds.append((median - truth) / scale)

                if not xs:
                    continue
                ax.plot(xs, meds, marker=marker, ls=PROTO_STYLE[proto_name],
                        color=PROTO_COLOUR[proto_name], lw=1.4, ms=5,
                        label=f"{PROTOCOLS[proto_name].label} {stage}")

        ax.set_title(PARAM_LATEX.get(param, param))
        ax.set_xlabel(r"$v_{glia}$ (truth)")
        ax.set_ylabel("Relative bias")
        ax.set_xticks(GLIA_FRACS)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=6, ncol=2, loc="best")

    # Hide unused axes
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Parameter recovery: relative bias by stage and protocol\n"
                 r"$(\hat\theta - \theta_{\rm true})/|\theta_{\rm true}|$",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    _save(fig, "parameter_recovery.png")


# ===================================================================
# Figure 3 -- Stage 6 T2 spotlight
# ===================================================================
def plot_stage6_T2_spotlight():
    """Posterior density of T2_t and T2_s, HCP vs Novel side-by-side."""

    n_vg = len(GLIA_FRACS)
    T2_PARAMS = ("T2_t", "T2_s")
    truths_map = {"T2_t": T2_TISSUE_TRUE, "T2_s": T2_SPHERE_TRUE}

    # Collect all samples for consistent x-limits per row
    row_samples = {0: [], 1: []}
    cache = {r: {c: {} for c in range(n_vg)} for r in range(2)}

    for col, vg in enumerate(GLIA_FRACS):
        for row, p in enumerate(T2_PARAMS):
            for proto_name in PROTOCOLS:
                res = _load_chain(proto_name, "stage6", vg)
                if res is None:
                    cache[row][col][proto_name] = None
                    continue
                flat, _, names, _, _ = res
                j = names.index(p)
                s = flat[:, j]
                cache[row][col][proto_name] = s
                row_samples[row].append(s)

    _FALLBACK = {0: (10, 300), 1: (5, 200)}
    row_xlim = {}
    for row in range(2):
        if row_samples[row]:
            all_s = np.concatenate(row_samples[row])
            lo, hi = np.percentile(all_s, 1), np.percentile(all_s, 99)
            pad = max((hi - lo) * 0.15, 2.0)
            row_xlim[row] = (lo - pad, hi + pad)
        else:
            row_xlim[row] = _FALLBACK[row]

    fig, axes = plt.subplots(2, n_vg, figsize=(4.0 * n_vg, 7.0), sharey="row")
    fig.patch.set_facecolor("white")

    for col, vg in enumerate(GLIA_FRACS):
        for row, p in enumerate(T2_PARAMS):
            ax = axes[row, col]
            truth_val = truths_map[p]
            any_data = False

            for proto_name in PROTOCOLS:
                s = cache[row][col].get(proto_name)
                if s is None:
                    continue
                any_data = True
                ax.hist(s, bins=50, alpha=0.55, density=True,
                        color=PROTO_COLOUR[proto_name],
                        label=PROTOCOLS[proto_name].label,
                        histtype="stepfilled", edgecolor="black", lw=0.4)

            ax.axvline(truth_val, color="black", lw=1.5, ls="--",
                       label=f"truth = {truth_val} ms")
            if not any_data:
                ax.text(0.5, 0.55, "no chain data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=9,
                        color="grey", style="italic")
            ax.set_xlim(*row_xlim[row])
            if row == 0:
                ax.set_title(rf"$v_{{glia}}^{{\rm true}}={vg:.2f}$")
            if col == 0:
                ax.set_ylabel(f"density of {PARAM_LATEX[p]}", fontweight="bold")
            ax.set_xlabel(PARAM_LATEX[p])
            ax.legend(fontsize=7)

    fig.suptitle(
        "Stage 6: $T_{2,t}$ and $T_{2,s}$ posterior densities, HCP vs Novel",
        fontsize=12, y=1.02)
    plt.tight_layout()
    _save(fig, "stage6_T2_spotlight.png")


# ===================================================================
# Figure 4 -- Corner plots
# ===================================================================
def plot_stage6_corners():
    if not HAS_CORNER:
        print("  [skip] corner not installed")
        return

    labels = [PARAM_LATEX[p] for p in STAGE_FREE_PARAMS["stage6"]]

    for proto_name in PROTOCOLS:
        for vg in GLIA_FRACS:
            res = _load_chain(proto_name, "stage6", vg)
            if res is None:
                continue
            flat, truths, _, _, _ = res
            truths_list = list(truths)

            try:
                fig = corner.corner(
                    flat, labels=labels, truths=truths_list,
                    truth_color="red", quantiles=[0.16, 0.5, 0.84],
                    show_titles=True, title_fmt=".3f",
                    color="steelblue",
                    label_kwargs={"fontsize": 10},
                    title_kwargs={"fontsize": 8},
                )
            except Exception as e:
                print(f"  [warn] corner failed {proto_name} vg={vg}: {e}")
                continue
            fig.suptitle(
                f"Stage 6 -- {PROTOCOLS[proto_name].label} | "
                rf"$v_{{glia}}^{{\rm true}}={vg:.2f}$",
                y=1.02, fontsize=12)
            _save(fig, f"stage6_corner_{proto_name}_{_vg_str(vg)}.png")


# ===================================================================
# Figure 5 -- Degeneracy widths
# ===================================================================
def plot_degeneracy_widths():
    """68% CI width for every free param across all 6 stages."""

    # Exclude orientation from this plot (less interesting for the headline)
    params_to_show = ["v_ic", "ODI", "v_glia", "T2_t", "T2_s"]
    layout = [("v_ic", 0, 0), ("ODI", 0, 1), ("v_glia", 0, 2),
              ("T2_t", 1, 0), ("T2_s", 1, 1)]

    fig, axes = plt.subplots(2, 3, figsize=(5.5 * 3, 4.5 * 2), sharey=False)
    fig.patch.set_facecolor("white")
    axes[1, 2].set_visible(False)

    x = np.arange(len(STAGE_NAMES))
    bw = 0.35

    for p, gr, gc in layout:
        ax = axes[gr, gc]
        for k, proto_name in enumerate(PROTOCOLS):
            heights = []
            for stage in STAGE_NAMES:
                if p not in STAGE_FREE_PARAMS[stage]:
                    heights.append(np.nan)
                    continue
                ws = []
                for vg in GLIA_FRACS:
                    if stage == "stage1" and vg != GLIA_FRACS[0]:
                        continue
                    # Skip unidentifiable cells (T2_s / v_glia at vg=0)
                    if vg == 0.0 and p in ("T2_s", "v_glia"):
                        continue
                    res = _load_chain(proto_name, stage, vg)
                    if res is None:
                        continue
                    flat, _, names, _, _ = res
                    j = names.index(p)
                    iqr = float(np.percentile(flat[:, j], 84)
                                - np.percentile(flat[:, j], 16))
                    ws.append(iqr)
                heights.append(np.mean(ws) if ws else np.nan)

            bar_x = x + (k - 0.5) * bw
            valid = np.array([not np.isnan(h) for h in heights])
            h_arr = np.array([h if not np.isnan(h) else 0 for h in heights])
            ax.bar(bar_x[valid], h_arr[valid], width=bw,
                   color=PROTO_COLOUR[proto_name],
                   label=PROTOCOLS[proto_name].label,
                   edgecolor="black", lw=0.5)

        for i, stage in enumerate(STAGE_NAMES):
            if p not in STAGE_FREE_PARAMS[stage]:
                ax.bar(x[i], 0.01, width=bw * 2.1,
                       color="none", edgecolor="#bbb", hatch="////",
                       lw=0.5, zorder=0)
                ax.text(x[i], 0, "N/A", ha="center", va="bottom",
                        fontsize=7, color="#aaa")
            if stage in _MISSPEC_STAGES:
                ax.axvspan(x[i] - 0.5, x[i] + 0.5, color="#fff0f0",
                           zorder=0, alpha=0.8)

        slab = [s + ("\n(mis)" if s in _MISSPEC_STAGES else "")
                for s in STAGE_NAMES]
        ax.set_xticks(x)
        ax.set_xticklabels(slab, rotation=15, fontsize=8)
        ax.set_ylabel(f"68% CI width of {PARAM_LATEX[p]}", fontweight="bold")
        ax.set_title(PARAM_LATEX[p])
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle("Posterior 68% CI widths across stages and parameters\n"
                 "(mean over $v_{glia}$ sweep)", fontsize=11, y=1.01)
    plt.tight_layout()
    _save(fig, "degeneracy_widths.png")


# ===================================================================
# Figure 6 -- Bias heatmap (Stage 6 only)
# ===================================================================
def plot_bias_heatmap():
    """Compact heatmap: params on y-axis, v_glia on x-axis, colour = bias.
    One panel for HCP, one for Novel.

    T2_s at v_glia=0.00 is masked (hatched) because the sphere compartment
    contributes no signal when v_glia=0, making T2_s fundamentally
    unidentifiable regardless of protocol.  Similarly v_glia at 0.00 is
    excluded from the colour scale since relative bias is undefined there.
    """

    params = ["v_ic", "ODI", "v_glia", "T2_t", "T2_s"]

    # Cells that are structurally unidentifiable: (param_index, vg_index)
    _UNIDENTIFIABLE = set()
    for j, vg in enumerate(GLIA_FRACS):
        if vg == 0.0:
            _UNIDENTIFIABLE.add((params.index("T2_s"), j))
            _UNIDENTIFIABLE.add((params.index("v_glia"), j))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    fig.patch.set_facecolor("white")

    for col, proto_name in enumerate(PROTOCOLS):
        ax = axes[col]
        bias_grid = np.full((len(params), len(GLIA_FRACS)), np.nan)

        for j, vg in enumerate(GLIA_FRACS):
            res = _load_chain(proto_name, "stage6", vg)
            if res is None:
                continue
            flat, truths, names, _, _ = res
            for i, p in enumerate(params):
                if (i, j) in _UNIDENTIFIABLE:
                    continue   # leave as nan
                if p not in names:
                    continue
                k = names.index(p)
                truth = float(truths[k])
                med   = float(np.percentile(flat[:, k], 50))
                scale = abs(truth) if abs(truth) > 1e-3 else 1.0
                bias_grid[i, j] = (med - truth) / scale

        # Use a shared colour scale across both panels (compute after
        # excluding unidentifiable cells)
        vmax = max(np.nanmax(np.abs(bias_grid)), 0.1)
        im = ax.imshow(bias_grid, aspect="auto", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, origin="upper")
        ax.set_xticks(range(len(GLIA_FRACS)))
        ax.set_xticklabels([f"{v:.2f}" for v in GLIA_FRACS])
        ax.set_xlabel(r"$v_{glia}$ (truth)")
        ax.set_yticks(range(len(params)))
        ax.set_yticklabels([PARAM_LATEX.get(p, p) for p in params])
        ax.set_title(PROTOCOLS[proto_name].label, fontweight="bold")
        plt.colorbar(im, ax=ax, label="Relative bias", shrink=0.8)

        # Annotate values; hatch unidentifiable cells
        for i in range(len(params)):
            for j in range(len(GLIA_FRACS)):
                if (i, j) in _UNIDENTIFIABLE:
                    ax.add_patch(plt.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1, fill=False,
                        hatch="////", edgecolor="#999", lw=0.5))
                    ax.text(j, i, "N/A", ha="center", va="center",
                            fontsize=7, color="#999", style="italic")
                else:
                    v = bias_grid[i, j]
                    if np.isfinite(v):
                        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                                fontsize=8,
                                color="white" if abs(v) > vmax * 0.6 else "black")

    fig.suptitle("Stage 6: relative bias heatmap\n"
                 "(hatched = structurally unidentifiable at $v_{glia}$=0)",
                 fontsize=11, y=1.04)
    plt.tight_layout()
    _save(fig, "bias_heatmap.png")


# ===================================================================
# Figure 7 -- Posterior width ratio (Novel / HCP)
# ===================================================================
def plot_width_ratio():
    """Bar chart of Novel_width / HCP_width for Stage 6 parameters.
    Values < 1 mean Novel is more precise."""

    params = ["v_ic", "ODI", "v_glia", "T2_t", "T2_s"]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("white")

    x_positions = []
    x_labels = []
    ratios = []
    colours = []
    pos = 0

    for vg in GLIA_FRACS:
        res_hcp   = _load_chain("hcp",   "stage6", vg)
        res_novel = _load_chain("novel", "stage6", vg)
        if res_hcp is None or res_novel is None:
            pos += len(params) + 1
            continue

        flat_h, _, names_h, _, _ = res_hcp
        flat_n, _, names_n, _, _ = res_novel

        for p in params:
            # Skip structurally unidentifiable cells
            if vg == 0.0 and p in ("T2_s", "v_glia"):
                continue
            if p not in names_h or p not in names_n:
                continue
            jh = names_h.index(p)
            jn = names_n.index(p)
            w_h = float(np.percentile(flat_h[:, jh], 84)
                        - np.percentile(flat_h[:, jh], 16))
            w_n = float(np.percentile(flat_n[:, jn], 84)
                        - np.percentile(flat_n[:, jn], 16))
            ratio = w_n / w_h if w_h > 1e-10 else np.nan

            x_positions.append(pos)
            x_labels.append(f"{PARAM_LATEX.get(p, p)}\nvg={vg:.2f}")
            ratios.append(ratio)
            colours.append(PARAM_COLOUR.get(p, "#666"))
            pos += 1
        pos += 1  # gap between v_glia groups

    ax.bar(x_positions, ratios, color=colours, edgecolor="black", lw=0.5)
    ax.axhline(1.0, color="black", lw=1.2, ls="--", alpha=0.7,
               label="Novel = HCP")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=7, rotation=30, ha="right")
    ax.set_ylabel("Posterior width ratio (Novel / HCP)")
    ax.set_title("Stage 6: precision comparison\n"
                 "< 1 means Novel is more precise, > 1 means HCP is",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    _save(fig, "width_ratio.png")


# ===================================================================
# Figure 8 -- Convergence diagnostics
# ===================================================================
def plot_convergence_diagnostics():
    """Rhat bar chart for Stage 6, all params, both protocols, at vg=0.20."""

    vg_show = 0.20
    params_s6 = STAGE_FREE_PARAMS["stage6"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig.patch.set_facecolor("white")

    for col, proto_name in enumerate(PROTOCOLS):
        ax = axes[col]
        res = _load_chain(proto_name, "stage6", vg_show)
        if res is None:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center", color="grey")
            continue

        _, _, names, rhat, tau = res
        if rhat is None:
            rhat = np.full(len(names), np.nan)
        if tau is None:
            tau = np.full(len(names), np.nan)

        x = np.arange(len(names))
        colours = ["#e74c3c" if r > 1.1 else "#2ecc71"
                    for r in rhat[:len(names)]]
        ax.bar(x, rhat[:len(names)], color=colours, edgecolor="black",
               lw=0.5)
        ax.axhline(1.0, color="black", lw=0.8, ls="-", alpha=0.5)
        ax.axhline(1.1, color="red",   lw=1.0, ls="--", alpha=0.7,
                   label="Rhat = 1.1 threshold")
        ax.set_xticks(x)
        ax.set_xticklabels([PARAM_LATEX.get(p, p) for p in names],
                           fontsize=9)
        ax.set_ylabel("Split $\\hat{R}$")
        ax.set_title(f"{PROTOCOLS[proto_name].label}   |   "
                     rf"$v_{{glia}}={vg_show:.2f}$", fontweight="bold")
        ax.legend(fontsize=8)

        # Annotate autocorrelation time
        for i, (r, t) in enumerate(zip(rhat[:len(names)], tau[:len(names)])):
            if np.isfinite(t):
                ax.text(i, r + 0.02, rf"$\tau$={t:.0f}", ha="center",
                        fontsize=7, color="grey")

    fig.suptitle("Stage 6 convergence: split-$\\hat{R}$ and autocorrelation time",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    _save(fig, "convergence_diagnostics.png")


# ===================================================================
# Figure 9 -- Classification pseudo-ROC
# ===================================================================
def plot_classification_roc():
    """Pseudo-ROC using existing Stage 6 posteriors.

    "Healthy" = v_glia=0.00 posterior of v_glia.
    "Inflamed" = v_glia=0.20 posterior of v_glia.
    Sweep a threshold on the posterior median to compute TPR / FPR.
    """

    vg_healthy  = 0.0
    vg_inflamed = 0.20

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor("white")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4, label="chance")

    for proto_name in PROTOCOLS:
        res_h = _load_chain(proto_name, "stage6", vg_healthy)
        res_i = _load_chain(proto_name, "stage6", vg_inflamed)
        if res_h is None or res_i is None:
            continue

        flat_h, _, names_h, _, _ = res_h
        flat_i, _, names_i, _, _ = res_i

        if "v_glia" not in names_h or "v_glia" not in names_i:
            continue
        jh = names_h.index("v_glia")
        ji = names_i.index("v_glia")

        samples_h = flat_h[:, jh]   # "negative" distribution
        samples_i = flat_i[:, ji]   # "positive" distribution

        thresholds = np.linspace(0, 0.5, 200)
        tpr = np.array([np.mean(samples_i >= t) for t in thresholds])
        fpr = np.array([np.mean(samples_h >= t) for t in thresholds])

        # AUC via trapezoidal rule (sort by FPR ascending)
        order = np.argsort(fpr)
        auc = np.trapz(tpr[order], fpr[order])

        ax.plot(fpr, tpr, color=PROTO_COLOUR[proto_name], lw=2,
                label=f"{PROTOCOLS[proto_name].label}  (AUC={auc:.3f})")

    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Stage 6: classification of healthy ($v_{glia}$=0.00) vs "
                 "inflamed ($v_{glia}$=0.20)\nusing posterior samples of "
                 "$v_{glia}$", fontsize=10)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    _save(fig, "classification_roc.png")


# ===================================================================
# Main
# ===================================================================
def main():
    ensure_dirs()
    print("=" * 72)
    print("1205_visualize_results.py -- publication figures")
    print("=" * 72)

    plot_signal_level()
    plot_parameter_recovery()
    plot_stage6_T2_spotlight()
    plot_stage6_corners()
    plot_degeneracy_widths()
    plot_bias_heatmap()
    plot_width_ratio()
    plot_convergence_diagnostics()
    plot_classification_roc()

    print()
    print(f"All figures written to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
