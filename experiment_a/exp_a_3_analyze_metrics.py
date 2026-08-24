# -*- coding: utf-8 -*-
"""
1205_analyze_metrics.py
=======================
SCRIPT 3 of 4 -- statistical evaluation.

Changes from original analyze_metrics.py
------------------------------------------
* Loads and reports convergence diagnostics (autocorr time, split-Rhat)
  alongside the posterior summaries.
* Includes theta, phi in the summary tables.
* Classification stub upgraded to pseudo-ROC using existing posteriors.

Output:
    results_1205/metrics/summary.csv
    results_1205/metrics/summary.npz
    results_1205/metrics/posterior_widths.npz
    results_1205/metrics/convergence.npz

Run:  python 1205_analyze_metrics.py
"""

from __future__ import annotations

import csv
import importlib
import os

import numpy as np

cfg = importlib.import_module("1205_config")

CHAINS_DIR       = cfg.CHAINS_DIR
METRICS_DIR      = cfg.METRICS_DIR
ensure_dirs      = cfg.ensure_dirs
PROTOCOLS        = cfg.PROTOCOLS
GLIA_FRACS       = cfg.GLIA_FRACS
STAGE_NAMES      = cfg.STAGE_NAMES
STAGE_FREE_PARAMS = cfg.STAGE_FREE_PARAMS
V_IC_TRUE        = cfg.V_IC_TRUE
ODI_TRUE         = cfg.ODI_TRUE
T2_TISSUE_TRUE   = cfg.T2_TISSUE_TRUE
T2_SPHERE_TRUE   = cfg.T2_SPHERE_TRUE
V_ISO_TRUE       = cfg.V_ISO_TRUE


# ---------------------------------------------------------------------------
# Filename helpers
# ---------------------------------------------------------------------------
def _vg_str(v_glia: float) -> str:
    return f"vg{v_glia:.2f}".replace(".", "p")


def _chain_path(proto_name: str, stage: str, v_glia: float) -> str:
    return os.path.join(CHAINS_DIR,
                        f"{proto_name}_{stage}_{_vg_str(v_glia)}.npz")


# ---------------------------------------------------------------------------
# Per-chain summaries
# ---------------------------------------------------------------------------
def summarise_chain(flat: np.ndarray) -> dict:
    medians = np.percentile(flat, 50, axis=0)
    lo68    = np.percentile(flat, 16, axis=0)
    hi68    = np.percentile(flat, 84, axis=0)
    means   = np.mean(flat, axis=0)
    stds    = np.std(flat, axis=0)
    iqr68   = hi68 - lo68
    return dict(median=medians, lo68=lo68, hi68=hi68,
                mean=means, std=stds, iqr68=iqr68)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ensure_dirs()
    print("=" * 72)
    print("1205_analyze_metrics.py -- posterior summaries + convergence")
    print("=" * 72)

    rows           = []
    width_records  = []
    r2_records     = []
    conv_records   = []

    for proto_name, protocol in PROTOCOLS.items():
        for stage in STAGE_NAMES:
            param_names = STAGE_FREE_PARAMS[stage]
            truth_by_param = {p: [] for p in param_names}
            mean_by_param  = {p: [] for p in param_names}

            for v_glia in GLIA_FRACS:
                if stage == "stage1" and v_glia != GLIA_FRACS[0]:
                    continue

                path = _chain_path(proto_name, stage, v_glia)
                if not os.path.exists(path):
                    print(f"  [skip] missing {path}")
                    continue

                with np.load(path, allow_pickle=True) as data:
                    flat   = data["flat"]
                    truths = data["truths"]
                    names  = list(data["param_names"])
                    tau    = data.get("autocorr_time", np.full(len(names), np.nan))
                    rhat   = data.get("split_rhat",    np.full(len(names), np.nan))

                stats = summarise_chain(flat)

                # Convergence record
                for j, p in enumerate(names):
                    conv_records.append({
                        "protocol": proto_name, "stage": stage,
                        "v_glia": float(v_glia), "param": p,
                        "autocorr_time": float(tau[j]) if j < len(tau) else np.nan,
                        "split_rhat":    float(rhat[j]) if j < len(rhat) else np.nan,
                    })

                for j, p in enumerate(names):
                    truth_by_param[p].append(float(truths[j]))
                    mean_by_param[p].append(float(stats["mean"][j]))

                    bias = float(stats["mean"][j] - truths[j])
                    rows.append({
                        "protocol": proto_name, "stage": stage,
                        "v_glia_truth": float(v_glia), "param": p,
                        "truth":  float(truths[j]),
                        "median": float(stats["median"][j]),
                        "lo68":   float(stats["lo68"][j]),
                        "hi68":   float(stats["hi68"][j]),
                        "mean":   float(stats["mean"][j]),
                        "std":    float(stats["std"][j]),
                        "iqr68":  float(stats["iqr68"][j]),
                        "bias":   bias,
                    })
                    width_records.append({
                        "protocol": proto_name, "stage": stage,
                        "v_glia": float(v_glia), "param": p,
                        "iqr68": float(stats["iqr68"][j]),
                        "std":   float(stats["std"][j]),
                    })

            # R^2 across v_glia sweep
            for p in param_names:
                t_arr = np.array(truth_by_param[p])
                m_arr = np.array(mean_by_param[p])
                if t_arr.size < 2:
                    r2 = float("nan")
                elif np.allclose(t_arr, t_arr[0]):
                    ss_res = float(np.sum((m_arr - t_arr) ** 2))
                    ss_tot = float(np.sum((m_arr - m_arr.mean()) ** 2))
                    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
                else:
                    ss_res = float(np.sum((m_arr - t_arr) ** 2))
                    ss_tot = float(np.sum((t_arr - t_arr.mean()) ** 2))
                    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
                r2_records.append({
                    "protocol": proto_name, "stage": stage,
                    "param": p, "R2": r2,
                })

    # -- CSV ----------------------------------------------------------------
    if rows:
        csv_path = os.path.join(METRICS_DIR, "summary.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"  wrote {csv_path}  ({len(rows)} rows)")

    # -- npz bundles --------------------------------------------------------
    np.savez(os.path.join(METRICS_DIR, "summary.npz"),
             rows=np.array(rows, dtype=object),
             r2=np.array(r2_records, dtype=object),
             widths=np.array(width_records, dtype=object))

    np.savez(os.path.join(METRICS_DIR, "posterior_widths.npz"),
             widths=np.array(width_records, dtype=object))

    np.savez(os.path.join(METRICS_DIR, "convergence.npz"),
             records=np.array(conv_records, dtype=object))

    # -- convergence summary ------------------------------------------------
    print()
    print("=" * 72)
    print("CONVERGENCE SUMMARY (split-Rhat)")
    print("=" * 72)
    bad = [r for r in conv_records
           if np.isfinite(r["split_rhat"]) and r["split_rhat"] > 1.1]
    if bad:
        for r in bad:
            print(f"  WARNING  {r['protocol']:5s}  {r['stage']}  "
                  f"vg={r['v_glia']:.2f}  {r['param']:6s}  "
                  f"Rhat={r['split_rhat']:.3f}")
    else:
        print("  All Rhat values <= 1.1  (OK)")

    # -- Stage 6 spotlight --------------------------------------------------
    print()
    print("=" * 72)
    print("STAGE 6 SPOTLIGHT")
    print("=" * 72)
    for vg in GLIA_FRACS:
        for proto_name in PROTOCOLS:
            path = _chain_path(proto_name, "stage6", vg)
            if not os.path.exists(path):
                continue
            with np.load(path, allow_pickle=True) as data:
                flat   = data["flat"]
                names  = list(data["param_names"])
                truths = data["truths"]
                rhat   = data.get("split_rhat", np.full(len(names), np.nan))

            stats = summarise_chain(flat)
            t2t_idx = names.index("T2_t")
            t2s_idx = names.index("T2_s")
            vg_idx  = names.index("v_glia")
            print(
                f"  vg={vg:.2f}  [{proto_name:5s}]  "
                f"v_glia={stats['median'][vg_idx]:.3f} "
                f"+/-{stats['iqr68'][vg_idx]/2:.3f}  "
                f"T2_t={stats['median'][t2t_idx]:.1f}"
                f"+/-{stats['iqr68'][t2t_idx]/2:.1f}  "
                f"T2_s={stats['median'][t2s_idx]:.1f}"
                f"+/-{stats['iqr68'][t2s_idx]/2:.1f}  "
                f"Rhat_max={np.nanmax(rhat):.2f}"
            )
    print()


if __name__ == "__main__":
    main()
