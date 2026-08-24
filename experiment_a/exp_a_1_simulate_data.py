# -*- coding: utf-8 -*-
"""
1205_simulate_data.py
=====================
SCRIPT 1 of 4 -- forward model / signal generation.

Generates synthetic Rician-noisy diffusion(-T2) signals for both protocols
across all six stages and the GLIA_FRACS sweep.

Output: results_1205/signals/

Run:  python 1205_simulate_data.py
"""

from __future__ import annotations

import importlib
import os

import numpy as np

cfg = importlib.import_module("essential_config")
fwd = importlib.import_module("essential_forward_models")

SIGNALS_DIR  = cfg.SIGNALS_DIR
PROTOCOLS    = cfg.PROTOCOLS
GLIA_FRACS   = cfg.GLIA_FRACS
STAGE_NAMES  = cfg.STAGE_NAMES
V_IC_TRUE    = cfg.V_IC_TRUE
ODI_TRUE     = cfg.ODI_TRUE
T2_TISSUE_TRUE = cfg.T2_TISSUE_TRUE
T2_SPHERE_TRUE = cfg.T2_SPHERE_TRUE
SNR          = cfg.SNR
SEED         = cfg.SEED
MU_TRUE      = cfg.MU_TRUE
THETA_TRUE   = cfg.THETA_TRUE
PHI_TRUE     = cfg.PHI_TRUE
ensure_dirs  = cfg.ensure_dirs
fibonacci_sphere = cfg.fibonacci_sphere


# ---------------------------------------------------------------------------
# Filename helpers
# ---------------------------------------------------------------------------
def _vg_str(v_glia: float) -> str:
    return f"vg{v_glia:.2f}".replace(".", "p")


def _signal_path(protocol_name: str, stage: str, v_glia: float,
                 kind: str) -> str:
    fname = f"{protocol_name}_{stage}_{_vg_str(v_glia)}_{kind}.npy"
    return os.path.join(SIGNALS_DIR, fname)


# ---------------------------------------------------------------------------
# Sanity banner
# ---------------------------------------------------------------------------
def _print_banner():
    print("=" * 72)
    print("1205_simulate_data.py -- forward model")
    print("=" * 72)
    print(f"  SNR              : {SNR}")
    print(f"  v_ic  (true)     : {V_IC_TRUE}")
    print(f"  ODI   (true)     : {ODI_TRUE}")
    print(f"  theta (true)     : {THETA_TRUE:.4f} rad")
    print(f"  phi   (true)     : {PHI_TRUE:.4f} rad")
    print(f"  mu    (true)     : [{MU_TRUE[0]:.4f}, {MU_TRUE[1]:.4f}, {MU_TRUE[2]:.4f}]")
    print(f"  T2_tissue (true) : {T2_TISSUE_TRUE} ms")
    print(f"  T2_sphere (true) : {T2_SPHERE_TRUE} ms")
    print(f"  D_GLIA           : {cfg.D_GLIA} um^2/ms")
    print(f"  GLIA_FRACS       : {GLIA_FRACS.tolist()}")
    print()
    for name, proto in PROTOCOLS.items():
        n_te = len(proto.te_multi)
        print(f"  [{proto.label}]")
        print(f"    b-values      : {proto.b_values.tolist()} ms/um^2  "
              f"({(proto.b_values * 1000).astype(int).tolist()} s/mm^2)")
        print(f"    n_grad        : {proto.n_grad} directions / shell")
        print(f"    TE single     : {proto.te_single} ms")
        print(f"    TE multi      : {proto.te_multi.tolist()}  (N_TE={n_te})")
        if name == "hcp":
            assert n_te == 1, "HCP must use exactly ONE echo time."
            print(f"    SANITY: HCP uses {n_te} TE -- T2 unresolvable.")
        elif name == "novel":
            assert n_te == 3, "Novel must use exactly THREE echo times."
            print(f"    SANITY: Novel uses {n_te} TEs -- T2 resolvable.")
    print("=" * 72)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ensure_dirs()
    _print_banner()

    rng_master = np.random.default_rng(SEED)
    n_signals = 0

    for proto_name, protocol in PROTOCOLS.items():
        print(f"[{protocol.label}] generating signals ...")
        for stage in STAGE_NAMES:
            for v_glia in GLIA_FRACS:
                if stage == "stage1" and v_glia != GLIA_FRACS[0]:
                    continue

                S_clean = fwd.truth_signal(stage, protocol, float(v_glia),
                                           MU_TRUE)

                rng = np.random.default_rng(
                    rng_master.integers(0, 2**31 - 1)
                )
                S_noisy = fwd.add_rician_noise(S_clean, rng)

                np.save(_signal_path(proto_name, stage, v_glia, "clean"),
                        S_clean)
                np.save(_signal_path(proto_name, stage, v_glia, "noisy"),
                        S_noisy)
                n_signals += 2

                print(f"    {stage}  v_glia={v_glia:.2f}  "
                      f"shape={S_clean.shape}  "
                      f"clean range=[{S_clean.min():.4f}, {S_clean.max():.4f}]")
        print()

    # Ground-truth bundle
    np.savez(
        os.path.join(SIGNALS_DIR, "truths.npz"),
        v_ic_true=V_IC_TRUE, odi_true=ODI_TRUE,
        theta_true=THETA_TRUE, phi_true=PHI_TRUE,
        mu_true=MU_TRUE,
        T2_tissue_true=T2_TISSUE_TRUE, T2_sphere_true=T2_SPHERE_TRUE,
        D_GLIA=cfg.D_GLIA, snr=SNR,
        glia_fracs=GLIA_FRACS,
        stage_names=np.array(STAGE_NAMES),
        protocol_names=np.array(list(PROTOCOLS.keys())),
        V_iso_true=cfg.V_ISO_TRUE,
        D_ISO=cfg.D_ISO,
        T2_CSF_true=cfg.T2_CSF_TRUE,
    )

    print(f"Done. Wrote {n_signals} signal files + truths.npz "
          f"to {SIGNALS_DIR}/")


if __name__ == "__main__":
    main()

