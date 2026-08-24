"""
1205_config.py
==============
Configuration for the 12-May revised pipeline.

Changes from original config.py
--------------------------------
* D_GLIA raised to 3.0 um^2/ms (was 1.0).
* Fibre orientation (theta, phi) added as free parameters in every stage
  so that the comparison is realistic -- in practice NODDI always estimates
  orientation.  HCP's 90 gradient directions per shell (its traditional
  count) make it strong at orientation; Novel has 64.
* MCMC: wider walker scatter for T2, starting guesses moved away from
  truth, much longer chains for Stage 6 to expose degeneracy.
* Results written to results_1205/ to keep the old run intact.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrapping -- bbdb_compartments.py and bbdb_sh_utils.py live one
# directory up.
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

# ---------------------------------------------------------------------------
# I/O paths
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(HERE, "try_4_results")
SIGNALS_DIR = os.path.join(RESULTS_DIR, "signals")
CHAINS_DIR  = os.path.join(RESULTS_DIR, "chains")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")


def ensure_dirs():
    for d in (SIGNALS_DIR, CHAINS_DIR, METRICS_DIR, FIGURES_DIR):
        os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
S_0    = 1.0        # baseline signal (no diffusion, TE = 0)
D_PAR  = 1.7        # um^2/ms  intra-axonal parallel diffusivity
D_GLIA = 3.0        # um^2/ms  intra-sphere diffusivity  ** CHANGED (was 1.0) **
R_GLIA = 5.0        # um       glial soma radius (fixed in stages 1-6)
D_ISO  = 3.0  

# PGSE timings
DELTA    = 43.1      # ms  big Delta
DELTA_PG = 10.6      # ms  little delta

# T2 truths (tied IC/EC; separate sphere)
T2_TISSUE_TRUE = 100.0   # ms
T2_SPHERE_TRUE = 30.0    # ms
T2_CSF_TRUE = 2000.0     # ms  (CSF T2 at 3T; fixed, not estimated)
V_ISO_TRUE  = 0.16       # CSF partial volume (HCP WM median)

# Gyromagnetic ratio  rad / (ms * T)
_GAMMA_MS = 2.6752218744e8 * 1e-3

# ---------------------------------------------------------------------------
# Fibre orientation truth
# ---------------------------------------------------------------------------
# theta slightly off-pole so phi is identifiable by the MCMC.
THETA_TRUE = 0.30        # rad  (~17 deg from z-axis)
PHI_TRUE   = 0.80        # rad

MU_TRUE = np.array([
    np.sin(THETA_TRUE) * np.cos(PHI_TRUE),
    np.sin(THETA_TRUE) * np.sin(PHI_TRUE),
    np.cos(THETA_TRUE),
])


def angles_to_mu(theta: float, phi: float) -> np.ndarray:
    """Spherical angles -> unit vector."""
    return np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])

def fibonacci_sphere(samples=1000): 
    """Generates evenly spaced points on a sphere."""
    phi = np.pi * (3. - np.sqrt(5.))
    indices = np.arange(samples, dtype=float) + 0.5
    y = 1.0 - (indices / float(samples - 1)) * 2.0
    y = np.clip(y, -1.0, 1.0)
    radius = np.sqrt(1.0 - y * y)
    theta = phi * indices
    x = np.cos(theta) * radius
    z = np.sin(theta) * radius
    return np.array([x, y, z]).T # (N, 3) shape
# ---------------------------------------------------------------------------
# Protocol definitions
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Protocol:
    name: str
    label: str
    b_values: np.ndarray        # ms/um^2
    te_single: float            # ms
    te_multi: np.ndarray        # ms
    has_multi_te: bool
    n_grad: int                 # gradient directions per shell


HCP = Protocol(
    name="hcp",
    label="HCP",
    b_values=np.array([1.0, 2.0, 3.0]),
    te_single=62.0,
    te_multi=np.array([62.0]),
    has_multi_te=False,
    n_grad=64,                  # matched to Novel for a clean comparison
)

NOVEL = Protocol(
    name="novel",
    label="Novel multi-TE",
    b_values=np.array([1.0, 2.0]),
    te_single=62.0,
    te_multi=np.array([62.0, 92.0, 130.0]),
    has_multi_te=True,
    n_grad=64,
)

PROTOCOLS = {"hcp": HCP, "novel": NOVEL}


def gradient_strength_for_b(b_value: float) -> float:
    """PGSE gradient strength G [mT/m] for given b [ms/um^2]."""
    G_T_per_um = np.sqrt(
        b_value / (_GAMMA_MS ** 2 * DELTA_PG ** 2 * (DELTA - DELTA_PG / 3.0))
    )
    return G_T_per_um * 1e3 * 1e6


# ---------------------------------------------------------------------------
# Tissue preset
# ---------------------------------------------------------------------------
TISSUE_LABEL = "white matter"
V_IC_TRUE = 0.73
ODI_TRUE  = 0.25

# ---------------------------------------------------------------------------
# Glia fraction sweep
# ---------------------------------------------------------------------------
GLIA_FRACS = np.array([0.0, 0.10, 0.20, 0.30])

# ---------------------------------------------------------------------------
# Noise model
# ---------------------------------------------------------------------------
SNR = 20.0
SIGMA_NOISE = S_0 / SNR

# ---------------------------------------------------------------------------
# Watson integral samples  (protocol-independent)
# ---------------------------------------------------------------------------
N_WATSON_SAMPLES = 500

# ---------------------------------------------------------------------------
# MCMC (emcee) settings
# ---------------------------------------------------------------------------
N_WALKERS = 56

# Chain depth -- same for all stages so every posterior gets equal
# exploration budget.  DEMove + DESnookerMove need the length to
# traverse correlated ridges (especially HCP Stage 6).
N_STEPS = 40000
N_BURN  = 10000
N_THIN  = 10

SEED = 42

# ---------------------------------------------------------------------------
# Stage book-keeping
# ---------------------------------------------------------------------------
STAGE_NAMES = [
    "stage1",   # std NODDI truth -> std NODDI fit (sanity)
    "stage2",   # NODDI+glia (same T2) truth -> std NODDI fit (mis-spec)
    "stage3",   # NODDI+glia (same T2) truth -> NODDI+glia fit
    "stage4",   # NODDI+glia (diff T2, single TE) truth -> std NODDI fit (mis-spec)
    "stage5",   # NODDI+glia (diff T2) truth -> NODDI+glia fit, T2 fixed
    "stage6",   # NODDI+glia (diff T2) truth -> full free-T2 fit
]

# Every stage now includes theta, phi (fibre orientation).
STAGE_FREE_PARAMS = {
    "stage1": ["v_ic", "ODI", "v_iso", "theta", "phi"],
    "stage2": ["v_ic", "ODI", "v_iso", "theta", "phi"],
    "stage3": ["v_ic", "ODI", "v_glia", "v_iso", "theta", "phi"],
    "stage4": ["v_ic", "ODI", "v_iso", "theta", "phi"],
    "stage5": ["v_ic", "ODI", "v_glia", "v_iso", "theta", "phi"],
    "stage6": ["v_ic", "ODI", "v_glia", "v_iso", "T2_t", "T2_s", "theta", "phi"],
}

PARAM_LATEX = {
    "v_ic":   r"$v_{ic}$",
    "ODI":    r"$\mathrm{ODI}$",
    "v_glia": r"$v_{glia}$",
    "T2_t":   r"$T_{2,t}$ [ms]",
    "T2_s":   r"$T_{2,s}$ [ms]",
    "theta":  r"$\theta$ [rad]",
    "phi":    r"$\phi$ [rad]",
    "v_iso":  r"$v_{iso}$",
}


def truth_for_stage(stage: str, v_glia: float) -> dict:
    base = {
        "v_ic": V_IC_TRUE, "ODI": ODI_TRUE, "v_glia": v_glia,
        "v_iso": V_ISO_TRUE,
        "T2_t": T2_TISSUE_TRUE, "T2_s": T2_SPHERE_TRUE,
        "theta": THETA_TRUE, "phi": PHI_TRUE,
    }
    return {p: base[p] for p in STAGE_FREE_PARAMS[stage]}

