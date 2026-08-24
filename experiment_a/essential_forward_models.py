"""
1205_forward_models.py
======================
Compartment-level signal functions.

Changes from original forward_models.py
-----------------------------------------
* Fibre orientation (mu) is now a *parameter* passed in, not a global
  constant.  Every forward function accepts ``mu`` (a 3-vector) or
  ``(theta, phi)`` spherical coordinates.
* Gradient directions are per-protocol (HCP = 90, Novel = 64) and
  computed lazily via ``get_g_vectors``.
* D_GLIA = 3.0 (pulled from 1205_config).
"""

from __future__ import annotations

import importlib
import numpy as np

# -- sibling import (module name starts with a digit) -----------------------
cfg = importlib.import_module("essential_config")

S_0          = cfg.S_0
D_PAR        = cfg.D_PAR
D_GLIA       = cfg.D_GLIA
R_GLIA       = cfg.R_GLIA
DELTA        = cfg.DELTA
DELTA_PG     = cfg.DELTA_PG
T2_TISSUE_TRUE = cfg.T2_TISSUE_TRUE
T2_SPHERE_TRUE = cfg.T2_SPHERE_TRUE
SIGMA_NOISE  = cfg.SIGMA_NOISE
N_WATSON_SAMPLES = cfg.N_WATSON_SAMPLES
D_ISO        = cfg.D_ISO
T2_CSF_TRUE  = cfg.T2_CSF_TRUE
V_ISO_TRUE   = cfg.V_ISO_TRUE
fibonacci_sphere = cfg.fibonacci_sphere
Protocol = cfg.Protocol
gradient_strength_for_b = cfg.gradient_strength_for_b
angles_to_mu = cfg.angles_to_mu

# -- upstream library imports -----------------------------------------------
from essential_compartments import (
    calculate_watson_stick_signal,
    calculate_noddi_extra_signal,
    calculate_sphere_signal,
    calculate_ball_signal,
)

# ---------------------------------------------------------------------------
# Per-protocol gradient directions (cached)
# ---------------------------------------------------------------------------
_GVEC_CACHE: dict[int, np.ndarray] = {}

def get_g_vectors(n_grad: int) -> np.ndarray:
    """Return (n_grad, 3) Fibonacci-sphere gradient directions, cached."""
    if n_grad not in _GVEC_CACHE:
        _GVEC_CACHE[n_grad] = fibonacci_sphere(samples=n_grad)
    return _GVEC_CACHE[n_grad]


# Watson integration samples -- always the same count, computed once.
WATSON_SAMPLES = fibonacci_sphere(samples=N_WATSON_SAMPLES)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def odi_to_kappa(odi: float) -> float:
    odi = np.clip(odi, 0.01, 0.99)
    return 1.0 / np.tan(odi * np.pi / 2.0)


def add_rician_noise(S_clean: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    nr = rng.normal(0.0, SIGMA_NOISE, S_clean.shape)
    ni = rng.normal(0.0, SIGMA_NOISE, S_clean.shape)
    return np.sqrt((S_clean + nr) ** 2 + ni ** 2)


# ---------------------------------------------------------------------------
# Per-compartment attenuation
# ---------------------------------------------------------------------------
def _diffusion_components(protocol: Protocol, v_ic: float, odi: float,
                          mu: np.ndarray):
    """IC and EC diffusion attenuations.  Shape (N_B, n_grad)."""
    kappa = odi_to_kappa(odi)
    g_vec = get_g_vectors(protocol.n_grad)
    n_b = len(protocol.b_values)
    n_g = protocol.n_grad
    A_ic = np.empty((n_b, n_g))
    A_ec = np.empty((n_b, n_g))
    for k, b in enumerate(protocol.b_values):
        A_ic[k] = calculate_watson_stick_signal(
            b, D_PAR, mu, kappa, g_vec, WATSON_SAMPLES
        )
        A_ec[k] = calculate_noddi_extra_signal(
            b, D_PAR, v_ic, kappa, mu, g_vec
        )
    return A_ic, A_ec


def _sphere_components(protocol: Protocol):
    """Sphere attenuation.  Shape (N_B, n_grad)."""
    g_vec = get_g_vectors(protocol.n_grad)
    n_b = len(protocol.b_values)
    n_g = protocol.n_grad
    A_sp = np.empty((n_b, n_g))
    for k, b in enumerate(protocol.b_values):
        G = gradient_strength_for_b(b)
        A_sp[k] = calculate_sphere_signal(
            b, G, DELTA_PG, DELTA, R_GLIA, D_GLIA, g_vec
        )
    return A_sp

def _ball_components(protocol: Protocol):
    """Ball (CSF) attenuation.  Shape (N_B, n_grad)."""
    g_vec = get_g_vectors(protocol.n_grad)
    n_b = len(protocol.b_values)
    n_g = protocol.n_grad
    A_ball = np.empty((n_b, n_g))
    for k, b in enumerate(protocol.b_values):
        A_ball[k] = calculate_ball_signal(b, D_ISO, g_vec)
    return A_ball


# ---------------------------------------------------------------------------
# Stage forward models
# ---------------------------------------------------------------------------
def forward_noddi(protocol, v_ic, odi, mu, v_iso=0.0):
    A_ic, A_ec = _diffusion_components(protocol, v_ic, odi, mu)
    A_ball = _ball_components(protocol)
    f_tissue = 1.0 - v_iso
    return S_0 * (f_tissue * (v_ic * A_ic + (1.0 - v_ic) * A_ec)
                  + v_iso * A_ball)


def forward_noddi_glia(protocol, v_ic, odi, v_glia, mu, v_iso=0.0):
    A_ic, A_ec = _diffusion_components(protocol, v_ic, odi, mu)
    A_sp = _sphere_components(protocol)
    A_ball = _ball_components(protocol)
    f_tissue = 1.0 - v_iso
    return S_0 * (
        f_tissue * ((1.0 - v_glia) * (v_ic * A_ic + (1.0 - v_ic) * A_ec)
                     + v_glia * A_sp)
        + v_iso * A_ball
    )

def forward_noddi_glia_singleTE_norm(protocol, v_ic, odi, v_glia, mu,
                                      v_iso=0.0, te=None):
    if te is None:
        te = protocol.te_single
    A_ic, A_ec = _diffusion_components(protocol, v_ic, odi, mu)
    A_sp = _sphere_components(protocol)
    A_ball = _ball_components(protocol)
    f_tissue = 1.0 - v_iso

    w_t = np.exp(-te / T2_TISSUE_TRUE)
    w_s = np.exp(-te / T2_SPHERE_TRUE)
    w_c = np.exp(-te / T2_CSF_TRUE)

    S_dw = (f_tissue * ((1.0 - v_glia) * (v_ic * A_ic + (1.0 - v_ic) * A_ec) * w_t
                         + v_glia * A_sp * w_s)
            + v_iso * A_ball * w_c)
    S_b0 = f_tissue * ((1.0 - v_glia) * w_t + v_glia * w_s) + v_iso * w_c
    return S_dw / S_b0


def forward_noddi_glia_multiTE(protocol, v_ic, odi, v_glia, mu,
                                v_iso=0.0,
                                T2_t=T2_TISSUE_TRUE, T2_s=T2_SPHERE_TRUE):
    A_ic, A_ec = _diffusion_components(protocol, v_ic, odi, mu)
    A_sp = _sphere_components(protocol)
    A_ball = _ball_components(protocol)
    f_tissue = 1.0 - v_iso

    n_te = len(protocol.te_multi)
    n_b  = len(protocol.b_values)
    n_g  = protocol.n_grad
    S = np.empty((n_te, n_b, n_g))
    for i, te in enumerate(protocol.te_multi):
        w_t = np.exp(-te / T2_t)
        w_s = np.exp(-te / T2_s)
        w_c = np.exp(-te / T2_CSF_TRUE)
        S[i] = S_0 * (
            f_tissue * ((1.0 - v_glia) * (v_ic * A_ic + (1.0 - v_ic) * A_ec) * w_t
                         + v_glia * A_sp * w_s)
            + v_iso * A_ball * w_c
        )
    return S


# ---------------------------------------------------------------------------
# Truth-signal dispatcher
# ---------------------------------------------------------------------------
def truth_signal(stage, protocol, v_glia, mu):
    v_ic  = cfg.V_IC_TRUE
    odi   = cfg.ODI_TRUE
    v_iso = cfg.V_ISO_TRUE

    if stage == "stage1":
        return forward_noddi(protocol, v_ic, odi, mu, v_iso=v_iso)
    if stage in ("stage2", "stage3"):
        return forward_noddi_glia(protocol, v_ic, odi, v_glia, mu, v_iso=v_iso)
    if stage == "stage4":
        return forward_noddi_glia_singleTE_norm(
            protocol, v_ic, odi, v_glia, mu, v_iso=v_iso, te=protocol.te_single)
    if stage in ("stage5", "stage6"):
        return forward_noddi_glia_multiTE(
            protocol, v_ic, odi, v_glia, mu, v_iso=v_iso,
            T2_t=T2_TISSUE_TRUE, T2_s=T2_SPHERE_TRUE)
    raise ValueError(f"Unknown stage: {stage}")

if __name__ == "__main__":
    main()
