
"""
1205_solver_engine.py
=====================
SCRIPT 2 of 4 -- MCMC inverse model.

Changes from original solver_engine.py
----------------------------------------
* Fibre orientation (theta, phi) is free in every stage.
* Stage 6 T2 starting guesses moved *away* from truth (T2_t=200, T2_s=100)
  to avoid biasing HCP toward the true values.
* Walker scatter for T2 widened to 40 ms (was 1 ms).
* Stage 6 runs 5000 steps with 1500 burn-in (was 1200 / 250).
* Convergence diagnostics (integrated autocorrelation time) saved per chain.

Run:
    python 1205_solver_engine.py
    python 1205_solver_engine.py --quick
"""

from __future__ import annotations

import argparse
import importlib
import os
import time

import emcee
import numpy as np
from scipy.optimize import minimize

cfg = importlib.import_module("essential_config")
fwd = importlib.import_module("essential_forward_models")

CHAINS_DIR       = cfg.CHAINS_DIR
SIGNALS_DIR      = cfg.SIGNALS_DIR
ensure_dirs      = cfg.ensure_dirs
PROTOCOLS        = cfg.PROTOCOLS
GLIA_FRACS       = cfg.GLIA_FRACS
STAGE_NAMES      = cfg.STAGE_NAMES
STAGE_FREE_PARAMS = cfg.STAGE_FREE_PARAMS
V_IC_TRUE        = cfg.V_IC_TRUE
ODI_TRUE         = cfg.ODI_TRUE
THETA_TRUE       = cfg.THETA_TRUE
PHI_TRUE         = cfg.PHI_TRUE
MU_TRUE          = cfg.MU_TRUE
T2_TISSUE_TRUE   = cfg.T2_TISSUE_TRUE
T2_SPHERE_TRUE   = cfg.T2_SPHERE_TRUE
SIGMA_NOISE      = cfg.SIGMA_NOISE
N_WALKERS        = cfg.N_WALKERS
N_STEPS          = cfg.N_STEPS
N_BURN           = cfg.N_BURN
N_THIN           = cfg.N_THIN
SEED             = cfg.SEED
angles_to_mu     = cfg.angles_to_mu
V_ISO_TRUE = cfg.V_ISO_TRUE


# ---------------------------------------------------------------------------
# Filename helpers
# ---------------------------------------------------------------------------
def _vg_str(v_glia: float) -> str:
    return f"vg{v_glia:.2f}".replace(".", "p")

def _signal_path(proto_name: str, stage: str, v_glia: float,
                 kind: str) -> str:
    return os.path.join(
        SIGNALS_DIR,
        f"{proto_name}_{stage}_{_vg_str(v_glia)}_{kind}.npy",
    )

def _chain_path(proto_name: str, stage: str, v_glia: float) -> str:
    return os.path.join(
        CHAINS_DIR,
        f"{proto_name}_{stage}_{_vg_str(v_glia)}.npz",
    )


# ---------------------------------------------------------------------------
# Gaussian log-likelihood (same sigma as the noise generator)
# ---------------------------------------------------------------------------
def _gaussian_loglike(resid: np.ndarray, sigma: float) -> float:
    n = resid.size
    return (-0.5 * n * np.log(2 * np.pi * sigma ** 2)
            - 0.5 * np.sum(resid ** 2) / sigma ** 2)


# ---------------------------------------------------------------------------
# Stage-specific log-posterior functions
# ---------------------------------------------------------------------------
def log_prob_stage_std_noddi(theta_vec, S_obs, sigma, protocol):
    v_ic, odi, v_iso, theta, phi = theta_vec
    if not (0.0 < v_ic < 1.0 and 0.01 < odi < 0.99
            and 0.0 <= v_iso < 0.5
            and 0.0 <= theta <= np.pi/2 and 0.0 <= phi <= 2*np.pi):
        return -np.inf
    mu = angles_to_mu(theta, phi)
    pred = fwd.forward_noddi(protocol, v_ic, odi, mu, v_iso=v_iso).ravel()
    return _gaussian_loglike(S_obs - pred, sigma)

def log_prob_stage_glia_sameT2(theta_vec, S_obs, sigma, protocol):
    """Stage 3 fit: NODDI+glia+CSF, shared T2 (6 free)."""
    v_ic, odi, v_glia, v_iso, theta, phi = theta_vec
    if not (0.0 < v_ic < 1.0 and 0.01 < odi < 0.99
            and 0.0 <= v_glia < 0.5 and 0.0 <= v_iso < 0.5
            and 0.0 <= theta <= np.pi/2 and 0.0 <= phi <= 2*np.pi):
        return -np.inf
    mu = angles_to_mu(theta, phi)
    pred = fwd.forward_noddi_glia(protocol, v_ic, odi, v_glia, mu, v_iso=v_iso).ravel()
    return _gaussian_loglike(S_obs - pred, sigma)

def log_prob_stage_glia_T2fixed(theta_vec, S_obs, sigma, protocol):
    v_ic, odi, v_glia, v_iso, theta, phi = theta_vec
    if not (0.0 < v_ic < 1.0 and 0.01 < odi < 0.99
            and 0.0 <= v_glia < 0.5 and 0.0 <= v_iso < 0.5
            and 0.0 <= theta <= np.pi/2 and 0.0 <= phi <= 2*np.pi):
        return -np.inf
    mu = angles_to_mu(theta, phi)
    pred = fwd.forward_noddi_glia_multiTE(
        protocol, v_ic, odi, v_glia, mu, v_iso=v_iso,
        T2_t=T2_TISSUE_TRUE, T2_s=T2_SPHERE_TRUE).ravel()
    return _gaussian_loglike(S_obs - pred, sigma)


def log_prob_stage_glia_T2free(theta_vec, S_obs, sigma, protocol):
    v_ic, odi, v_glia, v_iso, T2_t, T2_s, theta, phi = theta_vec
    if not (0.0 < v_ic < 1.0 and 0.01 < odi < 0.99
            and 0.0 <= v_glia < 0.5 and 0.0 <= v_iso < 0.5
            and 10.0 < T2_t < 500.0 and 5.0 < T2_s < 500.0
            and 0.0 <= theta <= np.pi/2 and 0.0 <= phi <= 2*np.pi):
        return -np.inf
    mu = angles_to_mu(theta, phi)
    pred = fwd.forward_noddi_glia_multiTE(
        protocol, v_ic, odi, v_glia, mu, v_iso=v_iso,
        T2_t=T2_t, T2_s=T2_s).ravel()
    return _gaussian_loglike(S_obs - pred, sigma)


# ---------------------------------------------------------------------------
# Stage configuration table
#
# KEY FIX: T2 starting guesses are far from truth (200, 100 vs truth 100, 30)
# and walker scatter is 40 ms (was 1 ms).  This prevents the MCMC from
# staying near the initialization and faking good HCP T2 recovery.
# ---------------------------------------------------------------------------
STAGE_CONFIG = {
    "stage1": dict(
        log_prob=log_prob_stage_std_noddi,
        # v_ic, ODI, v_iso, theta, phi
        p0_center=np.array([0.5, 0.4, 0.1, 0.2, 1.0]),
        p0_scatter=np.array([0.05, 0.05, 0.05, 0.05, 0.1]),
        bounds=[(0.01, 0.99), (0.01, 0.99), (0.0, 0.49),
                (0.0, np.pi/2), (0.0, 2*np.pi)],
    ),
    "stage2": dict(
        log_prob=log_prob_stage_std_noddi,
        p0_center=np.array([0.5, 0.4, 0.1, 0.2, 1.0]),
        p0_scatter=np.array([0.05, 0.05, 0.05, 0.05, 0.1]),
        bounds=[(0.01, 0.99), (0.01, 0.99), (0.0, 0.49),
                (0.0, np.pi/2), (0.0, 2*np.pi)],
    ),
    "stage3": dict(
        log_prob=log_prob_stage_glia_sameT2,
        # v_ic, ODI, v_glia, v_iso, theta, phi
        p0_center=np.array([0.5, 0.4, 0.1, 0.1, 0.2, 1.0]),
        p0_scatter=np.array([0.05, 0.05, 0.02, 0.05, 0.05, 0.1]),
        bounds=[(0.01, 0.99), (0.01, 0.99), (0.0, 0.49), (0.0, 0.49),
                (0.0, np.pi/2), (0.0, 2*np.pi)],
    ),
    "stage4": dict(
        log_prob=log_prob_stage_std_noddi,
        p0_center=np.array([0.5, 0.4, 0.1, 0.2, 1.0]),
        p0_scatter=np.array([0.05, 0.05, 0.05, 0.05, 0.1]),
        bounds=[(0.01, 0.99), (0.01, 0.99), (0.0, 0.49),
                (0.0, np.pi/2), (0.0, 2*np.pi)],
    ),
    "stage5": dict(
        log_prob=log_prob_stage_glia_T2fixed,
        # v_ic, ODI, v_glia, v_iso, theta, phi
        p0_center=np.array([0.5, 0.4, 0.1, 0.1, 0.2, 1.0]),
        p0_scatter=np.array([0.05, 0.05, 0.02, 0.05, 0.05, 0.1]),
        bounds=[(0.01, 0.99), (0.01, 0.99), (0.0, 0.49), (0.0, 0.49),
                (0.0, np.pi/2), (0.0, 2*np.pi)],
    ),
    "stage6": dict(
        log_prob=log_prob_stage_glia_T2free,
        # v_ic, ODI, v_glia, v_iso, T2_t, T2_s, theta, phi
        p0_center=np.array([0.5, 0.4, 0.1, 0.1, 200.0, 100.0, 0.2, 1.0]),
        p0_scatter=np.array([0.05, 0.05, 0.02, 0.05, 40.0, 40.0, 0.05, 0.1]),
        bounds=[(0.01, 0.99), (0.01, 0.99), (0.0, 0.49), (0.0, 0.49),
                (10.0, 500.0), (5.0, 500.0),
                (0.0, np.pi/2), (0.0, 2*np.pi)],
    ),
}


# ---------------------------------------------------------------------------
# Convergence diagnostics
# ---------------------------------------------------------------------------
def compute_split_rhat(chain_3d: np.ndarray) -> np.ndarray:
    """
    Split-Rhat from a (n_walkers, n_steps, n_dim) chain.
    Splits each walker chain in half, treats halves as independent chains.
    Rhat ~ 1 means convergence; > 1.1 is a warning.
    """
    n_walkers, n_steps, n_dim = chain_3d.shape
    half = n_steps // 2
    # Stack first-half and second-half of each walker as 2*n_walkers chains
    splits = np.concatenate([chain_3d[:, :half, :],
                              chain_3d[:, half:2*half, :]], axis=0)
    # splits shape: (2*n_walkers, half, n_dim)
    m = splits.shape[0]
    n = splits.shape[1]

    chain_means = splits.mean(axis=1)          # (m, n_dim)
    chain_vars  = splits.var(axis=1, ddof=1)   # (m, n_dim)

    grand_mean = chain_means.mean(axis=0)      # (n_dim,)
    B = n * chain_means.var(axis=0, ddof=1)    # between-chain var
    W = chain_vars.mean(axis=0)                # within-chain var

    var_hat = (1 - 1.0/n) * W + B / n
    rhat = np.sqrt(var_hat / np.where(W > 0, W, 1e-10))
    return rhat


# ---------------------------------------------------------------------------
# Generic MCMC runner
# ---------------------------------------------------------------------------
def run_mcmc_for(stage: str, protocol, S_obs_flat: np.ndarray, seed: int,
                 n_walkers: int, n_steps: int, n_burn: int, n_thin: int):
    """
    Run a full MCMC.  Returns (flat_chain, flat_logprob, map_estimate,
    autocorr_time, split_rhat).
    """
    config = STAGE_CONFIG[stage]
    log_prob = config["log_prob"]
    bounds   = config["bounds"]
    p0_center = config["p0_center"]
    p0_scatter = config["p0_scatter"]
    n_dim = p0_center.size

    n_walkers = max(n_walkers, 2 * n_dim + 2)
    rng = np.random.default_rng(seed)

    # 1. L-BFGS-B warm-start
    def nll(x):
        lp = log_prob(x, S_obs_flat, SIGMA_NOISE, protocol)
        return 1e10 if not np.isfinite(lp) else -lp

    sol = minimize(nll, p0_center, bounds=bounds, method="L-BFGS-B")

    # 2. Scatter walkers around MAP, clipped to bounds
    pos = sol.x + p0_scatter * rng.standard_normal((n_walkers, n_dim))
    for d in range(n_dim):
        pos[:, d] = np.clip(pos[:, d], bounds[d][0] + 1e-6,
                            bounds[d][1] - 1e-6)

    # 3. Run emcee with DEMove + DESnookerMove for better ridge exploration.
    #    The default StretchMove struggles with correlated banana-shaped
    #    posteriors (exactly what HCP Stage 6 produces).  DEMove proposes
    #    along the difference of two random walkers, which naturally follows
    #    the ridge; DESnookerMove adds a perpendicular component.
    moves = [
        (emcee.moves.DEMove(),         0.8),
        (emcee.moves.DESnookerMove(),  0.2),
    ]
    sampler = emcee.EnsembleSampler(
        n_walkers, n_dim, log_prob,
        args=(S_obs_flat, SIGMA_NOISE, protocol),
        moves=moves,
    )
    sampler.run_mcmc(pos, n_steps, progress=False)

    # 4. Convergence diagnostics (on post-burn-in chain, before thinning)
    full_chain = sampler.get_chain()   # (n_steps, n_walkers, n_dim)
    post_burn = full_chain[n_burn:]    # (n_steps-n_burn, n_walkers, n_dim)
    # Transpose to (n_walkers, n_steps_kept, n_dim) for split-Rhat
    post_burn_t = post_burn.transpose(1, 0, 2)

    try:
        tau = sampler.get_autocorr_time(quiet=True)
    except Exception:
        tau = np.full(n_dim, np.nan)

    try:
        rhat = compute_split_rhat(post_burn_t)
    except Exception:
        rhat = np.full(n_dim, np.nan)

    # 5. Flatten with burn-in and thinning
    flat    = sampler.get_chain(discard=n_burn, thin=n_thin, flat=True)
    flat_lp = sampler.get_log_prob(discard=n_burn, thin=n_thin, flat=True)

    return flat, flat_lp, sol.x, tau, rhat


# ---------------------------------------------------------------------------
# Truth-vector helper
# ---------------------------------------------------------------------------
def truth_vector(stage, v_glia):
    bank = {
        "v_ic": V_IC_TRUE, "ODI": ODI_TRUE, "v_glia": v_glia,
        "v_iso": V_ISO_TRUE,
        "T2_t": T2_TISSUE_TRUE, "T2_s": T2_SPHERE_TRUE,
        "theta": THETA_TRUE, "phi": PHI_TRUE,
    }
    return np.array([bank[p] for p in STAGE_FREE_PARAMS[stage]])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(quick: bool = False):
    ensure_dirs()

    if quick:
        n_walkers = 20
        n_steps, n_burn, n_thin = 300, 80, 5
    else:
        n_walkers = N_WALKERS
        n_steps, n_burn, n_thin = N_STEPS, N_BURN, N_THIN

    print("=" * 72)
    print(f"1205_solver_engine.py -- emcee MCMC ({'QUICK' if quick else 'FULL'})")
    print("=" * 72)
    print(f"  walkers  = {n_walkers}")
    print(f"  steps    = {n_steps}")
    print(f"  burn     = {n_burn}")
    print(f"  thin     = {n_thin}")
    print()

    seed_counter = SEED
    t_start = time.time()
    n_runs = 0

    for proto_name, protocol in PROTOCOLS.items():
        print(f"[{protocol.label}]  n_grad={protocol.n_grad}  "
              f"multi-TE={len(protocol.te_multi)} TE(s)")

        for stage in STAGE_NAMES:
            for v_glia in GLIA_FRACS:
                if stage == "stage1" and v_glia != GLIA_FRACS[0]:
                    continue

                signal_path = _signal_path(proto_name, stage, v_glia, "noisy")
                if not os.path.exists(signal_path):
                    raise FileNotFoundError(
                        f"Missing {signal_path}. "
                        "Run 1205_simulate_data.py first."
                    )

                S_obs = np.load(signal_path).ravel()
                seed_counter += 1
                t0 = time.time()

                flat, flat_lp, p_map, tau, rhat = run_mcmc_for(
                    stage=stage, protocol=protocol,
                    S_obs_flat=S_obs, seed=seed_counter,
                    n_walkers=n_walkers, n_steps=n_steps,
                    n_burn=n_burn, n_thin=n_thin,
                )
                dt = time.time() - t0
                n_runs += 1

                truths = truth_vector(stage, float(v_glia))
                np.savez(
                    _chain_path(proto_name, stage, v_glia),
                    flat=flat, log_prob=flat_lp,
                    param_names=np.array(STAGE_FREE_PARAMS[stage]),
                    truths=truths, v_glia=float(v_glia),
                    map_estimate=p_map,
                    autocorr_time=tau,
                    split_rhat=rhat,
                )

                med = np.median(flat, axis=0)
                rhat_max = np.nanmax(rhat)
                rhat_flag = " **RHAT>{:.2f}**".format(rhat_max) if rhat_max > 1.1 else ""
                summary = "  ".join(
                    f"{p}={m:.3f}(t={t:.3f})"
                    for p, m, t in zip(STAGE_FREE_PARAMS[stage], med, truths)
                )
                print(f"  {stage}  vg={v_glia:.2f}  [{dt:5.1f}s]  "
                      f"{summary}{rhat_flag}")

        print()

    total = time.time() - t_start
    print(f"Done. {n_runs} MCMC runs in {total:.1f}s. "
          f"Chains saved to {CHAINS_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Abbreviated MCMC for smoke tests.")
    args = parser.parse_args()
    main(quick=args.quick)
