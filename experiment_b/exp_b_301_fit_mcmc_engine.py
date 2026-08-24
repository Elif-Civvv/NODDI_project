import nibabel as nib
import numpy as np
import os
import time
import multiprocessing
from functools import partial
from scipy.optimize import minimize
import emcee

try:
    from hcp_compartments import (
        calculate_watson_stick_signal,
        calculate_noddi_extra_signal,
    )
    from bbdb_compartments import calculate_sphere_signal
    from bbdb_sh_utils import fibonacci_sphere
except ImportError:
    raise ImportError("Ensure compartment and sh_utils files are accessible.")

# Physics constants (MUST match the phantom generator exactly)
D_PAR = 1.7
D_ISO = 3.0
D_GLIA = 3.0
R_GLIA = 5.0
LITTLE_DELTA = 10.6
BIG_DELTA = 43.1

T2_TISSUE = 100.0   # ms, fixed at truth in Stage 5
T2_SPHERE = 30.0    # ms, fixed at truth in Stage 5
T2_CSF = 2000.0     # ms, always fixed

N_WALKERS = 32
N_STEPS = 1200
N_BURN = 400
N_THIN = 5

PROTOCOLS = ["hcp", "novel"]

# Save every CKPT_EVERY new voxels OR every CKPT_SECONDS seconds, whichever
# first -- so a wall-kill costs at most a few voxels, not hundreds.
CKPT_EVERY = 25
CKPT_SECONDS = 300


def b_to_G(b, delta, Delta):
    if b < 1e-10:
        return 0.0
    gamma = 2.6752218744e8
    return np.sqrt(
        (b * 1e9) / (gamma**2 * (delta * 1e-3)**2 * ((Delta * 1e-3) - (delta * 1e-3) / 3.0))
    ) * 1e3


def odi_to_kappa(odi):
    odi = np.clip(odi, 0.01, 0.99)
    return 1.0 / np.tan(odi * np.pi / 2.0)


def precompute_spheres(bvals, bvecs):
    cache = {}
    for b in np.unique(bvals):
        if b < 1e-10:
            cache[0.0] = 1.0
        else:
            cache[round(b, 6)] = calculate_sphere_signal(
                b, b_to_G(b, LITTLE_DELTA, BIG_DELTA),
                LITTLE_DELTA, BIG_DELTA, R_GLIA, D_GLIA, bvecs[:1]
            )[0]
    return cache


def compute_split_rhat(chain_3d):
    n_walkers, n_steps, n_dim = chain_3d.shape
    half = n_steps // 2
    splits = np.concatenate([chain_3d[:, :half, :], chain_3d[:, half:2 * half, :]], axis=0)
    m, n = splits.shape[0], splits.shape[1]
    chain_means = splits.mean(axis=1)
    chain_vars = splits.var(axis=1, ddof=1)
    B = n * chain_means.var(axis=0, ddof=1)
    W = chain_vars.mean(axis=0)
    var_hat = (1 - 1.0 / n) * W + B / n
    return np.sqrt(var_hat / np.where(W > 0, W, 1e-10))


def fit_voxel_mcmc(S_raw, bvals, bvecs, TEs, watson_samples, sphere_cache, seed=42):
    n_dim = 6
    failed_result = np.zeros(n_dim * 5)
    if np.max(S_raw) < 0.05:
        return failed_result

    bounds = [(0.01, 0.99), (0.01, 0.99), (0.0, 0.49), (0.0, 0.49), (0.0, np.pi), (0.0, 2 * np.pi)]
    p0_scatter = np.array([0.05, 0.05, 0.02, 0.05, 0.05, 0.1])

    b0_mask = bvals < 1e-10
    dynamic_sigma = (np.mean(S_raw[b0_mask]) / 20.0) if np.any(b0_mask) else 0.05

    def build_signal(x):
        v_ic, odi, v_glia, v_iso, theta, phi = x
        kappa = odi_to_kappa(odi)
        mu = np.array([np.sin(theta) * np.cos(phi),
                       np.sin(theta) * np.sin(phi),
                       np.cos(theta)])
        A_ic = calculate_watson_stick_signal(bvals, D_PAR, mu, kappa, bvecs, watson_samples)
        A_ec = calculate_noddi_extra_signal(bvals, D_PAR, v_ic, kappa, mu, bvecs)
        A_ball = np.exp(-bvals * D_ISO)
        A_sphere = np.array([sphere_cache.get(round(b, 6), 1.0) for b in bvals])
        w_t = np.exp(-TEs / T2_TISSUE)
        w_s = np.exp(-TEs / T2_SPHERE)
        w_c = np.exp(-TEs / T2_CSF)
        tissue = (1.0 - v_glia) * (v_ic * A_ic + (1.0 - v_ic) * A_ec) * w_t
        glia = v_glia * A_sphere * w_s
        csf = v_iso * A_ball * w_c
        return (1.0 - v_iso) * (tissue + glia) + csf

    def log_prob(x):
        for i, (low, high) in enumerate(bounds):
            if not (low <= x[i] <= high):
                return -np.inf
        pred = build_signal(x)
        return (-0.5 * len(S_raw) * np.log(2 * np.pi * dynamic_sigma**2)
                - 0.5 * np.sum((S_raw - pred)**2) / dynamic_sigma**2)

    def nll(x):
        lp = log_prob(x)
        return -lp if np.isfinite(lp) else 1e10

    te_min = np.min(TEs)
    dwi_mask = (bvals > 0.5) & np.isclose(TEs, te_min)
    if np.any(dwi_mask):
        min_idx = np.argmin(S_raw[dwi_mask])
        mu_est = bvecs[dwi_mask][min_idx]
        theta_init = np.arccos(np.clip(mu_est[2], -1, 1))
        phi_init = np.arctan2(mu_est[1], mu_est[0]) % (2 * np.pi)
    else:
        theta_init, phi_init = np.pi / 4, 0.0

    x0 = [0.5, 0.4, 0.1, 0.1, theta_init, phi_init]
    try:
        sol = minimize(nll, x0, method='L-BFGS-B', bounds=bounds)
        p_map = sol.x
    except Exception:
        p_map = np.array(x0)

    rng = np.random.default_rng(seed)
    pos = p_map + p0_scatter * rng.standard_normal((N_WALKERS, n_dim))
    for d in range(n_dim):
        pos[:, d] = np.clip(pos[:, d], bounds[d][0] + 1e-6, bounds[d][1] - 1e-6)

    moves = [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)]
    sampler = emcee.EnsembleSampler(N_WALKERS, n_dim, log_prob, moves=moves)
    try:
        sampler.run_mcmc(pos, N_STEPS, progress=False)
    except Exception:
        return failed_result

    chain = sampler.get_chain()
    post_burn_t = chain[N_BURN:].transpose(1, 0, 2)
    try:
        tau = sampler.get_autocorr_time(quiet=True)
    except Exception:
        tau = np.full(n_dim, np.nan)
    try:
        rhat = compute_split_rhat(post_burn_t)
    except Exception:
        rhat = np.full(n_dim, np.nan)

    flat_chain = sampler.get_chain(discard=N_BURN, thin=N_THIN, flat=True)
    p_median = np.median(flat_chain, axis=0)
    p_std = np.std(flat_chain, axis=0)
    return np.concatenate([p_map, p_median, p_std, tau, rhat])


def run_mcmc_phantom(pathology_name, protocol_name, roi_only=False, healthy_cap=300):
    out_suffix = "_roi" if roi_only else ""
    HERE = os.path.dirname(os.path.abspath(__file__))
    PHANTOM_DIR = os.path.join(HERE, "results", "glia", pathology_name)
    prefix = f"phantom_{pathology_name}"

    print(f"\n{'=' * 72}")
    print(f"MCMC Fitting: {pathology_name} | Protocol: {protocol_name}")
    print(f"{'=' * 72}")

    data_path = os.path.join(PHANTOM_DIR, f"{prefix}_data_{protocol_name}.nii")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing {data_path}. Run the phantom generator first.")

    img = nib.load(data_path)
    data, affine = img.get_fdata(), img.affine
    wm_mask = nib.load(os.path.join(PHANTOM_DIR, f"{prefix}_vic_gt.nii")).get_fdata()[:, :, 0] > 0

    if roi_only:
        roi = nib.load(os.path.join(PHANTOM_DIR, f"{prefix}_roi_labels.nii")).get_fdata()[:, :, 0] > 0
        healthy = wm_mask & ~roi
        h_idx = np.argwhere(healthy)
        rng = np.random.default_rng(0)
        if len(h_idx) > healthy_cap:
            keep = rng.choice(len(h_idx), healthy_cap, replace=False)
            healthy_sub = np.zeros_like(healthy)
            for r, c in h_idx[keep]:
                healthy_sub[r, c] = True
            healthy = healthy_sub
        mask = roi | healthy
        print(f"ROI-only mode: {int(roi.sum())} lesion + {int(healthy.sum())} healthy = {int(mask.sum())} voxels")
    else:
        mask = wm_mask

    bvals = np.loadtxt(os.path.join(PHANTOM_DIR, f"bvals_{protocol_name}.txt"))
    bvecs = np.loadtxt(os.path.join(PHANTOM_DIR, f"bvecs_{protocol_name}.txt"))
    TEs = np.loadtxt(os.path.join(PHANTOM_DIR, f"TEs_{protocol_name}.txt"))

    watson_samples = fibonacci_sphere(samples=500)
    sphere_cache = precompute_spheres(bvals, bvecs)

    voxels_to_fit = data[mask, 0, :]
    total_voxels = len(voxels_to_fit)
    cores = int(os.environ.get("NCPUS", multiprocessing.cpu_count()))

    print(f"Total voxels: {total_voxels} | CPUs: {cores}")
    print(f"Measurements/voxel: {voxels_to_fit.shape[1]} | unique TEs: {sorted(np.unique(TEs).tolist())}")
    print(f"MCMC Config: {N_WALKERS} walkers, {N_STEPS} steps, {N_BURN} burn-in")

    worker_func = partial(fit_voxel_mcmc, bvals=bvals, bvecs=bvecs, TEs=TEs,
                          watson_samples=watson_samples, sphere_cache=sphere_cache)

    ckpt_path = os.path.join(PHANTOM_DIR, f"{prefix}_{protocol_name}_ckpt{out_suffix}.npy")
    if os.path.exists(ckpt_path):
        results_array = np.load(ckpt_path)
        if results_array.shape != (total_voxels, 30):
            print(f"WARNING: checkpoint shape {results_array.shape} != ({total_voxels}, 30); starting fresh.")
            results_array = np.zeros((total_voxels, 30))
        done = int(np.count_nonzero(results_array.any(axis=1)))
        print(f"Resuming from checkpoint: {done}/{total_voxels} voxels already fitted.")
    else:
        results_array = np.zeros((total_voxels, 30))
        done = 0

    start_time = time.time()
    last_save = start_time

    remaining_idx = [i for i in range(total_voxels) if not results_array[i].any()]
    remaining_vox = voxels_to_fit[remaining_idx]
    print(f"Voxels remaining to fit this run: {len(remaining_idx)}")

    if len(remaining_idx) == 0:
        print("Nothing to fit -- all voxels already in checkpoint.")
    else:
        with multiprocessing.Pool(processes=cores) as pool:
            for k, res in enumerate(pool.imap(worker_func, remaining_vox, chunksize=1)):
                i = remaining_idx[k]
                results_array[i, :] = res
                if (k + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    speed = (k + 1) / elapsed
                    eta = (len(remaining_vox) - k - 1) / speed / 60
                    print(f"  {done + k + 1}/{total_voxels} | {speed:.2f} vox/s | ETA: {eta:.1f} min", end="\r")
                if (k + 1) % CKPT_EVERY == 0 or (time.time() - last_save) > CKPT_SECONDS:
                    np.save(ckpt_path, results_array)
                    last_save = time.time()

    np.save(ckpt_path, results_array)
    print(f"\nMCMC Fitting complete in {(time.time() - start_time) / 60:.1f} minutes")

    metric_names = ["map", "median", "std", "tau", "rhat"]
    mask_indices = np.where(mask)
    for m_idx, m_name in enumerate(metric_names):
        metric_vol = np.zeros((data.shape[0], data.shape[1], 1, 6))
        for p_idx in range(6):
            flat_idx = (m_idx * 6) + p_idx
            for v_idx in range(total_voxels):
                r, c = mask_indices[0][v_idx], mask_indices[1][v_idx]
                metric_vol[r, c, 0, p_idx] = results_array[v_idx, flat_idx]
        save_path = os.path.join(PHANTOM_DIR, f"{prefix}_{protocol_name}_mcmc_{m_name}{out_suffix}.nii")
        nib.save(nib.Nifti1Image(metric_vol, affine), save_path)

    print(f"Saved MCMC maps to {PHANTOM_DIR}")
    print("  Parameter order per map: [v_ic, odi, v_glia, v_iso, theta, phi]")


if __name__ == "__main__":
    import argparse
    ALL_PATHOLOGIES = ["astrogliosis", "edema", "chronic_tbi"]
    parser = argparse.ArgumentParser(description="Spatial-phantom MCMC fitter (Stage 5).")
    parser.add_argument("--protocol", choices=["hcp", "novel", "both"], default="both")
    parser.add_argument("--pathology", choices=ALL_PATHOLOGIES + ["all"], default="all")
    parser.add_argument("--roi-only", action="store_true")
    args = parser.parse_args()
    protocols_to_run = PROTOCOLS if args.protocol == "both" else [args.protocol]
    pathologies_to_run = ALL_PATHOLOGIES if args.pathology == "all" else [args.pathology]
    for pathology in pathologies_to_run:
        for protocol in protocols_to_run:
            run_mcmc_phantom(pathology, protocol_name=protocol, roi_only=args.roi_only)
