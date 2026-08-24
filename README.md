# Biophysical Modelling of Tissue Microstructure with MRI

Separating neuroinflammation from axonal degeneration in diffusion MRI using a
glial compartment and multi-echo-time (multi-TE) acquisition.

**MEng Biomedical Engineering Individual Project — Imperial College London**
Author: Elif Civelekoglu · Supervisor: Dr. Amy Howard

---

## Overview

Traumatic brain injury (TBI) triggers neuroinflammation (glial swelling) and axonal degeneration at the same time. Standard diffusion MRI cannot tell the two apart, because the widely used NODDI model has no glial compartment and acquires at a single echo time (TE). At a fixed TE every compartment contributes a signal equal to `v · exp(-TE/T2)`, which is the product of its volume fraction and its T2-weighting. So, one measurement cannot separate volume fraction from relaxation, making the fit degenerate

This project tests, by simulation, whether adding a restricted glial-sphere compartment together with multi-TE acquisition and compartment-specific T2 relaxation resolves that degeneracy. It does so with two experiments built on a shared forward model, noise model, and Bayesian (MCMC) inference engine:
- **Experiment A — single-voxel parameter sweep** (`noddi_project/`): sweeps the  glial fraction at one voxel and isolates the mathematical separability of the glial compartment. Reported with T2 estimated freely, which exposes the degeneracy directly.
- **Experiment B — spatially realistic white-matter phantom** (`phantom_hcp/`): embeds synthetic TBI lesions in a real HCP white-matter slice and tests whether recovery survives variable fibre orientation and per-voxel noise. Reported with T2 fixed at ground truth, which yields interpretable spatial recovery maps.

---

## The two acquisition protocols

| | NODDI / gaNODDI (single-TE) | Novel (multi-TE) |
|---|---|---|
| b-shells (ms/µm²) | 1.0, 2.0, 3.0 | 1.0, 2.0 |
| Echo times (ms) | 62 | 62, 92, 130 |
| Gradient dirs / shell | 64 (Fibonacci) | 64 (Fibonacci) |
| b=0 volumes | 6 | 6 per TE |
| Total measurements | 198 | 402 |

Shared PGSE timings: big Δ = 43.1 ms, little δ = 10.6 ms. Glial cells are modelled as restricted spheres via the Gaussian Phase Distribution (GPD) approximation with radius R = 5.0 µm and diffusivity D = 3.0 µm²/ms. Signals are corrupted with Rician noise at SNR = 20.

### Ground-truth tissue parameters

| Condition | v_ic | ODI | v_iso | v_glia |
|---|---|---|---|---|
| Healthy white matter | 0.73 | 0.25 | 0.16 | 0.00 |
| Astrogliosis | 0.65 | 0.55 | 0.16 | 0.20 |
| Edema | 0.65 | 0.25 | 0.45 | 0.00 |
| Chronic TBI | 0.55 | 0.50 | 0.35 | 0.15 |

Compartment relaxation times: tissue (intra + extra) T2 = 100 ms, glial sphere T2 = 30 ms, free water (CSF) T2 = 2000 ms.

---

## Repository structure

```
.
├── README.md                       ← this file
├── requirements.txt
├── .gitignore
│
├── experiment_a/                   ← Experiment A: single-voxel sweep
│   ├── essential_config.py            constants, protocols, stage definitions, MCMC settings
│   ├── essential_forward_models.py    compartment + full forward signal models
│   ├── essential_compartments.py      analytical compartment signals (stick, sphere, ball, EC)
│   ├── essential_sh_utils.py          Fibonacci sphere + spherical-harmonic utilities
│   ├── exp_a_1_simulate_data.py       [step 1] generate noisy synthetic signals
│   ├── exp_a_2_solver_engine.py       [step 2] emcee MCMC inversion + split-Rhat
│   ├── exp_a_3_analyze_metrics.py     [step 3] posterior summaries, convergence, pseudo-ROC
│   └── exp_a_4_visualize_results.py   [step 4] figures (posteriors, sweeps, accuracy/precision)
│
└── experiment_b/                   ← Experiment B: spatial HCP phantom
    ├── exp_b_101_extract_priors.py    derive healthy-WM baselines from HCP NODDI maps 
    ├── exp_b_102_extract_slice.py     extract axial slice + WM mask 
    ├── exp_b_201_roi_adder.py         MAIN phantom generator (astro / edema / chronic_tbi,
    │                                  both protocols, full multi-TE glial signal)
    ├── exp_b_301_fit_mcmc_engine.py   per-voxel MCMC fitter (fixed-T2), parallel + checkpointing
    ├── exp_b_401_fullwm_roc_cm.py     ROC + confusion matrices 
    ├── exp_b_402_recovery_audit.py    lesion + healthy recovery: bias / MAE / RMSE 
    ├── exp_b_403_convergence.py       per-voxel split-Rhat summary
    ├── exp_b_404_fullwm_spatial.py    spatial true-vs-predicted lesion maps
    ├── essential_hcp_compartments.py  vectorised compartment signals (arrays of b-values)
    ├── essential_bbdb_compartments.py sphere (GPD) compartment signal
    ├── essential_bbdb_sh_utils.py     Fibonacci sphere utilities
    ├── essential_check_speed.py       time the fitter on a few voxels before a long job
    └── essential_hcp_cube_parallel.py parallel NODDI fit of a WM cube (utility / sanity check,
                                       expects a 100307/ subfolder with wm_cube.nii, bvals, bvecs)
```
---

## Method summary

### Forward signal models

The standard three-compartment NODDI signal is a volume-weighted sum of a restricted intra-cellular stick population (Watson-dispersed, integrated over 500 Fibonacci samples), a tortuosity-constrained hindered extra-cellular tensor, and an isotropic free-water ball:

```
S_NODDI = (1 - v_iso) · [ v_ic·A_ic + (1 - v_ic)·A_ec ] + v_iso·A_iso
```

The novel model adds a restricted glial sphere and attaches a compartment-specific T2 weight `exp(-TE/T2)` to every compartment:

```
S_novel = (1 - v_iso) · [ (1 - v_glia)·(v_ic·A_ic + (1-v_ic)·A_ec)·e^(-TE/T2,t)
                          +  v_glia·A_sphere·e^(-TE/T2,s) ]
        +  v_iso·A_iso·e^(-TE/T2,c)
```

evaluated at each TE in the protocol. Compartment signal functions live in the respective `essential_` compartment files.

### Bayesian inference

Model inversion uses MCMC with the **emcee** ensemble sampler. Each fit is warm-started with a multi-start L-BFGS-B optimiser, then sampled with a mixture of Differential Evolution (80%) and DE-Snooker (20%) moves to traverse correlated posterior ridges. Priors are uniform within biologically plausible bounds:

| Parameter | Symbol | Lower | Upper |
|---|---|---|---|
| Intracellular fraction | v_ic | 0.01 | 0.99 |
| Orientation dispersion | ODI | 0.01 | 0.99 |
| Glia fraction | v_glia | 0.00 | 0.49 |
| Isotropic fraction | v_iso | 0.00 | 0.49 |
| Tissue T2 | T2,t | 10 ms | 500 ms |
| Sphere T2 | T2,s | 5 ms | 500 ms |
| Fibre elevation | θ | 0 | π |
| Fibre azimuth | φ | 0 | 2π |

A Gaussian log-likelihood is used as a high-SNR approximation to the Rician noise (σ = mean b=0 signal / SNR). Convergence is assessed by integrated autocorrelation time and **split-Rhat** (Rhat < 1.1 = acceptable mixing).

Experiment A uses 56 walkers and long chains (config: 40 000 steps, 10 000 burn-in) over a glia sweep `v_glia ∈ {0.00, 0.10, 0.20, 0.30}` and a staged fit ladder (stages 1–6 in `1205_config.py`, from a sanity NODDI fit through the full free-T2 fit). Experiment B fits each white-matter voxel independently under the lower-dimensional fixed-T2 configuration (32 walkers, ~4000 steps, 400 burn-in).

### Evaluation

- **Experiment A:** posterior median, relative bias, and 68% credible-interval width per parameter.
- **Experiment B:** per-ROI bias / MAE / RMSE; global Pearson correlation; and a pseudo-ROC analysis (AUC, Youden-optimal threshold, sensitivity, specificity, precision) classifying lesion vs healthy WM from the posterior v_glia.

---

## Installation

Python 3.9+ needed.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Core dependencies: `numpy`, `scipy`, `emcee`, `nibabel`, `matplotlib`, `corner`, `scikit-learn` (`numba` optional, only for the whole-brain NODDI map utilities).

---

## Data

**Experiment A** needs no external data — it generates its own synthetic signals.

**Experiment B** uses diffusion MRI from the **Human Connectome Project (HCP)**,
WU-Minn Consortium (subjects 100206 / 100307 in the scripts). HCP data are not
redistributed here; obtain them from <https://www.humanconnectome.org/> and place
the subject folder (`data.nii`, `bvals`, `bvecs`, `wm_mask.nii`,
`nodif_brain_mask.nii`, and the precomputed NODDI maps the scripts reference)
under `phantom_hcp/HCP Brain Phantom/<subject_id>/`. Slice/ROI paths default to
subject `100206`.

---

## Reproducing the experiments

### Experiment A — single-voxel sweep (`noddi_project/`)

```bash
cd experiment_a
python exp_a_1_simulate_data.py      # 1. generate noisy synthetic signals  -> signals/
python exp_a_2_solver_engine.py      # 2. MCMC inversion -> chains/
python exp_a_3_analyze_metrics.py    # 3. posterior summaries + convergence  -> metrics/
python exp_a_4_visualize_results.py  # 4. figures
```

### Experiment B — spatial phantom (`phantom_hcp/HCP Brain Phantom/`)

```bash
cd experiment_b

# 1. Preprocess HCP data
python exp_b_101_extract_priors.py          # healthy-WM baselines from HCP NODDI maps
python exp_b_102_extract_slice.py           # extract axial slice + cleaned WM mask

# 2. Generate phantoms (astrogliosis, edema, chronic_tbi) for both protocols
python exp_b_201_roi_adder.py          # -> results/glia/<pathology>/phantom_*_data_{hcp,novel}.nii

# 3. Per-voxel MCMC fit (fixed-T2). 
python exp_b_301_fit_mcmc_engine.py --pathology all --protocol both
python essential_check_speed.py --pathology edema --protocol novel --n 8   # benchmark first

# 4. Analysis, tables, and figures
python exp_b_401_fullwm_roc_cm.py           # ROC + confusion matrices 
python exp_b_402_recovery_audit.py          # recovery bias/MAE/RMSE 
python exp_b_403_convergence.py             # per-voxel split-Rhat 
python exp_b_404_fullwm_spatial.py          # spatial true-vs-predicted lesion maps
```

The fitter checkpoints results (`*_ckpt*.npy`) and resumes automatically, and saves per-parameter NIfTI maps for MAP, median, std, autocorrelation time, and Rhat. Map volume order is `[v_ic, ODI, v_glia, v_iso, theta, phi]`.

---

## Key results

**Experiment A.** With T2 free, the single-TE NODDI fit is degenerate: posteriors are broad and biased, and v_ic/v_iso drift away from truth as the unmodelled glial fraction grows. The multi-TE protocol stays close to truth and is ~6× more precise on tissue T2. The eight-parameter novel fit mixes more slowly (a few chains sit just above Rhat = 1.1), but median-based estimates are unaffected.

**Experiment B** (fixed-T2, full white-matter slice; n = 1073 lesion, 4346 healthy):

| Pathology | Protocol | AUC | Threshold | Sensitivity | Specificity |
|---|---|---|---|---|---|
| Astrogliosis | gaNODDI | 1.000 | 0.090 | 1.000 | 1.000 |
| Astrogliosis | Novel | 1.000 | 0.131 | 1.000 | 1.000 |
| Edema | gaNODDI | 0.887 | 0.029 | 0.854 | 0.743 |
| Edema | Novel | 0.862 | 0.018 | 0.840 | 0.718 |

Under edema the true glial fraction is zero everywhere, so the high AUC is misleading — both protocols leak free-water signal into the glial channel. v_ic is consistently the most biased estimate across all pathologies.

---

## Limitations

Intrinsic diffusivities and glial geometry (radius, diffusivity) are held fixed to preserve identifiability; the phantom is built from closed-form analytical compartments and excludes effects such as membrane permeability. The sphere T2 (T2,s) is unrecoverable because the short-T2 glial signal has largely decayed by the shortest TE sampled. Inference uses a Gaussian approximation to Rician noise, which is mild at SNR 20 but would matter more at clinical SNR. Validation on numerical substrates or ex-vivo data is a necessary next step before clinical use.

---

## Acknowledgements

Data were provided in part by the Human Connectome Project, WU-Minn Consortium
(Principal Investigators: David Van Essen and Kamil Ugurbil; 1U54MH091657) funded
by the 16 NIH Institutes and Centers that support the NIH Blueprint for
Neuroscience Research, and by the McDonnell Center for Systems Neuroscience at
Washington University. FSLeyes was used to visualise HCP data during development.

## References

Key methods follow Zhang et al. (NODDI), the emcee ensemble sampler
(Foreman-Mackey et al.), the GPD sphere approximation (Stanisz et al.), and the
NODDI degeneracy literature (Jelescu et al.; Novikov et al.; Howard et al.). Full
citations are in the project report.
