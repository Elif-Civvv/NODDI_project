import numpy as np
from scipy.special import dawsn

# ---------------------------------------------------------------------------
# Sphere compartment (GPD approximation, Stanisz et al.)
# Used to model isotropically restricted diffusion in glial cell soma.
# ---------------------------------------------------------------------------

# Gyromagnetic ratio
_gamma = 2.6752218744e8          # [rad s^-1 T^-1]
_gamma_ms = _gamma * 1e-3        # [rad ms^-1 T^-1]

# 60 roots of  am * J_{3/2}'(am) - (1/2) * J_{3/2}(am) = 0  (from Camino)
_am = np.array([
    2.08157597781810,  5.94036999057271,  9.20584014293667,
   12.4044450219020,  15.5792364103872,  18.7426455847748,
   21.8996964794928,  25.0528252809930,  28.2033610039524,
   31.3520917265645,  34.4995149213670,  37.6459603230864,
   40.7916552312719,  43.9367614714198,  47.0813974121542,
   50.2256516491831,  53.3695918204908,  56.5132704621986,
   59.6567290035279,  62.8000005565198,  65.9431119046553,
   69.0860849466452,  72.2289377620154,  75.3716854092873,
   78.5143405319308,  81.6569138240367,  84.7994143922025,
   87.9418500396598,  91.0842274914688,  94.2265525745684,
   97.3688303629010, 100.511065295271,  103.653261271734,
  106.795421732944,  109.937549725876,  113.079647958579,
  116.221718846033,  116.221718846033,  119.363764548757,
  122.505787005472,  125.647787960854,  128.789768989223,
  131.931731514843,  135.073676829384,  138.215606107009,
  141.357520417437,  144.499420737305,  147.641307960079,
  150.783182904724,  153.925046323312,  157.066898907715,
  166.492397790874,  169.634212946261,  172.776020008465,
  175.917819411203,  179.059611557741,  182.201396823524,
  185.343175558534,  188.484948089409,  191.626714721361,
])


def _compute_GPDsum(am_r, pulse_duration, diffusion_time, diffusivity, radius):
    """
    Computes the GPD sum for a sphere (scalar or 1-D b array).

    am_r            : shape (60,) or (60, 1)  — am / radius
    pulse_duration  : delta  [ms]
    diffusion_time  : Delta  [ms]
    diffusivity     : D      [um^2 / ms]
    radius          : R      [um]
    """
    dam = diffusivity * am_r * am_r          # (60,) or (60, N)
    e1  = np.exp(-dam * pulse_duration)
    e2  = np.exp(-dam * diffusion_time)
    dif = diffusion_time - pulse_duration
    e3  = np.exp(-dam * dif)
    plus = diffusion_time + pulse_duration
    e4  = np.exp(-dam * plus)

    nom   = 2*dam*pulse_duration - 2 + 2*e1 + 2*e2 - e3 - e4
    denom = dam**2 * am_r**2 * (radius**2 * am_r**2 - 2)

    return np.sum(nom / denom, axis=0)


def calculate_sphere_signal(b, G, pulse_duration, diffusion_time, radius, diffusivity, g_vectors):
    """
    Signal from water isotropically restricted inside a sphere (GPD approximation).

    Restricted diffusion is isotropic, so the signal is direction-independent.

    Parameters
    ----------
    b               : float  — b-value [ms / um^2]  (scalar, single shell)
    G               : float  — gradient strength [mT / m]  (used only to compute log_att;
                               must be consistent with b = gamma^2 * G^2 * delta^2 * (Delta - delta/3))
    pulse_duration  : float  — little delta  [ms]
    diffusion_time  : float  — big Delta     [ms]
    radius          : float  — sphere radius [um]
    diffusivity     : float  — intra-sphere diffusivity [um^2 / ms]
    g_vectors       : ndarray (N, 3) — gradient directions (unit vectors)

    Returns
    -------
    S : ndarray (N,) — signal attenuation (isotropic, same for all directions)
    """
    # Convert G: mT/m -> T/um
    G_T_per_um = G * 1e-3 * 1e-6          # [T / um]

    am_r = _am / radius                    # (60,)

    GPDsum = _compute_GPDsum(am_r, pulse_duration, diffusion_time, diffusivity, radius)

    log_att = -2.0 * _gamma_ms**2 * G_T_per_um**2 * GPDsum

    S = np.exp(log_att)
    return np.full(g_vectors.shape[0], S)

def calculate_ball_signal(b, D_iso, g_vectors):
    """S_ball = exp(-b * D_iso)"""
    # Isotropic, so signal is the same in all directions
    ADC = D_iso
    S = np.exp(-b * ADC)
    # Return an array with the same value for all gradients
    return np.full(g_vectors.shape[0], S)

def calculate_stick_signal(b, D_stick, mu_vector, g_vectors):
    """
    Calculates signal for the stick model:
    S = S_0 * exp(-b * D * (g . mu)^2)
    """
    # ensure mu is a unit vector
    mu = mu / np.linalg.norm(mu)
    # g_vectors has shape (30, 3)
    # mu_vector has shape (3,)
    # (30, 3) @ (3,) -> (30,)
    g_dot_mu = g_vectors @ mu_vector 

    cos_squared_theta = g_dot_mu**2 #(g . mu)^2 = cos^2(theta)

    # Calculate the signal
    S = np.exp(-b * D_stick * cos_squared_theta)
    
    return S

def calculate_tensor_signal(b, D_tensor_vals, g_vectors):
    """
    S_tensor = exp(-b * (Dx*gx^2 + Dy*gy^2 + Dz*gz^2))
    D_tensor_vals is an array [Dx, Dy, Dz]
    """
    g_squared = g_vectors**2  # [gx^2, gy^2, gz^2] for each vector
    # ADC = Dx*gx^2 + Dy*gy^2 + Dz*gz^2
    ADC = g_squared @ D_tensor_vals
    S = np.exp(-b * ADC)
    return S

#  NODDI specific things I got from Zhang et al.

# this one is basically the intra signal discussed in zhang et al. 
def calculate_watson_stick_signal(b, D_stick, mu_vector, kappa, g_vectors, sample_orientations):
    # 1. Calculate Watson Distribution Weights f(n) 
    # Ensure mu is unit vector
    mu_vector = mu_vector / np.linalg.norm(mu_vector)
    
    mu_dot_n = sample_orientations @ mu_vector
    unnormalized_probs = np.exp(kappa * mu_dot_n**2)
    weights = unnormalized_probs / np.sum(unnormalized_probs) # (N,) array
    
    # 2. Calculate Signal for EACH Sampled Stick 
    # g_vectors shape: (M, 3), sample_orientations shape: (N, 3) -> (M, N)
    g_dot_n_matrix = g_vectors @ sample_orientations.T 
    cos_squared_matrix = g_dot_n_matrix**2
    stick_signals_matrix = np.exp(-b * D_stick * cos_squared_matrix)
    
    # 3. Compute the Weighted Average (The Integral) 
    S_attenuated = stick_signals_matrix @ weights # (M, N) @ (N,) -> (M,)
    
    return S_attenuated

# calculating the extracellular signal
def calculate_noddi_extra_signal(b, D_par, v_ic, kappa, mu, g_vectors):
    # 1. Calculate tau_1 (Eq 8) - The effect of dispersion on tortuosity
    if kappa < 1e-5:
        tau_1 = 1.0/3.0 # Isotropic limit
    else:
        sqrt_k = np.sqrt(kappa)
        # Eq 8: -1/(2k) + 1/(2 * sqrt(k) * Dawson(sqrt(k)))
        tau_1 = (-1.0 / (2.0 * kappa)) + (1.0 / (2.0 * sqrt_k * dawsn(sqrt_k)))
        
    # 2. Calculate parallel and perpendicular diffusivities for the EC tensor (Eqs 5 & 6)
    # d_par_prime = d_par * (1 - v_ic * (1 - tau_1))
    # d_perp_prime = d_par * (1 - v_ic * ((1 + tau_1) / 2))
    d_par_prime = D_par * (1.0 - v_ic * (1.0 - tau_1))
    d_perp_prime = D_par * (1.0 - v_ic * ((1.0 + tau_1) / 2.0))
    
    # 3. Calculate Signal for this cylindrically symmetric tensor
    # The tensor is aligned with 'mu'.
    # ADC = D_perp + (D_par - D_perp) * (g . mu)^2
    mu = mu / np.linalg.norm(mu)
    g_dot_mu = g_vectors @ mu
    
    ADC = d_perp_prime + (d_par_prime - d_perp_prime) * (g_dot_mu**2)
    
    S_extra = np.exp(-b * ADC)
    return S_extra