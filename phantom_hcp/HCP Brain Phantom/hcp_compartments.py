import numpy as np
from scipy.special import dawsn

def calculate_ball_signal(b, D_iso, g_vectors):
    """Isotropic water signal"""
    return np.full(g_vectors.shape[0], np.exp(-b * D_iso))

def calculate_watson_stick_signal(b, D_stick, mu_vector, kappa, g_vectors, sample_orientations):
    """Intracellular signal (Watson distribution of sticks)"""
    # Normalize mu
    mu_vector = mu_vector / (np.linalg.norm(mu_vector) + 1e-10)
    
    # Calculate weights for orientations
    mu_dot_n = sample_orientations @ mu_vector
    weights = np.exp(kappa * mu_dot_n**2)
    weights /= np.sum(weights)
    
    # Calculate signal for all samples
    g_dot_n_matrix = g_vectors @ sample_orientations.T 
    stick_signals_matrix = np.exp(-b[:, None] * D_stick * (g_dot_n_matrix**2))
    
    return stick_signals_matrix @ weights

def calculate_noddi_extra_signal(b, D_par, v_ic, kappa, mu, g_vectors):
    """Extracellular signal using tortuosity model"""
    if kappa < 1e-5:
        tau_1 = 1.0/3.0
    else:
        sqrt_k = np.sqrt(kappa)
        tau_1 = (-1.0 / (2.0 * kappa)) + (1.0 / (2.0 * sqrt_k * dawsn(sqrt_k)))
        
    d_par_prime = D_par * (1.0 - v_ic * (1.0 - tau_1))
    d_perp_prime = D_par * (1.0 - v_ic * ((1.0 + tau_1) / 2.0))
    
    mu = mu / (np.linalg.norm(mu) + 1e-10)
    g_dot_mu = g_vectors @ mu
    ADC = d_perp_prime + (d_par_prime - d_perp_prime) * (g_dot_mu**2)
    
    return np.exp(-b * ADC)