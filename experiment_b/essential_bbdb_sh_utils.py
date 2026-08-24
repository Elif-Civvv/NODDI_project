import numpy as np
import scipy.special as sps
import matplotlib.pyplot as plt
# We must import this for the 3D plotting
from mpl_toolkits.mplot3d import Axes3D

# --- Helper function ---
def cartesian_to_spherical(x, y, z):
    """
    Converts Cartesian coordinates (x, y, z) to spherical
    coordinates (r, theta, phi).
    theta is the polar (colatitudinal) angle [0, pi].
    phi is the azimuthal angle [0, 2*pi].
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    # Handle the r=0 case to avoid division by zero
    r_safe = np.where(r == 0, 1e-15, r)
    theta = np.arccos(np.clip(z / r_safe, -1.0, 1.0)) # clip for numerical stability
    phi = np.arctan2(y, x)
    return r, theta, phi
# ------------------------

def real_sh_tournier(sh_order, theta, phi,
                     full_basis=False):
    """ Compute real spherical harmonics (MRtrix3 convention)"""
    m, n = sph_harm_ind_list(sh_order, full_basis)

    phi = np.reshape(phi, [-1, 1])
    theta = np.reshape(theta, [-1, 1])

    sh = sps.sph_harm(np.abs(m), n, phi, theta)
    real_sh = np.where(m < 0, sh.imag, sh.real)
    real_sh *= np.where(m == 0, 1., np.sqrt(2))

    return real_sh, m, n


def sph_harm_ind_list(sh_order, full_basis=False):
    """ Returns the degree (m) and order (n)"""
    if full_basis:
        n_range = np.arange(0, sh_order + 1, dtype=int)
        ncoef = int(np.sum(2 * n_range + 1))
    else:
        if sh_order % 2 != 0:
            raise ValueError('sh_order must be an even integer >= 0')
        n_range = np.arange(0, sh_order + 1, 2, dtype=int)
        ncoef = int((sh_order + 2) * (sh_order + 1) // 2)

    n_list = np.repeat(n_range, n_range * 2 + 1)
    offset = 0
    m_list = np.empty(ncoef, 'int')
    for ii in n_range:
        m_list[offset:offset + 2 * ii + 1] = np.arange(-ii, ii + 1)
        offset = offset + 2 * ii + 1

    return m_list, n_list

def find_sh_basis(coords, lmax, normalise_basis=True):
    """Outputs spherical harmonic basis functions (matrix)"""
    if coords.shape[0] == 3 and coords.shape[1] != 3:
        coords = coords.T 
    elif coords.shape[1] != 3:
         raise ValueError("Coords shape must be (N, 3) or (3, N)")

    _, th, phi = cartesian_to_spherical(coords[:, 0], coords[:, 1], coords[:, 2])
    
    basis, m, l = real_sh_tournier(lmax, th, phi)
    
    if normalise_basis:
        basis = basis / basis[0,0] 
    return basis, m, l

def find_sh_coeffs(signal, coords, lmax, normalise_basis=True, lambda_reg=0.0):
    
    # Design Matrix (Basis)
    X, m, l = find_sh_basis(coords, lmax, normalise_basis)
    
    if signal.shape[0] != X.shape[0]:
         raise ValueError(f"Signal size ({signal.shape[0]}) and coords size ({X.shape[0]}) do not match.")

    #  Least Squares Solution
    if lambda_reg > 0:
        # Regularized Least Squares (Laplace-Beltrami Regularization)
        # Formula: beta = (X^T X + lambda * L)^-1 X^T y
        
        XtX = X.T @ X
        
        # Create Smoothing Matrix L
        # Penalty increases with harmonic order l: L_ii = l^2 * (l + 1)^2
        # This suppresses high-frequency spikes (high l) while keeping the main shape (low l).
        regularization_matrix = np.diag(l**2 * (l + 1)**2)
        
        # Calculate regularized inverse
        inv_term = np.linalg.inv(XtX + lambda_reg * regularization_matrix)
        coeffs = inv_term @ X.T @ signal
        
    else:
        # Standard Ordinary Least Squares (OLS)
        mat_inv = np.linalg.pinv(X.T)
        coeffs = signal @ mat_inv

    return coeffs, m, l


def visualizeSHcoeffs(coeffs, lmax, npoints=100, ax=None, title=""):
    """
    Visualise FOD described by spherical harmonics on a given 3D axis.
    """
    
    # --- 1. Setup the axis if one isn't provided ---
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        show_plot = True
    else:
        show_plot = False 

    # --- 2. Create the spherical mesh ---
    theta = np.linspace(0, np.pi, npoints)
    phi = np.linspace(0, 2 * np.pi, npoints)
    theta, phi = np.meshgrid(theta, phi)

    X_flat = (np.sin(theta) * np.cos(phi)).flatten()
    Y_flat = (np.sin(theta) * np.sin(phi)).flatten()
    Z_flat = (np.cos(theta)).flatten()

    # --- 3. Calculate SH basis on the mesh ---
    coords_sphere = np.vstack([X_flat, Y_flat, Z_flat]).T
    SHsphere, m, l = find_sh_basis(coords_sphere, lmax)

    # --- 4. Get radii from coefficients ---
    radii = np.dot(SHsphere, coeffs)
    radii = np.abs(radii) 

    # --- 5. Calculate 3D glyph coordinates ---
    donut = np.vstack([radii * X_flat, radii * Y_flat, radii * Z_flat]).T

    x1 = donut[:, 0].reshape((npoints, npoints))
    y1 = donut[:, 1].reshape((npoints, npoints))
    z1 = donut[:, 2].reshape((npoints, npoints))

    # --- 6. Get colors (color by direction) ---
    cc0 = donut[:,0].reshape(x1.shape)
    cc1 = donut[:,1].reshape(x1.shape)
    cc2 = donut[:,2].reshape(x1.shape)
    cc = np.concatenate((cc0[:,:,np.newaxis],cc1[:,:,np.newaxis]),axis=2)
    cc = np.concatenate((cc,cc2[:,:,np.newaxis]),axis=2)
    cc = np.abs(cc)
    max_c = np.max(cc, axis=2)
    max_c_safe = np.where(max_c == 0, 1, max_c)
    cc /= np.tile(np.expand_dims(max_c_safe,2),(1,1,3))

    # --- 7. Plot the surface on the given ax ---
    ax.plot_surface(x1, y1, z1, facecolors=cc, rstride=1, cstride=1,
                    antialiased=True)

    # --- 8. Set limits and labels ---
    
    # REMOVED max_amplitude calculation
    # max_amplitude = np.max(radii)
    # if max_amplitude == 0: max_amplitude = 1
        
    # SET fixed axes from -1 to 1 for all plots
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    
    ax.set_box_aspect([1, 1, 1]) 
    
    # Add axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    ax.set_title(title, fontsize=10)
    ax.view_init(elev=20, azim=30)

def fibonacci_sphere(samples=1):
    points = []
    phi = np.pi * (3. - np.sqrt(5.)) 
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2
        radius = np.sqrt(1 - y * y)
        theta = phi * i
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        points.append([x, y, z])
    return np.array(points)