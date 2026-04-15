"""
ruwe.py  —  Part 3: RUWE Calculation


    RUWE = sqrt[ Σᵢ (rᵢ / σᵢ)² / (N - 5) ]

r = al_model - H_np @ al_model
      = (I - H_np) @ al_model
"""

import os
os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")

import numpy as np
import pytensor.tensor as pt

GAIA_EPOCH_OFFSET_DAYS = 2192.0
DAYS_PER_YEAR          = 365.25


def sigma_al(g_mag: float) -> float:
    """
    gaia AL single-measurement uncertainty as a function of G magnitude.

    σ_AL [mas] = sqrt( σ_floor² + (10^(0.2*(G_eff - 12.09)))² )

    σ_floor = 0.029 mas (calibration noise floor).
    G_eff   = max(G, 13) to avoid underflow for very bright stars.

    """
    g_eff      = max(float(g_mag), 13.0)
    sigma_phot = 10.0 ** (0.2 * (g_eff - 12.09))   
    sigma_floor = 0.029                              
    return float(np.sqrt(sigma_floor**2 + sigma_phot**2))


def _design_matrix_np(t_tcb_days: np.ndarray,
                       scan_angle: np.ndarray,
                       parallax_factor: np.ndarray) -> np.ndarray:
    """
    build the N×5 astrometric design matrix in numpy.

    rows: A[i] = [sin(ψᵢ), cos(ψᵢ), f(tᵢ), tᵢ·sin(ψᵢ), tᵢ·cos(ψᵢ)]

    Columns encode sensitivity to (Δα₀, Δδ₀, ϖ, μ_α, μ_δ).
    """
    t_yr    = (t_tcb_days - GAIA_EPOCH_OFFSET_DAYS) / DAYS_PER_YEAR
    sin_psi = np.sin(scan_angle)
    cos_psi = np.cos(scan_angle)

    A = np.column_stack([
        sin_psi,
        cos_psi,
        parallax_factor,
        t_yr * sin_psi,
        t_yr * cos_psi,
    ])
    return A


def _hat_matrix_np(A: np.ndarray) -> np.ndarray:
    """
    OLS hat matrix H = A (AᵀA)⁻¹ Aᵀ  

    residuals = (I - H) y,  so we precompute (I - H) once.
    """
    ATA      = A.T @ A                    
    ATA_inv  = np.linalg.inv(ATA)            
    H        = A @ ATA_inv @ A.T             
    return H


def precompute_projection(t_tcb_days: np.ndarray,
                           scan_angle: np.ndarray,
                           parallax_factor: np.ndarray) -> np.ndarray:
    """
    precompute the residual projection matrix (I - H) in numpy.


    parameters
    t_tcb_days, scan_angle, parallax_factor 

    
    returns
    I_minus_H : np.ndarray, shape (N, N)
    """
    A         = _design_matrix_np(t_tcb_days, scan_angle, parallax_factor)
    H         = _hat_matrix_np(A)
    I_minus_H = np.eye(len(t_tcb_days)) - H
    return I_minus_H


def compute_ruwe(
    al_model,
    I_minus_H: np.ndarray,
    g_mag: float,
    N: int,
):
    """
    model-predicted RUWE 

        r         = (I - H) · al_model          [residuals, mas]
        RUWE      = sqrt( Σ (rᵢ/σ)² / (N - 5) )

    parameters
    al_model  : Synthetic AL positions from single_star() or planet_model().
    I_minus_H : Precomputed residual projection matrix from precompute_projection().
    g_mag     : Gaia G magnitude (used for σ_AL).
    N         : Number of AL observations.

                
    returns
    ruwe_model : PyTensor scalar

    compute residuals = (I - H) @ al_model without ANY pt matrix ops.
    PyTensor generates broken C code (ssize_t narrowing) for pt.dot and all 2D tensor multiplications 
    expand the matrix-vector product as a Python-level list comprehension over rows of I_minus_H. Each row is a fixed numpy 1D
    array, so (row * al_model) is elementwise scalar multiplication — PyTensor only emits scalar C ops which compile fine.
    pt.stack then assembles the N scalars into a vector.
    """

    residuals = pt.stack([
        pt.sum(pt.as_tensor_variable(row) * al_model)
        for row in I_minus_H.astype("float64")
    ])

    sigma        = sigma_al(g_mag)                  
    chi2_red     = pt.sum((residuals / sigma) ** 2) / (N - 5)
    ruwe_model   = pt.sqrt(chi2_red)
    return ruwe_model


# test

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    SOURCE_ID = "927713095040"
    params = query_companion(SOURCE_ID)

    t_np  = params["t_obs"]
    psi_np = params["scan_angle"]
    f_np   = params["parallax_factor"]
    N      = len(t_np)

    I_minus_H = precompute_projection(t_np, psi_np, f_np)
    print(f"Projection matrix (I-H) shape: {I_minus_H.shape}")
    print(f"σ_AL at G={params['g_mag']:.2f}:  {sigma_al(params['g_mag']):.4f} mas")

    t_pt   = pt.as_tensor_variable(t_np.astype("float64"))
    psi_pt = pt.as_tensor_variable(psi_np.astype("float64"))
    f_pt   = pt.as_tensor_variable(f_np.astype("float64"))

    al_star = single_star(
        t_tcb_days=t_pt, scan_angle=psi_pt, parallax_factor=f_pt,
        ra_off=0., dec_off=0.,
        pmra=params["pmra"], pmdec=params["pmdec"],
        parallax_mas=params["parallax"],
    )
    ruwe_star = compute_ruwe(al_star, I_minus_H, params["g_mag"], N)
    print(f"\nSingle-star model RUWE  : {ruwe_star.eval():.4f}")
    print(f"Observed RUWE (Gaia DR3): {params['ruwe']:.4f}")
    print("(Single-star RUWE ≈ 0 on noise-free synthetic data — expected)")

    al_planet = planet_model(
        t_tcb_days=t_pt, scan_angle=psi_pt, parallax_factor=f_pt,
        ra_off=0., dec_off=0.,
        pmra=params["pmra"], pmdec=params["pmdec"],
        parallax_mas=params["parallax"],
        semimajor_au=5.2, inclination_deg=45., eccentricity=0.05,
        Omega_deg=100., omega_deg=30., t_p_yr=0.,
        Mp=1e-3, Ms=1.0, lum_ratio=0.0,
    )
    ruwe_planet = compute_ruwe(al_planet, I_minus_H, params["g_mag"], N)
    print(f"Planet model RUWE       : {ruwe_planet.eval():.4f}")

    # gradient check 
    print("\nGradient check (dRUWE/dMp):")
    Mp_sym = pt.dscalar("Mp")
    ruwe_expr = compute_ruwe(
        planet_model(
            t_tcb_days=t_pt, scan_angle=psi_pt, parallax_factor=f_pt,
            ra_off=0., dec_off=0.,
            pmra=params["pmra"], pmdec=params["pmdec"],
            parallax_mas=params["parallax"],
            semimajor_au=5.2, inclination_deg=45., eccentricity=0.05,
            Omega_deg=100., omega_deg=30., t_p_yr=0.,
            Mp=Mp_sym, Ms=1.0, lum_ratio=0.0,
        ),
        I_minus_H, params["g_mag"], N,
    )
    import pytensor
    grad_fn = pytensor.function([Mp_sym], pt.grad(ruwe_expr, Mp_sym))
    print(f"  dRUWE/dMp at Mp=1e-3 Msun = {grad_fn(1e-3):.6f}")
    print("  Autodiff through RUWE → Kepler solver: OK")