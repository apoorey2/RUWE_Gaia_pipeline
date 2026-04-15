import numpy as np
import pandas as pd
from astroquery.gaia import Gaia
from astropy.time import Time
from astropy.coordinates import get_body_barycentric, ICRS, SkyCoord
import astropy.units as u
from gaiaunlimited.scanninglaw import GaiaScanningLaw


#correlation columns needed to construct the 5×5 covariance matrix.
_CORR_COLS = [
    "ra_dec_corr",
    "ra_parallax_corr",
    "ra_pmra_corr",
    "ra_pmdec_corr",
    "dec_parallax_corr",
    "dec_pmra_corr",
    "dec_pmdec_corr",
    "parallax_pmra_corr",
    "parallax_pmdec_corr",
    "pmra_pmdec_corr",
]

_ADQL_QUERY = """
SELECT
    source_id,
    ra, ra_error,
    dec, dec_error,
    parallax, parallax_error,
    pmra, pmra_error,
    pmdec, pmdec_error,
    ruwe,
    phot_g_mean_mag,
    astrometric_n_good_obs_al,
    astrometric_chi2_al,
    {corr_cols}
FROM gaiadr3.gaia_source
WHERE source_id = {source_id}
""".format(
    corr_cols=",\n    ".join(_CORR_COLS),
    source_id="{source_id}",         
)


def _build_covariance(row: dict) -> np.ndarray:
    """
    construct the 5×5 astrometric covariance matrix from the
    individual errors and pairwise correlation coefficients stored
    in the Gaia archive.

    parameters : (ra, dec, parallax, pmra, pmdec)
    Units            (mas, mas, mas, mas/yr, mas/yr)

    ra_error in the archive is in mas 
    """
    params  = ["ra", "dec", "parallax", "pmra", "pmdec"]
    errors  = np.array([row[f"{p}_error"] for p in params])  

    # correlation matrix (symmetric, diagonal = 1)
    corr_map = {
        (0, 1): row["ra_dec_corr"],
        (0, 2): row["ra_parallax_corr"],
        (0, 3): row["ra_pmra_corr"],
        (0, 4): row["ra_pmdec_corr"],
        (1, 2): row["dec_parallax_corr"],
        (1, 3): row["dec_pmra_corr"],
        (1, 4): row["dec_pmdec_corr"],
        (2, 3): row["parallax_pmra_corr"],
        (2, 4): row["parallax_pmdec_corr"],
        (3, 4): row["pmra_pmdec_corr"],
    }

    C = np.eye(5)
    for (i, j), rho in corr_map.items():
        C[i, j] = rho
        C[j, i] = rho

    # convert correlation matrix to covariance matrix: Σ_ij = ρ_ij * σ_i * σ_j
    cov = C * np.outer(errors, errors)
    return cov


def query_gaia_archive(source_id: int | str) -> dict:
    """
    query the Gaia DR3 archive for astrometric + photometric parameters.

    parameters
    source_id 

    returns
    dict with keys:
        source_id, ra, dec, parallax, pmra, pmdec,
        ra_error, dec_error, parallax_error, pmra_error, pmdec_error,
        ruwe, g_mag, astrometric_n_good_obs_al, astrometric_chi2_al, 
        covariance_matrix  (5×5 numpy array)
    """
    adql = _ADQL_QUERY.format(source_id=int(source_id))
    job  = Gaia.launch_job_async(adql)
    tbl  = job.get_results()

    row = {col: tbl[col][0] for col in tbl.colnames}

    cov = _build_covariance(row)

    params = {
        "source_id"                 : int(source_id),
        "ra"                        : float(row["ra"]),
        "dec"                       : float(row["dec"]),
        "parallax"                  : float(row["parallax"]),
        "pmra"                      : float(row["pmra"]),
        "pmdec"                     : float(row["pmdec"]),
        "ra_error"                  : float(row["ra_error"]),
        "dec_error"                 : float(row["dec_error"]),
        "parallax_error"            : float(row["parallax_error"]),
        "pmra_error"                : float(row["pmra_error"]),
        "pmdec_error"               : float(row["pmdec_error"]),
        "ruwe"                      : float(row["ruwe"]),
        "g_mag"                     : float(row["phot_g_mean_mag"]),
        "astrometric_n_good_obs_al" : int(row["astrometric_n_good_obs_al"]),
        "astrometric_chi2_al"       : float(row["astrometric_chi2_al"]),
        "covariance_matrix"         : cov,
    }
    return params


def compute_parallax_factor(t_tcb: np.ndarray, ra_deg: float, dec_deg: float) -> np.ndarray:
    """
    compute the Gaia along-scan (AL) parallax factor

    AL parallax factor:

        f(t) = sin(ψ) · X_bary(t) + cos(ψ) · Y_bary(t)
        
        f(t) = cos(δ) · cos(α) · X⊕(t)
             + cos(δ) · sin(α) · Y⊕(t)
             + sin(δ) · Z⊕(t)

    X⊕, Y⊕, Z⊕ : barycentric position of the Earth (≈ Gaia) in AU
    (α, δ) : star's RA/Dec in radians.  

    result is dimensionless. 
    multiplying by parallax gives the AL displacement in mas at each epoch.

    parameters
    t_tcb : np.ndarray
        observation times as Gaia Julian days

    ra_deg, dec_deg : float
        target ICRS coordinates in degrees.

    returns
    parallax_factor : np.ndarray  
    """

    # Convert Gaia days to standard Julian Date
    GAIA_TCB_ORIGIN_JD = 2455197.5
    jd_tcb = t_tcb + GAIA_TCB_ORIGIN_JD

    # astropy Time in TCB scale
    times = Time(jd_tcb, format="jd", scale="tcb")

    # barycentric position of Earth in AU 
    earth_bary = get_body_barycentric("earth", times)   
    X = earth_bary.x.to(u.au).value   
    Y = earth_bary.y.to(u.au).value
    Z = earth_bary.z.to(u.au).value

    # direction cosines of the target star
    ra_rad  = np.deg2rad(ra_deg)
    dec_rad = np.deg2rad(dec_deg)
    l_x = np.cos(dec_rad) * np.cos(ra_rad)
    l_y = np.cos(dec_rad) * np.sin(ra_rad)
    l_z = np.sin(dec_rad)

    # AL parallax factor 
    parallax_factor = l_x * X + l_y * Y + l_z * Z
    return parallax_factor


def query_scanning_law(ra_deg: float, dec_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    return the Gaia DR3 scanning-law time series for a sky position.

    parameters
    ra_deg, dec_deg : float
        ICRS coordinates of the target in degrees.

    returns
    -------
    t_tcb : np.ndarray
        observation times in Gaia's time system
    scan_angle : np.ndarray
        scan-position angle in radians 
    parallax_factor : np.ndarray
        along-scan parallax factor, dimensionless.

    gaiaunlimited returns individual CCD rows.
    """

    sl = GaiaScanningLaw()

    """
    GaiaScanningLaw.query(ra_deg, dec_deg) returns a LIST of pandas
    one DataFrame per Gaia field-of-view pass (two FoVs per transit: preceding and following).  
    concatenate them
    """
    result = sl.query(ra_deg, dec_deg)

    spin_phase = np.asarray(result[0], dtype=float)
    t_tcb      = np.asarray(result[1], dtype=float)
    
    # result[0] and result[1] may differ in length by 1 due to how
    # gaiaunlimited pads its output arrays. Trim to the shorter length.
    n = min(len(spin_phase), len(t_tcb))
    spin_phase = spin_phase[:n]
    t_tcb      = t_tcb[:n]

    # wrap cumulative spin phase → instantaneous scan position angle ψ ∈ [0, 2π]
    scan_angle = spin_phase % (2 * np.pi)

    parallax_factor = compute_parallax_factor(t_tcb, ra_deg, dec_deg)
    
    # sort chronologically
    order = np.argsort(t_tcb)
    return t_tcb[order], scan_angle[order], parallax_factor[order]

def query_companion(source_id: int | str) -> dict:
    """
    combines the archive query (astrometry, photometry, RUWE, covariance)
    with the scanning-law query (times, scan angles, parallax factors).

    parameters
    source_id 

    returns
    dict with all fields documented in query_gaia_archive() plus:
        t_obs            : observation times 
        scan_angle       : ψ(t) in radians
        parallax_factor  : f(t), dimensionless AL parallax factor
    """
    print(f"[1/2] Querying Gaia DR3 archive for source_id = {source_id} ...")
    params = query_gaia_archive(source_id)
    print(f"      RA={params['ra']:.5f}°  Dec={params['dec']:.5f}°  "
          f"parallax={params['parallax']:.4f} mas  RUWE={params['ruwe']:.3f}")

    print("[2/2] Querying Gaia scanning law (gaiaunlimited) ...")
    t_obs, psi, f = query_scanning_law(params["ra"], params["dec"])
    print(f"      {len(t_obs)} CCD observations retrieved.")

    params["t_obs"]           = t_obs
    params["scan_angle"]      = psi
    params["parallax_factor"] = f

    return params


def print_summary(params: dict) -> None:
    """Print a tidy summary of the query results."""
    print("\n" + "=" * 60)
    print(f"  Gaia DR3  source_id = {params['source_id']}")
    print("=" * 60)
    print(f"  RA            = {params['ra']:.6f} deg")
    print(f"  Dec           = {params['dec']:.6f} deg")
    print(f"  Parallax      = {params['parallax']:.4f} ± {params['parallax_error']:.4f} mas")
    print(f"  pmRA          = {params['pmra']:.4f} ± {params['pmra_error']:.4f} mas/yr")
    print(f"  pmDec         = {params['pmdec']:.4f} ± {params['pmdec_error']:.4f} mas/yr")
    print(f"  RUWE          = {params['ruwe']:.4f}")
    print(f"  G mag         = {params['g_mag']:.3f}")
    print(f"  N_obs (AL)    = {params['astrometric_n_good_obs_al']}")
    print()
    print("  Covariance matrix (ra, dec, plx, pmra, pmdec) [mas²]:")
    cov = params["covariance_matrix"]
    for row in cov:
        print("    " + "  ".join(f"{v:+10.5f}" for v in row))
    print()
    n = len(params["t_obs"])
    print(f"  Scanning law: {n} CCD observations")
    print(f"    t_obs[0:3]            = {params['t_obs'][:3]}")
    print(f"    scan_angle[0:3] [rad] = {params['scan_angle'][:3]}")
    print(f"    scan_angle[0:3] [deg] = {np.rad2deg(params['scan_angle'][:3])}")
    print(f"    parallax_factor[0:3]  = {params['parallax_factor'][:3]}")
    print("=" * 60 + "\n")





"""
forward_model.py  —  Part 2: Forward Modeling Gaia Along-Scan Positions
PyTensor-compatible implementation of the single-star and planet-companion AL forward models.

part 1 returns t_obs 

AL observation equation
  AL(t) = (ra_off + pmra·t_yr) · sin(ψ)
         + (dec_off + pmdec·t_yr) · cos(ψ)
         + ϖ · f(t)
"""

import numpy as np
import pytensor.tensor as pt


GAIA_EPOCH_OFFSET_DAYS = 2192.0   # TCB days from 2010-01-01 to J2016.0
DAYS_PER_YEAR          = 365.25
TWO_PI                 = 2.0 * np.pi


def tcb_days_to_yr2016(t_tcb_days):
    return (t_tcb_days - GAIA_EPOCH_OFFSET_DAYS) / DAYS_PER_YEAR


def _eccentric_anomaly_pt(t_yr, period_yr, eccentricity, t_p_yr, steps=6):
    """
    solve Kepler's equation for eccentric anomaly E via Newton–Raphson.

    M = 2π · (t - t_p) / P   (mean anomaly)
    E - e·sin(E) = M          (Kepler's equation)
    """
    M = pt.mod((t_yr - t_p_yr) / period_yr * TWO_PI, TWO_PI)

    E = M + 0.0  # copy avoids mutation of M
    for i in range(steps):
        E = E - (E - eccentricity * pt.sin(E) - M) / (1.0 - eccentricity * pt.cos(E))
    return E


def _orbital_xy_pt(t_yr, period_yr, eccentricity, t_p_yr):
    """
    compute dimensionless Thiele-Innes orbital coordinates X, Y.

      X = cos(E) - e
      Y = sqrt(1 - e²) · sin(E)

    """
    E = _eccentric_anomaly_pt(t_yr, period_yr, eccentricity, t_p_yr)
    X = pt.cos(E) - eccentricity
    Y = pt.sqrt(1.0 - eccentricity ** 2) * pt.sin(E)
    return X, Y


def _thiele_innes_pt(semimajor_mas, inclination_deg, Omega_deg, omega_deg):
    """
    compute Thiele-Innes coefficients A, B, F, G in mas.

    """
    i = pt.deg2rad(inclination_deg)
    W = pt.deg2rad(Omega_deg)
    w = pt.deg2rad(omega_deg)

    A = semimajor_mas * ( pt.cos(w) * pt.cos(W) - pt.sin(w) * pt.sin(W) * pt.cos(i))
    B = semimajor_mas * ( pt.cos(w) * pt.sin(W) + pt.sin(w) * pt.cos(W) * pt.cos(i))
    F = semimajor_mas * (-pt.sin(w) * pt.cos(W) - pt.cos(w) * pt.sin(W) * pt.cos(i))
    G = semimajor_mas * (-pt.sin(w) * pt.sin(W) + pt.cos(w) * pt.cos(W) * pt.cos(i))
    return A, B, F, G


def _photocenter_offset_pt(
    t_yr, semimajor_mas, period_yr, eccentricity,
    inclination_deg, Omega_deg, omega_deg, t_p_yr,
    mass_ratio, lum_ratio
):
    """
    photocenter offset (Δα*, Δδ) in mas due to the companion's orbital motion.

      Δα* = (B·X + G·Y) · w_lum
      Δδ  = (A·X + F·Y) · w_lum
      w_lum = |q - l| / [(1 + l)(1 + q)]

      
    parameters

    t_yr          : observation times in years since J2016.0  (pt tensor)
    semimajor_mas : semi-major axis in mas = semimajor_au * parallax_mas
    period_yr     : orbital period in years  (Kepler's 3rd law)
    eccentricity  : orbital eccentricity
    inclination_deg, Omega_deg, omega_deg : orbital angles in degrees
    t_p_yr        : time of periastron in years since J2016.0
    mass_ratio    : q = Mp / Ms
    lum_ratio     : l = L_planet / L_star  (≈ 0 for dark companions)

    
    returns

    dra_mas, ddec_mas
    """
    A, B, F, G = _thiele_innes_pt(semimajor_mas, inclination_deg, Omega_deg, omega_deg)
    X, Y       = _orbital_xy_pt(t_yr, period_yr, eccentricity, t_p_yr)

    w_eff = mass_ratio / (1.0 + mass_ratio) - lum_ratio / (1.0 + lum_ratio)

    dra_mas  = (B * X + G * Y) * w_eff
    ddec_mas = (A * X + F * Y) * w_eff
    return dra_mas, ddec_mas


def single_star(
    t_tcb_days,
    scan_angle,
    parallax_factor,
    ra_off,
    dec_off,
    pmra,
    pmdec,
    parallax_mas,
):
    """
    Gaia AL positions for a single-star model.

    AL(t) = (ra_off + pmra·t_yr) · sin(ψ)
           + (dec_off + pmdec·t_yr) · cos(ψ)
           + ϖ · f(t)

           
    parameters
  
    t_tcb_days      : Gaia TCB days since 2010-01-01  (numpy array or pt tensor)
    scan_angle      : scan-position angle ψ(t) in radians  (numpy array or pt tensor)
    parallax_factor : AL parallax factor f(t), dimensionless  (numpy array or pt tensor)
    ra_off          : RA offset at J2016.0 (mas)  — pt scalar in PyMC context
    dec_off         : Dec offset at J2016.0 (mas)
    pmra            : proper motion in RA·cos(Dec) (mas/yr)
    pmdec           : proper motion in Dec (mas/yr)
    parallax_mas    : parallax ϖ (mas)

    
    returns
    -------
    AL_star
    """
    t_yr = tcb_days_to_yr2016(t_tcb_days)

    dra  = ra_off  + pmra  * t_yr
    ddec = dec_off + pmdec * t_yr

    AL_star = dra * pt.sin(scan_angle) + ddec * pt.cos(scan_angle) + parallax_mas * parallax_factor
    return AL_star



def planet_model(
    t_tcb_days,
    scan_angle,
    parallax_factor,
    ra_off,
    dec_off,
    pmra,
    pmdec,
    parallax_mas,
    semimajor_au,
    inclination_deg,
    eccentricity,
    Omega_deg,
    omega_deg,
    t_p_yr,
    Mp,
    Ms,
    lum_ratio=0.0,
):
    """
    Gaia AL positions for a star + planet-companion model.

    AL(t) = (ra_off + pmra·t_yr + Δα*_planet(t)) · sin(ψ)
           + (dec_off + pmdec·t_yr + Δδ_planet(t)) · cos(ψ)
           + ϖ · f(t)

    parameters

    t_tcb_days      : Gaia TCB days since 2010-01-01  (numpy array or pt tensor)
    scan_angle      : ψ(t) in radians  (numpy array or pt tensor)
    parallax_factor : f(t), dimensionless  (numpy array or pt tensor)
    ra_off          : RA offset at J2016.0 (mas)
    dec_off         : Dec offset at J2016.0 (mas)
    pmra            : proper motion in RA·cos(Dec) (mas/yr)
    pmdec           : proper motion in Dec (mas/yr)
    parallax_mas    : parallax ϖ (mas)

    Orbital parameters:
    semimajor_au    : semi-major axis of companion orbit (AU)
    inclination_deg : orbital inclination (degrees)
    eccentricity    : orbital eccentricity [0, 1)
    Omega_deg       : longitude of ascending node (degrees)
    omega_deg       : argument of periastron (degrees)
    t_p_yr          : time of periastron passage (years since J2016.0)
    Mp              : companion mass (solar masses)
    Ms              : host star mass (solar masses)
    lum_ratio       : L_companion / L_star  (default 0 for dark companion)

    returns
    -------
    AL_planet 
    """
    t_yr = tcb_days_to_yr2016(t_tcb_days)

    # orbital period from Kepler's third law: P² = a³ / (M_total)
    period_yr = pt.sqrt(semimajor_au ** 3 / (Mp + Ms))

    # semi-major axis in mas: a_mas = a_AU × ϖ
    semimajor_mas = semimajor_au * parallax_mas

    # mass ratio
    q = Mp / Ms

    # photocenter offsets from orbital motion
    dra_planet, ddec_planet = _photocenter_offset_pt(
        t_yr, semimajor_mas, period_yr, eccentricity,
        inclination_deg, Omega_deg, omega_deg, t_p_yr,
        q, lum_ratio
    )

    dra  = ra_off  + pmra  * t_yr + dra_planet
    ddec = dec_off + pmdec * t_yr + ddec_planet

    AL_planet = dra * pt.sin(scan_angle) + ddec * pt.cos(scan_angle) + parallax_mas * parallax_factor
    return AL_planet








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








"""
inference.py  —  Part 4: Bayesian Inference with PyMC
Fits planetary companion parameters to Gaia RUWE using PyMC.

Likelihood
    RUWE_obs ~ Normal(RUWE_model(θ), σ_RUWE)


parameters inferred

astrometric (5-parameter solution offsets):
    ra_off   : RA offset at J2016.0 [mas]
    dec_off  : Dec offset at J2016.0 [mas]
    pmra     : proper motion in RA·cos(Dec) [mas/yr]
    pmdec    : proper motion in Dec [mas/yr]
    parallax : parallax ϖ [mas]

orbital:
    log_a    : log10(semi-major axis [AU])
    inc      : inclination [deg]
    ecc      : eccentricity [0, 1)
    Omega    : longitude of ascending node [deg]
    omega    : argument of periastron [deg]
    tp       : time of periastron [yr since J2016.0]
    log_Mp   : log10(planet mass [Mjup])

noise:
    log_sigma0 : log10(RUWE jitter term)

"""

import os
os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")

import numpy as np
import pytensor.tensor as pt
import pymc as pm
import arviz as az


MJUP_MSUN = 9.547919e-4          # 1 Jupiter mass in solar masses
RUWE_ERR  = 0.01                 # approximate uncertainty on published RUWE
                                  # (Lindegren 2021: ~1% for DR3)



def build_model(params: dict, Ms: float = 1.0, lum_ratio: float = 0.0):
    """
    build and return the PyMC model for RUWE-based companion inference.

    parameters
    params : dict
        Output of query_companion() from Part 1. Must contain:
        t_obs, scan_angle, parallax_factor, pmra, pmdec, parallax,
        pmra_error, pmdec_error, parallax_error, ra_error, dec_error,
        ruwe, g_mag.
    Ms : float
        Host star mass in solar masses. Default 1.0 (can be refined from
        spectroscopy or isochrone fitting).
    lum_ratio : float
        L_companion / L_star. Default 0 (dark companion).

        

    returns
    model : pm.Model
    coords : dict 
    """

    t_np   = params["t_obs"].astype("float64")
    psi_np = params["scan_angle"].astype("float64")
    f_np   = params["parallax_factor"].astype("float64")
    N      = len(t_np)


    I_minus_H = precompute_projection(t_np, psi_np, f_np)

    t_pt   = pt.as_tensor_variable(t_np)
    psi_pt = pt.as_tensor_variable(psi_np)
    f_pt   = pt.as_tensor_variable(f_np)

    ruwe_obs = float(params["ruwe"])
    g_mag    = float(params["g_mag"])

    with pm.Model() as model:

   
        # astrometric priors

        ra_off = pm.Normal(
            "ra_off",
            mu=0.0,
            sigma=5.0 * params["ra_error"],
        )
        dec_off = pm.Normal(
            "dec_off",
            mu=0.0,
            sigma=5.0 * params["dec_error"],
        )
        pmra = pm.Normal(
            "pmra",
            mu=params["pmra"],
            sigma=5.0 * params["pmra_error"],
        )
        pmdec = pm.Normal(
            "pmdec",
            mu=params["pmdec"],
            sigma=5.0 * params["pmdec_error"],
        )
        parallax = pm.Normal(
            "parallax",
            mu=params["parallax"],
            sigma=5.0 * params["parallax_error"],
        )


        # orbital priors

        log_a = pm.Uniform("log_a", lower=-1.0, upper=2.0)
        a_au  = pm.Deterministic("a_au", 10.0 ** log_a)

        # inclination: uniform in cos(i) → sin prior on i
        # cos_i ~ Uniform(-1, 1)  :  i ~ arccos(Uniform)
        cos_i = pm.Uniform("cos_i", lower=-1.0, upper=1.0)
        inc   = pm.Deterministic("inc_deg", pt.arccos(cos_i) * 180.0 / np.pi)

        # eccentricity: beta(1.12, 3.09) : astrophysically motivated prior

        ecc = pm.Beta("ecc", alpha=1.12, beta=3.09)

        # angles
        Omega = pm.Uniform("Omega_deg", lower=0.0, upper=360.0)
        omega = pm.Uniform("omega_deg", lower=0.0, upper=360.0)

        # time of periastron: uniform over one full period
        tp_frac = pm.Uniform("tp_frac", lower=0.0, upper=1.0)

   
        # planet mass prior
        # log-uniform from 0.1 Mjup to 80 Mjup (below brown dwarf limit)
     
        log_Mp_jup = pm.Uniform("log_Mp_jup", lower=-1.0, upper=np.log10(80.0))
        Mp_jup     = pm.Deterministic("Mp_jup", 10.0 ** log_Mp_jup)
        Mp_msun    = pm.Deterministic("Mp_msun", Mp_jup * MJUP_MSUN)

  
        # jitter / noise term
        # accounts for unmodelled noise in RUWE 
        log_sigma0 = pm.Uniform("log_sigma0", lower=-3.0, upper=0.0)
        sigma0     = pm.Deterministic("sigma0", 10.0 ** log_sigma0)


        # derived quantities
        # orbital period from kepler's 3rd law
        period_yr = pm.Deterministic(
            "period_yr",
            pt.sqrt(a_au ** 3 / (Mp_msun + Ms))
        )

        # time of periastron in years 
        tp_yr = pm.Deterministic("tp_yr", tp_frac * period_yr)

  
        # Forward model: planet AL positions (Part 2)
    
        """planet_model() includes the star's barycentric reflex motion due
         to the companion (via the corrected w_eff = q/(1+q) - l/(1+l)
         weighting). for a dark companion (l=0), this equals q/(1+q),
         which is the physically correct star-around-barycentre amplitude.
        
         the 5-parameter single-star fit cannot absorb orbital motion
         (especially for periods comparable to or shorter than Gaia's
         34-month baseline), so this extra wobble leaks through as
         residuals : elevated RUWE.
        """

        al_pred = planet_model(
            t_tcb_days      = t_pt,
            scan_angle      = psi_pt,
            parallax_factor = f_pt,
            ra_off          = ra_off,
            dec_off         = dec_off,
            pmra            = pmra,
            pmdec           = pmdec,
            parallax_mas    = parallax,
            semimajor_au    = a_au,
            inclination_deg = inc,
            eccentricity    = ecc,
            Omega_deg       = Omega,
            omega_deg       = omega,
            t_p_yr          = tp_yr,
            Mp              = Mp_msun,
            Ms              = Ms,
            lum_ratio       = lum_ratio,
        )

   
        # RUWE calculation 

        ruwe_model = pm.Deterministic(
            "ruwe_model",
            compute_ruwe(al_pred, I_minus_H, g_mag, N)
        )


        # likelihood
        # total RUWE uncertainty = RUWE_err (fixed) + sigma0 (jitter)

        ruwe_sigma = pm.Deterministic(
            "ruwe_sigma",
            pt.sqrt(RUWE_ERR ** 2 + sigma0 ** 2)
        )

        pm.Normal(
            "ruwe_likelihood",
            mu    = ruwe_model,
            sigma = ruwe_sigma,
            observed = ruwe_obs,
        )

    return model




def run_inference(
    params: dict,
    Ms: float = 1.0,
    lum_ratio: float = 0.0,
    n_draws: int = 1000,
    n_tune: int = 1000,
    n_chains: int = 2,
    target_accept: float = 0.9,
    random_seed: int = 42,
) -> az.InferenceData:
    """
    run NUTS sampling on the RUWE companion model.

    parameters

    params       : dict from query_companion() (Part 1)
    Ms           : host star mass [solar masses]
    lum_ratio    : L_companion / L_star (0 for dark companion)
    n_draws      : posterior samples per chain
    n_tune       : tuning steps per chain
    n_chains     : number of parallel chains
    target_accept: NUTS target acceptance rate (higher = smaller steps)
    random_seed  : for reproducibility

    returns

    idata : arviz.InferenceData
        Contains posterior, sample_stats, and observed_data groups.
    """
    model = build_model(params, Ms=Ms, lum_ratio=lum_ratio)

    with model:
        print(f"\nSampling {n_chains} chains × {n_draws} draws "
              f"(+ {n_tune} tuning steps each)...")
        idata = pm.sample(
            draws          = n_draws,
            tune           = n_tune,
            chains         = n_chains,
            target_accept  = target_accept,
            random_seed    = random_seed,
            progressbar    = True,
            return_inferencedata = True,
        )

    return idata



def print_posterior_summary(idata: az.InferenceData) -> None:
    """Print key posterior statistics."""
    var_names = [
        "parallax", "pmra", "pmdec",
        "a_au", "inc_deg", "ecc", "Omega_deg", "omega_deg",
        "Mp_jup", "period_yr", "sigma0", "ruwe_model",
    ]

    present = [v for v in var_names
               if v in idata.posterior.data_vars]

    summary = az.summary(idata, var_names=present, round_to=4)
    print("\n" + "=" * 70)
    print("  Posterior summary")
    print("=" * 70)
    print(summary.to_string())
    print("=" * 70 + "\n")

    # convergence diagnostics
    rhat = az.rhat(idata, var_names=present)
    max_rhat = float(max(rhat[v].values.max() for v in present
                         if v in rhat.data_vars))
    print(f"  Max R-hat : {max_rhat:.4f}  (< 1.01 is good)")

    ess = az.ess(idata, var_names=present)
    min_ess = float(min(ess[v].values.min() for v in present
                        if v in ess.data_vars))
    print(f"  Min ESS   : {min_ess:.0f}   (> 400 is good)")
    print()



    #test

if __name__ == "__main__":
    SOURCE_ID = "31958852451968"

    params = query_companion(SOURCE_ID)

    print("\nRunning prior predictive check (10 samples)...")
    model = build_model(params, Ms=1.0)
    with model:
        prior = pm.sample_prior_predictive(samples=10, random_seed=42)
    ruwe_prior = prior.prior["ruwe_model"].values.flatten()
    print(f"  Prior RUWE range: [{ruwe_prior.min():.3f}, {ruwe_prior.max():.3f}]")
    print(f"  Observed RUWE   : {params['ruwe']:.4f}")

    idata = run_inference(
        params,
        Ms           = 1.0,
        lum_ratio    = 0.0,
        n_draws      = 500,      # increase to ≥2000 for publication
        n_tune       = 500,
        n_chains     = 4,
        target_accept= 0.9,
        random_seed  = 42,
    )

    print_posterior_summary(idata)

    out_path = f"posterior_{SOURCE_ID}.nc"
    idata.to_netcdf(out_path)
    print(f"Posterior saved to: {out_path}")
    print("Load with: az.from_netcdf('posterior_927713095040.nc')")