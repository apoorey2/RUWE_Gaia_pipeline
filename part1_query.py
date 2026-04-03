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


#test

if __name__ == "__main__":
    SOURCE_ID = "927713095040"

    params = query_companion(SOURCE_ID)
    print_summary(params)