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


# test

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    SOURCE_ID = "927713095040"
    params = query_companion(SOURCE_ID)

    t_pt   = pt.as_tensor_variable(params["t_obs"].astype("float64"))
    psi_pt = pt.as_tensor_variable(params["scan_angle"].astype("float64"))
    f_pt   = pt.as_tensor_variable(params["parallax_factor"].astype("float64"))

    AL_star = single_star(
        t_tcb_days      = t_pt,
        scan_angle      = psi_pt,
        parallax_factor = f_pt,
        ra_off          = 0.0,
        dec_off         = 0.0,
        pmra            = params["pmra"],
        pmdec           = params["pmdec"],
        parallax_mas    = params["parallax"],
    )

    print("\nSingle-star AL positions (mas):")
    AL_star_val = AL_star.eval()
    for i, (ti, ali) in enumerate(zip(params["t_obs"], AL_star_val)):
        print(f"  t={ti:.2f} d  ψ={np.rad2deg(params['scan_angle'][i]):.1f}°  AL={ali:.4f} mas")

    AL_planet = planet_model(
        t_tcb_days      = t_pt,
        scan_angle      = psi_pt,
        parallax_factor = f_pt,
        ra_off          = 0.0,
        dec_off         = 0.0,
        pmra            = params["pmra"],
        pmdec           = params["pmdec"],
        parallax_mas    = params["parallax"],
        semimajor_au    = 5.2,
        inclination_deg = 45.0,
        eccentricity    = 0.05,
        Omega_deg       = 100.0,
        omega_deg       = 30.0,
        t_p_yr          = 0.0,
        Mp              = 1e-3,
        Ms              = 1.0,
        lum_ratio       = 0.0,
    )

    print("\nPlanet-model AL positions (mas):")
    AL_planet_val = AL_planet.eval()
    for i, (ti, ali) in enumerate(zip(params["t_obs"], AL_planet_val)):
        print(f"  t={ti:.2f} d  ψ={np.rad2deg(params['scan_angle'][i]):.1f}°  AL={ali:.4f} mas")