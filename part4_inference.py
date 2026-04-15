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
    SOURCE_ID = "927713095040"

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