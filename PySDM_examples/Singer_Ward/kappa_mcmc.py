import time

import numpy as np
from joblib import Parallel, delayed
from PySDM import Formulae
from PySDM.physics import constants_defaults as const
from PySDM.physics import si
from scipy.optimize import minimize_scalar


# parameter transformation so the MCMC parameters range from [-inf, inf]
# but the compressed film parameters are bounded appropriately
# for Ovadnevaite:
# sgm_org = [0,72.8] and delta_min = [0,inf]
# for Ruehl:
# A0 = [0,inf], C0 = [0,inf], sgm_min = [0,inf], and m_sigma = [-inf,inf]
# for SzyszkowskiLangmuir
# A0 = [0,inf], C0 = [0,inf], and sgm_min = [0,inf]
def param_transform(mcmc_params, model):
    film_params = np.copy(mcmc_params)

    if model == "CompressedFilmOvadnevaite":
        film_params[0] = (
            const.sgm_w / (1 + np.exp(-1 * mcmc_params[0])) / (si.mN / si.m)
        )
        film_params[1] = np.exp(mcmc_params[1])
    elif model == "CompressedFilmRuehl":
        film_params[0] = mcmc_params[0] * 1e-20
        film_params[1] = np.exp(mcmc_params[1])
        film_params[2] = np.exp(mcmc_params[2])
        film_params[3] = mcmc_params[3] * 1e17
    elif model == "SzyszkowskiLangmuir":
        film_params[0] = mcmc_params[0] * 1e-20
        film_params[1] = np.exp(mcmc_params[1])
        film_params[2] = np.exp(mcmc_params[2])
    else:
        raise AssertionError()

    return film_params


def negSS(r_wet, SS_args):
    formulae, T, r_dry, kappa, f_org = SS_args
    v_dry = formulae.trivia.volume(r_dry)
    v_wet = formulae.trivia.volume(r_wet)
    sigma = formulae.surface_tension.sigma(T, v_wet, v_dry, f_org)
    RH_eq = formulae.hygroscopicity.RH_eq(r_wet, T, kappa, r_dry**3, sigma)
    SS = (RH_eq - 1) * 100
    return -1 * SS


# def get_rcrit(SS_args, bracket):
#     return minimize_scalar(negSS, args=SS_args, bracket=bracket).x


# evaluate the y-values of the model, given the current guess of parameter values
def get_model(params, args):
    T, r_dry, _, aerosol_list, model = args
    kappa = [ai.modes[0]["kappa"][model] for ai in aerosol_list]
    forg = [ai.modes[0]["f_org"] for ai in aerosol_list]
    nu_org = aerosol_list[0].modes[0]["nu_org"]

    if model == "CompressedFilmOvadnevaite":
        formulae = Formulae(
            surface_tension=model,
            constants={
                "sgm_org": param_transform(params, model)[0] * si.mN / si.m,
                "delta_min": param_transform(params, model)[1] * si.nm,
            },
        )
    elif model == "SzyszkowskiLangmuir":
        formulae = Formulae(
            surface_tension=model,
            constants={
                "RUEHL_nu_org": nu_org,
                "RUEHL_A0": param_transform(params, model)[0] * si.m**2,
                "RUEHL_C0": param_transform(params, model)[1],
                "RUEHL_sgm_min": param_transform(params, model)[2] * si.mN / si.m,
            },
        )
    elif model == "CompressedFilmRuehl":
        formulae = Formulae(
            surface_tension=model,
            constants={
                "RUEHL_nu_org": nu_org,
                "RUEHL_A0": param_transform(params, model)[0] * si.m**2,
                "RUEHL_C0": param_transform(params, model)[1],
                "RUEHL_sgm_min": param_transform(params, model)[2] * si.mN / si.m,
                "RUEHL_m_sigma": param_transform(params, model)[3] * si.J / si.m**2,
            },
        )
    else:
        raise AssertionError()

    rcrit = np.zeros(len(r_dry))
    for i, rd in enumerate(r_dry):
        SS_args = [formulae, T, rd, kappa[i], forg[i]]
        res = minimize_scalar(negSS, args=SS_args, bracket=[rd / 2, 100e-6])
        rcrit[i] = res.x

    # rcrit = np.array(Parallel(verbose=0, n_jobs=-1, backend="threading")(
    #     delayed(get_rcrit)(
    #         [formulae, T, rd, kappa[i], forg[i]],
    #         [rd / 2, 100e-6]
    #     )
    #     for i,rd in enumerate(r_dry)
    # ))

    kap_eff = (
        (2 * rcrit**2) / (3 * r_dry**3 * const.Rv * T * const.rho_w) * const.sgm_w
    )

    return kap_eff


# obtain the chi2 value of the model y-values given current parameters
# vs. the measured y-values
# calculate chi2 not log likelihood
def get_chi2(params, args, y, error):
    model = get_model(params, args)
    chi2 = np.sum(((y - model) / error) ** 2)
    return chi2


# propose a new parameter set
# take a step in one paramter
# of random length in random direction
# with stepsize chosen from a normal distribution with width sigma
def propose_param(current_param, stepsize):
    picker = int(np.floor(np.random.random(1) * len(current_param)))
    sigma = stepsize[picker]
    perturb_value = np.random.normal(0.0, sigma)

    try_param = np.zeros(len(current_param))
    try_param[~picker] = current_param[~picker]
    try_param[picker] = current_param[picker] + perturb_value

    try_param = np.copy(current_param)
    try_param[picker] = current_param[picker] + perturb_value

    return try_param, picker


# evaluate whether to step to the new trial value
def step_eval(params, stepsize, args, y, error):
    chi2_old = get_chi2(params, args, y, error)
    try_param, picker = propose_param(params, stepsize)
    chi2_try = get_chi2(try_param, args, y, error)

    # determine whether a step should be taken
    if chi2_try <= chi2_old:
        new_param = try_param
        accept_value = 1
    else:
        alpha = np.exp(chi2_old - chi2_try)
        r = np.random.random(1)
        if r < alpha:
            new_param = try_param
            accept_value = 1
        else:
            new_param = params
            accept_value = 0

    chi2_value = get_chi2(new_param, args, y, error)
    return new_param, picker, accept_value, chi2_value


# run the whole MCMC routine, calling the subroutines written above
def MCMC(params, stepsize, args, y, error, n_steps):
    param_chain = np.zeros((len(params), n_steps))
    accept_chain = np.empty((len(params), n_steps))
    accept_chain[:] = np.nan
    chi2_chain = np.zeros(n_steps)

    for i in np.arange(n_steps):
        t = time.time()
        param_chain[:, i], ind, accept_value, chi2_chain[i] = step_eval(
            params, stepsize, args, y, error
        )
        accept_chain[ind, i] = accept_value
        params = param_chain[:, i]
        # print("step time: ", time.time() - t)

    return param_chain, accept_chain, chi2_chain
