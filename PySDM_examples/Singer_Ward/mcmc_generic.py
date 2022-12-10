import os
import warnings

import numpy as np
from numba.core.errors import NumbaExperimentalFeatureWarning
from PySDM.physics import si

from PySDM_examples.Singer_Ward.aerosol import (
    AerosolAlphaPineneDark,
    AerosolAlphaPineneLight,
    AerosolBetaCaryophylleneDark,
    AerosolBetaCaryophylleneLight,
)
from PySDM_examples.Singer_Ward.kappa_mcmc import MCMC, param_transform
from PySDM_examples.Singer_Ward.mcmc_plots import (
    plot_corner,
    plot_keff,
    plot_ovf_kappa_fit,
    plot_param_chain,
)


def mcmc_generic(
    filename="bcary_dark.csv", model="CompressedFilmOvadnevaite", n_steps=200, plot=True
):

    ######
    # open data file
    ######
    ds = np.loadtxt("data/" + filename, skiprows=1, delimiter=",")
    if filename == "bcary_dark.csv":
        ds = np.delete(ds, [26, 65], axis=0)  # remove outliers
    r_dry = ds[:, 0] / 2 * 1e-9
    ovf = np.minimum(ds[:, 1], 0.99)
    d_ovf = ds[:, 2]
    kappa_eff = ds[:, 3]
    d_kappa_eff = ds[:, 4]
    T = 300 * si.K

    datay = kappa_eff
    errory = d_kappa_eff

    ######
    # set up MCMC
    ######
    if model == "CompressedFilmOvadnevaite":
        params = [0.5, 0.2]
        stepsize = [0.1, 0.1]
    elif model == "SzyszkowskiLangmuir":
        params = [20, -12.0, 3.9]
        stepsize = [0.5, 0.1, 0.05]
    elif model == "CompressedFilmRuehl":
        params = [15.1, -12.0, 3.3, 0.8]
        stepsize = [0.1, 0.05, 0.01, 0.05]
    else:
        print("error model name not recognized")

    if filename == "bcary_dark.csv":
        aerosol_list = [AerosolBetaCaryophylleneDark(ovfi) for ovfi in ovf]
    elif filename == "bcary_light.csv":
        aerosol_list = [AerosolBetaCaryophylleneLight(ovfi) for ovfi in ovf]
    elif filename == "apinene_dark.csv":
        aerosol_list = [AerosolAlphaPineneDark(ovfi) for ovfi in ovf]
    elif filename == "apinene_light.csv":
        aerosol_list = [AerosolAlphaPineneLight(ovfi) for ovfi in ovf]
    else:
        print("error aerosol type doesn't exist")
    args = [T, r_dry, ovf, aerosol_list, model]

    ######
    # run MCMC
    ######
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=NumbaExperimentalFeatureWarning)
        param_chain, accept_chain, chi2_chain = MCMC(
            params, stepsize, args, datay, errory, n_steps
        )
    p = param_transform(param_chain, model)

    print(p[:, -1])
    print(param_chain[:, -1])

    ######
    # plot and save results
    ######
    if plot:
        if not os.path.isdir("mcmc_output/"):
            os.mkdir("mcmc_output/")
        plot_param_chain(param_chain, args)
        plot_corner(param_chain, args)
        plot_ovf_kappa_fit(param_chain, args, d_ovf, datay, errory)
        plot_keff(param_chain, args, datay, errory)
