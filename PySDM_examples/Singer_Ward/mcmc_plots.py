import numpy as np
from corner import corner
from matplotlib import pylab

from PySDM_examples.Singer_Ward.kappa_mcmc import get_model, param_transform


def plot_param_chain(param_chain, args):
    _, _, _, c, model = args
    p = param_transform(param_chain, model)

    if model == "CompressedFilmOvadnevaite":
        labels = ["sgm_org", "delta_min"]
        _, axes = pylab.subplots(2, 1, figsize=(6, 8))
    elif model == "CompressedFilmRuehl":
        labels = ["A0", "C0", "sgm_min", "m_sigma"]
        _, axes = pylab.subplots(2, 2, figsize=(12, 8))
    elif model == "SzyszkowskiLangmuir":
        labels = ["A0", "C0", "sgm_min"]
        _, axes = pylab.subplots(3, 1, figsize=(6, 12))
    else:
        raise AssertionError()

    for i, ax in enumerate(axes.flatten()):
        p[i, 0:100] = np.nan
        ax.plot(p[i, :])
        ax.set_ylabel(labels[i])
        ax.grid()
    pylab.tight_layout()

    modelname = model.split("CompressedFilm")[-1]
    aerosolname = c[0].__class__.__name__.split("Aerosol")[-1]
    pylab.savefig(
        "mcmc_output/" + aerosolname + "_" + modelname + "_chain.png",
        dpi=200,
        bbox_inches="tight",
        facecolor="w",
    )
    pylab.show()

    filename = (
        "mcmc_output/"
        + aerosolname
        + "_"
        + modelname
        + "_chain"
        + str(np.max(np.shape(param_chain)))
        + ".csv"
    )
    np.savetxt(filename, param_chain.T, fmt="%.6e", delimiter=",")


def plot_corner(param_chain, args):
    _, _, _, c, model = args
    data = param_transform(param_chain, model).T
    data = data[100:, :]

    if model == "CompressedFilmOvadnevaite":
        labels = ["sgm_org", "delta_min"]
    elif model == "CompressedFilmRuehl":
        labels = ["A0", "C0", "sgm_min", "m_sigma"]
    elif model == "SzyszkowskiLangmuir":
        labels = ["A0", "C0", "sgm_min"]
    else:
        raise AssertionError()

    pylab.rcParams.update({"font.size": 12})
    _ = corner(
        data,
        labels=labels,
        label_kwargs={"fontsize": 12},
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt=".1e",
        title_kwargs={"fontsize": 12},
    )

    modelname = model.split("CompressedFilm")[-1]
    aerosolname = c[0].__class__.__name__.split("Aerosol")[-1]
    pylab.savefig(
        "mcmc_output/" + aerosolname + "_" + modelname + "_corner.png",
        dpi=200,
        bbox_inches="tight",
        facecolor="w",
    )
    pylab.show()


def plot_ovf_kappa_fit(param_chain, args, errorx, datay, errory):
    _, _, ovf, c, model = args

    # create aerosol
    dat = np.zeros((len(ovf), 4))

    pylab.figure(figsize=(10, 6))

    # before
    kap_eff = get_model(param_chain[:, 0], args)
    s = np.argsort(ovf)
    dat[:, 0] = ovf[s]
    dat[:, 2] = kap_eff[s]
    pylab.plot(ovf[s], kap_eff[s], "b:", label="before")

    # after
    kap_eff2 = get_model(param_chain[:, -1], args)
    dat[:, 3] = kap_eff2[s]
    pylab.plot(ovf[s], kap_eff2[s], "r-", label="after")

    # data
    s = np.argsort(ovf)
    dat[:, 1] = datay[s]
    pylab.errorbar(ovf, datay, yerr=errory, xerr=errorx, fmt="ko")

    pylab.legend()
    pylab.xlabel("Organic Volume Fraction")
    pylab.ylabel(r"$\kappa_{eff}$", fontsize=20)
    pylab.rcParams.update({"font.size": 15})
    pylab.grid()
    pylab.tight_layout()

    modelname = model.split("CompressedFilm")[-1]
    aerosolname = c[0].__class__.__name__.split("Aerosol")[-1]
    pylab.savefig(
        "mcmc_output/" + aerosolname + "_" + modelname + "_fit.png",
        dpi=200,
        bbox_inches="tight",
        facecolor="w",
    )
    pylab.show()


def plot_keff(param_chain, args, datay, errory):
    pylab.figure(figsize=(10, 6))

    # before
    kap_eff = get_model(param_chain[:, 0], args)

    # after
    kap_eff2 = get_model(param_chain[:, -1], args)

    pylab.figure(figsize=(7, 6))
    pylab.errorbar(kap_eff, datay, yerr=errory, fmt="bo", label="before")
    pylab.errorbar(kap_eff2, datay, yerr=errory, fmt="ro", label="after")
    pylab.legend()
    pylab.xlabel(r"$\kappa_{eff}$ modeled", fontsize=20)
    pylab.ylabel(r"$\kappa_{eff}$ measured", fontsize=20)
    pylab.plot([-0.05, 0.55], [-0.05, 0.55], "k-")
    pylab.xlim([-0.05, 0.55])
    pylab.ylim([-0.05, 0.55])
    pylab.rcParams.update({"font.size": 15})
    pylab.grid()

    _, _, _, c, model = args
    modelname = model.split("CompressedFilm")[-1]
    aerosolname = c[0].__class__.__name__.split("Aerosol")[-1]
    pylab.savefig(
        "mcmc_output/" + aerosolname + "_" + modelname + "_keff.png",
        dpi=200,
        bbox_inches="tight",
        facecolor="w",
    )
    pylab.show()
