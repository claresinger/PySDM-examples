import numpy as np
from matplotlib import pyplot
from PySDM.physics.constants import convert_to, si


def plot_ax(
    ax,
    var,
    qlabel,
    output,
    contour_var1=None,
    contour_lvl1=None,
    contour_var2=None,
    contour_lvl2=None,
    cmin=None,
    cmax=None,
):
    tgrid = output["t"].copy()
    zgrid = output["z"].copy()
    convert_to(zgrid, si.km)

    if cmin is not None and cmax is not None:
        levels = np.linspace(cmin, cmax, 20)
        mesh = ax.contourf(
            tgrid,
            zgrid,
            output[var],
            levels=levels,
            cmap="BuPu",
            vmin=cmin,
            vmax=cmax,
            extend="max",
        )
    else:
        mesh = ax.contourf(
            tgrid, zgrid, output[var], levels=20, cmap="BuPu", extend="max"
        )

    if contour_var1 is not None and contour_lvl1 is not None:
        ax.contour(
            tgrid,
            zgrid,
            output[contour_var1],
            contour_lvl1,
            colors="k",
            linestyles="--",
        )
    if contour_var2 is not None and contour_lvl2 is not None:
        ax.contour(
            tgrid,
            zgrid,
            output[contour_var2],
            contour_lvl2,
            colors="r",
            linestyles="--",
        )

    ax.set_xlabel("time [s]")
    ax.set_ylabel("z [km]")
    ax.set_ylim(0, None)

    cbar_levels = np.linspace(cmin, cmax, 5, endpoint="True")
    cbar = pyplot.colorbar(mesh, fraction=0.05, location="top", ax=ax)
    cbar.set_ticks(cbar_levels)
    cbar.set_label(qlabel)


def plot_zeros_ax(ax, var, qlabel, output, cmin=None, cmax=None):
    dt = output["t"][1] - output["t"][0]
    dz = output["z"][1] - output["z"][0]
    tgrid = np.concatenate(((output["t"][0] - dt / 2,), output["t"] + dt / 2))
    zgrid = np.concatenate(((output["z"][0] - dz / 2,), output["z"] + dz / 2))
    convert_to(zgrid, si.km)

    # fig = pyplot.figure(constrained_layout=True)
    output[var + "zero"] = np.zeros_like(output[var])
    if cmin is not None and cmax is not None:
        mesh = ax.pcolormesh(
            tgrid, zgrid, output[var + "zero"], cmap="BuPu", vmin=cmin, vmax=cmax
        )
    else:
        mesh = ax.pcolormesh(tgrid, zgrid, output[var], cmap="BuPu")

    ax.set_xlabel("time [s]")
    ax.set_ylabel("z [km]")
    ax.set_ylim(0, None)

    cbar = pyplot.colorbar(mesh, fraction=0.05, location="top", ax=ax)
    cbar.set_label(qlabel)
