import matplotlib
import numpy as np
from matplotlib import pyplot
from PySDM.physics import si

from PySDM_examples.UIUC_2021.frozen_fraction import FrozenFraction

labels = {
    True: "singular (Niemand et al. 2012)",
    False: "time-dependent (ABIFM, illite)",
}
colors = {True: "black", False: "teal"}
qi_unit = si.g / si.m**3


def make_sampling_plot(data):
    ensemble_n = len(data) // 2
    _, axs = pyplot.subplots(nrows=ensemble_n, ncols=2)  # , constrained_layout=True)
    for i, v in enumerate(data):
        row = i % ensemble_n
        col = i // ensemble_n
        if "freezing temperature" in v["spectrum"]:
            x = v["spectrum"]["freezing temperature"]
            axs[row, col].set_xlabel("$T_{fz}$")
            axs[row, col].invert_xaxis()
        else:
            x = v["spectrum"]["immersed surface area"]
            axs[row, col].set_xlabel("$A$")
        axs[row, col].stem(x, v["spectrum"]["n"])
        axs[row, col].set_ylabel("n")


def make_temperature_plot(data):
    pyplot.xlabel("time [s]")

    xy1 = pyplot.gca()

    xy1.set_ylabel("temperature [K]")
    xy1.set_ylim(200, 300)
    datum = data[0]["products"]
    xy1.plot(datum["t"], datum["T"], marker=".", label="T", color="black")

    xy2 = xy1.twinx()
    plotted = {singular: False for singular in (True, False)}
    for v in data:
        datum = v["products"]
        xy2.plot(
            datum["t"],
            np.asarray(datum["qi"]) / qi_unit,  # marker='.',
            label=f"Monte-Carlo ({labels[v['singular']]})"
            if not plotted[v["singular"]]
            else "",
            color=colors[v["singular"]],
        )
        plotted[v["singular"]] = True
    xy2.set_ylabel("ice water content [g/m3]")

    xy1.grid()
    xy1.legend()  # bbox_to_anchor=(.2, 1.15))
    xy2.legend()  # bbox_to_anchor=(.7, 1.15))


def make_freezing_spec_plot(
    data, formulae, volume, droplet_volume, total_particle_number, surf_spec
):
    pyplot.xlabel("temperature [K]")
    plotted = {singular: False for singular in (True, False)}

    prim = pyplot.gca()
    for v in data:
        datum = v["products"]
        color = colors[v["singular"]]
        prim.plot(
            datum["T"],
            np.asarray(datum["qi"]) / qi_unit,
            marker=".",
            linewidth=0.333,
            label=f"Monte-Carlo: {labels[v['singular']]}"
            if not plotted[v["singular"]]
            else "",
            color=color,
        )
        plotted[v["singular"]] = True

    ff = FrozenFraction(
        volume=volume,
        droplet_volume=droplet_volume,
        total_particle_number=total_particle_number,
        rho_w=formulae.constants.rho_w,
    )
    twin = prim.secondary_yaxis(
        "right",
        functions=(lambda x: ff.qi2ff(x * qi_unit), lambda x: ff.ff2qi(x) / qi_unit),
    )
    twin.set_ylabel("frozen fraction")

    T = np.linspace(max(datum["T"]), min(datum["T"]))
    for multiplier, color in {0.1: "orange", 1: "red", 10: "brown"}.items():
        prim.plot(
            T,
            ff.ff2qi(
                formulae.freezing_temperature_spectrum.cdf(
                    T, multiplier * surf_spec.median
                )
            )
            / qi_unit,
            label=f"singular CDF for {multiplier}x median surface",
            linewidth=2.5,
            color=color,
            linestyle="--",
        )
    prim.set_title(f"$σ_g$=exp({np.log(surf_spec.s_geom):.3g})")
    prim.set_ylabel("ice water content [$g/m^3$]")
    prim.set_xlim(T[0], T[-1])
    prim.legend(bbox_to_anchor=(1.2, -0.2))
    prim.grid()


def make_pdf_plot(A_spec, Shima_T_fz, A_range, T_range):
    N = 256
    T_space = np.linspace(*T_range, N)
    A_space = np.linspace(*A_range, N)
    grid = np.meshgrid(T_space, A_space)
    sampled_pdf = Shima_T_fz(*grid) * A_spec.pdf(grid[1])

    fig = pyplot.figure(
        figsize=(7, 6),
    )
    ax = fig.add_subplot(111)
    ax.set_xlabel("freezing temperature [K]")
    ax.set_ylabel("insoluble surface [$μm^2$]")
    cnt = ax.contourf(
        grid[0],
        grid[1] / si.um**2,
        sampled_pdf * si.um**2,
        norm=matplotlib.colors.LogNorm(),
        cmap="Blues",
        levels=np.logspace(-3, 0, 7),
    )
    cbar = pyplot.colorbar(cnt)
    cbar.set_label("pdf [$K^{-1} μm^{-2}$]")
    ax.set_title(f"$σ_g$=exp({np.log(A_spec.s_geom):.3g})")
    pyplot.grid()
