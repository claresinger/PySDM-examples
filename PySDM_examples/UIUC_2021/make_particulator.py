import numpy as np
from PySDM import Builder, Formulae
from PySDM import products as PySDM_products
from PySDM.backends import CPU
from PySDM.dynamics import Freezing
from PySDM.environments import Box
from PySDM.initialisation.sampling.spectral_sampling import ConstantMultiplicity
from PySDM.initialisation.sampling.spectro_glacial_sampling import (
    SpectroGlacialSampling,
)

A_VALUE_LARGER_THAN_ONE = 44


def make_particulator(
    *,
    constants,
    n_sd,
    dt,
    initial_temperature,
    singular,
    seed,
    shima_T_fz,
    ABIFM_spec,
    droplet_volume,
    total_particle_number,
    volume
):
    attributes = {"volume": np.ones(n_sd) * droplet_volume}

    formulae_ctor_args = {"seed": seed, "constants": constants}
    if singular:
        formulae_ctor_args["freezing_temperature_spectrum"] = shima_T_fz
    else:
        formulae_ctor_args["heterogeneous_ice_nucleation_rate"] = "ABIFM"
    formulae = Formulae(**formulae_ctor_args)

    if singular:
        sampling = SpectroGlacialSampling(
            freezing_temperature_spectrum=formulae.freezing_temperature_spectrum,
            insoluble_surface_spectrum=ABIFM_spec,
            seed=formulae.seed,
        )
        attributes["freezing temperature"], _, attributes["n"] = sampling.sample(n_sd)
    else:
        sampling = ConstantMultiplicity(
            spectrum=ABIFM_spec,
            # seed=formulae.seed
        )
        attributes["immersed surface area"], attributes["n"] = sampling.sample(n_sd)
    attributes["n"] *= total_particle_number

    builder = Builder(n_sd, CPU(formulae))

    env = Box(dt, volume)
    builder.set_environment(env)
    env["T"] = initial_temperature
    env["RH"] = A_VALUE_LARGER_THAN_ONE

    builder.add_dynamic(Freezing(singular=singular))

    return builder.build(
        attributes=attributes,
        products=(
            PySDM_products.Time(name="t"),
            PySDM_products.AmbientTemperature(name="T"),
            PySDM_products.SpecificIceWaterContent(name="qi"),
        ),
    )
