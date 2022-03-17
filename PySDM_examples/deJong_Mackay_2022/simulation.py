from PySDM.backends import CPU
from PySDM.builder import Builder
from PySDM.dynamics import Collision
from PySDM.environments import Box
from PySDM.initialisation.sampling.spectral_sampling import ConstantMultiplicity
from PySDM.products.collision.collision_rates import (
    CollisionRateDeficitPerGridbox,
    CollisionRatePerGridbox,
)
from PySDM.products.size_spectral import ParticleSizeSpectrumPerVolume


def make_core(settings):
    backend = CPU

    builder = Builder(n_sd=settings.n_sd, backend=backend(settings.formulae))
    env = Box(dv=settings.dv, dt=settings.dt)
    builder.set_environment(env)
    env["rhod"] = 1.0
    attributes = {}
    attributes["volume"], attributes["n"] = ConstantMultiplicity(
        settings.spectrum
    ).sample(settings.n_sd)
    collision = Collision(
        settings.kernel,
        settings.coal_eff,
        settings.break_eff,
        settings.fragmentation,
        adaptive=settings.adaptive,
    )
    builder.add_dynamic(collision)
    products = (
        ParticleSizeSpectrumPerVolume(
            radius_bins_edges=settings.radius_bins_edges, name="dv/dlnr"
        ),
        CollisionRatePerGridbox(name="cr"),
        CollisionRateDeficitPerGridbox(name="crd"),
    )
    return builder.build(attributes, products)
