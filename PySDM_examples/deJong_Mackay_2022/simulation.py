from PySDM.backends import CPU
from PySDM.builder import Builder
from PySDM.dynamics import Coalescence, Collision
from PySDM.environments import Box
from PySDM.initialisation.sampling.spectral_sampling import ConstantMultiplicity
from PySDM.physics import si
from PySDM.products.collision.collision_rates import (
    CollisionRateDeficitPerGridbox,
    CollisionRatePerGridbox,
)
from PySDM.products.size_spectral import (
    ParticleSizeSpectrumPerVolume,
    ParticleVolumeVersusRadiusLogarithmSpectrum,
)


def run_box_breakup(settings, step):
    backend = CPU

    builder = Builder(n_sd=settings.n_sd, backend=backend(settings.formulae))
    env = Box(dv=settings.dv, dt=settings.dt)
    builder.set_environment(env)
    env["rhod"] = 1.0
    attributes = {}
    attributes["volume"], attributes["n"] = ConstantMultiplicity(
        settings.spectrum
    ).sample(settings.n_sd)
    breakup = Collision(
        collision_kernel=settings.kernel,
        coalescence_efficiency=settings.coal_eff,
        breakup_efficiency=settings.break_eff,
        fragmentation_function=settings.fragmentation,
        adaptive=settings.adaptive,
    )
    builder.add_dynamic(breakup)
    products = (
        ParticleVolumeVersusRadiusLogarithmSpectrum(
            radius_bins_edges=settings.radius_bins_edges, name="dv/dlnr"
        ),
        CollisionRatePerGridbox(name="cr"),
        CollisionRateDeficitPerGridbox(name="crd"),
    )
    core = builder.build(attributes, products)

    # run
    core.run(step - core.n_steps)

    x = (settings.radius_bins_edges[:-1] / si.micrometres,)
    y = core.products["dv/dlnr"].get()

    return (x, y)


def run_box_NObreakup(settings, step):
    backend = CPU

    builder = Builder(n_sd=settings.n_sd, backend=backend(settings.formulae))
    env = Box(dv=settings.dv, dt=settings.dt)
    builder.set_environment(env)
    env["rhod"] = 1.0
    attributes = {}
    attributes["volume"], attributes["n"] = ConstantMultiplicity(
        settings.spectrum
    ).sample(settings.n_sd)
    coal = Coalescence(
        collision_kernel=settings.kernel,
        coalescence_efficiency=settings.coal_eff,
        adaptive=settings.adaptive,
    )
    builder.add_dynamic(coal)
    products = (
        ParticleVolumeVersusRadiusLogarithmSpectrum(
            radius_bins_edges=settings.radius_bins_edges, name="dv/dlnr"
        ),
        CollisionRatePerGridbox(name="cr"),
        CollisionRateDeficitPerGridbox(name="crd"),
    )
    core = builder.build(attributes, products)

    # run
    core.run(step - core.n_steps)

    x = (settings.radius_bins_edges[:-1] / si.micrometres,)
    y = core.products["dv/dlnr"].get()

    return (x, y)


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
        collision_kernel=settings.kernel,
        coalescence_efficiency=settings.coal_eff,
        breakup_efficiency=settings.break_eff,
        fragmentation_function=settings.fragmentation,
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
