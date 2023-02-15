import numpy as np
from PySDM.backends import CPU
from PySDM.builder import Builder
from PySDM.dynamics import Coalescence, Collision
from PySDM.environments import Box
from PySDM.initialisation.sampling.spectral_sampling import ConstantMultiplicity
from PySDM.physics import si
from PySDM.products.collision.collision_rates import (
    BreakupRatePerGridbox,
    CoalescenceRatePerGridbox,
    CollisionRateDeficitPerGridbox,
    CollisionRatePerGridbox,
)
from PySDM.products.size_spectral import ParticleVolumeVersusRadiusLogarithmSpectrum


def run_box_breakup(settings, steps=None, backend_class=CPU):
    builder = Builder(n_sd=settings.n_sd, backend=backend_class(settings.formulae))
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
        warn_overflows=settings.warn_overflows,
    )
    builder.add_dynamic(breakup)
    products = (
        ParticleVolumeVersusRadiusLogarithmSpectrum(
            radius_bins_edges=settings.radius_bins_edges, name="dv/dlnr"
        ),
        CollisionRatePerGridbox(name="cr"),
        CollisionRateDeficitPerGridbox(name="crd"),
        CoalescenceRatePerGridbox(name="cor"),
        BreakupRatePerGridbox(name="br"),
    )
    core = builder.build(attributes, products)

    if steps is None:
        steps = settings.output_steps
    y = np.ndarray((len(steps), len(settings.radius_bins_edges) - 1))
    rates = np.zeros((len(steps), 4))
    # run
    for i, step in enumerate(steps):
        core.run(step - core.n_steps)
        y[i] = core.products["dv/dlnr"].get()[0]
        rates[i, 0] = core.products["cr"].get()
        rates[i, 1] = core.products["crd"].get()
        rates[i, 2] = core.products["cor"].get()
        rates[i, 3] = core.products["br"].get()

    x = (settings.radius_bins_edges[:-1] / si.micrometres,)[0]

    return (x, y, rates)


def run_box_NObreakup(settings, steps=None, backend_class=CPU):
    builder = Builder(n_sd=settings.n_sd, backend=backend_class(settings.formulae))
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
        CoalescenceRatePerGridbox(name="cor"),
    )
    core = builder.build(attributes, products)

    if steps is None:
        steps = settings.output_steps
    y = np.ndarray((len(steps), len(settings.radius_bins_edges) - 1))
    rates = np.zeros((len(steps), 4))
    # run
    for i, step in enumerate(steps):
        core.run(step - core.n_steps)
        y[i] = core.products["dv/dlnr"].get()[0]
        rates[i, 0] = core.products["cr"].get()
        rates[i, 1] = core.products["crd"].get()
        rates[i, 2] = core.products["cor"].get()

    x = (settings.radius_bins_edges[:-1] / si.micrometres,)[0]

    return (x, y, rates)
