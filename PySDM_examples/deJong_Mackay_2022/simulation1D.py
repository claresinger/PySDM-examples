import PySDM.products as PySDM_products
from PySDM.dynamics import Collision
from PySDM.dynamics.collisions.collision_kernels import Geometric

from PySDM_examples.Shipway_and_Hill_2012.simulation import Simulation as SimulationSH


class Simulation1D(SimulationSH):
    def __init__(self, settings):
        super().__init__(settings)
        self.output_steps = settings.output_steps

    @staticmethod
    def add_collision_dynamic(builder, settings, products):
        if settings.breakup:
            builder.add_dynamic(
                Collision(
                    collision_kernel=Geometric(collection_efficiency=1),
                    coalescence_efficiency=settings.coalescence_efficiency,
                    breakup_efficiency=settings.breakup_efficiency,
                    fragmentation_function=settings.fragmentation_function,
                    adaptive=settings.coalescence_adaptive,
                    warn_overflows=settings.warn_breakup_overflow,
                )
            )
            products.append(
                PySDM_products.BreakupRateDeficitPerGridbox(
                    name="breakup_deficit",
                )
            )
            products.append(
                PySDM_products.BreakupRatePerGridbox(
                    name="breakup_rate",
                )
            )
        else:
            SimulationSH.add_collision_dynamic(builder, settings, products)

    def save(self, step):
        if step in self.output_steps:
            super().save(step)

    def run(self):
        result = super().run()
        for key, val in result.products.items():
            if len(val.shape) == 2:
                result.products[key] = val[:, self.output_steps]
        result.products["t"] = result.products["t"][self.output_steps]
        return result
