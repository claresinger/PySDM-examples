import numpy as np
from PySDM.backends import CPU
from PySDM.builder import Builder
from PySDM.dynamics import (
    AmbientThermodynamics,
    Coalescence,
    Condensation,
    Displacement,
    EulerianAdvection,
    Freezing,
)
from PySDM.environments import Kinematic2D
from PySDM.initialisation.sampling import spatial_sampling

from PySDM_examples.Szumowski_et_al_1998.make_default_product_collection import (
    make_default_product_collection,
)
from PySDM_examples.Szumowski_et_al_1998.mpdata_2d import MPDATA_2D
from PySDM_examples.utils import DummyController


class Simulation:
    def __init__(self, settings, storage, SpinUp, backend_class=CPU):
        self.settings = settings
        self.storage = storage
        self.particulator = None
        self.backend_class = backend_class
        self.SpinUp = SpinUp

    @property
    def products(self):
        return self.particulator.products

    def reinit(self, products=None):
        formulae = self.settings.formulae
        backend = self.backend_class(formulae=formulae)
        builder = Builder(n_sd=self.settings.n_sd, backend=backend)
        environment = Kinematic2D(
            dt=self.settings.dt,
            grid=self.settings.grid,
            size=self.settings.size,
            rhod_of=self.settings.rhod_of_zZ,
            mixed_phase=self.settings.processes["freezing"],
        )
        builder.set_environment(environment)

        if products is not None:
            products = list(products)
        else:
            products = make_default_product_collection(self.settings)

        if self.settings.processes["fluid advection"]:
            builder.add_dynamic(AmbientThermodynamics())
        if self.settings.processes["condensation"]:
            condensation = Condensation(
                rtol_x=self.settings.condensation_rtol_x,
                rtol_thd=self.settings.condensation_rtol_thd,
                adaptive=self.settings.condensation_adaptive,
                substeps=self.settings.condensation_substeps,
                dt_cond_range=self.settings.condensation_dt_cond_range,
                schedule=self.settings.condensation_schedule,
            )
            builder.add_dynamic(condensation)
        displacement = None
        if self.settings.processes["particle advection"]:
            displacement = Displacement(
                enable_sedimentation=self.settings.processes["sedimentation"],
                adaptive=self.settings.displacement_adaptive,
                rtol=self.settings.displacement_rtol,
            )
        if self.settings.processes["fluid advection"]:
            initial_profiles = {
                "th": self.settings.initial_dry_potential_temperature_profile,
                "qv": self.settings.initial_vapour_mixing_ratio_profile,
            }
            advectees = dict(
                (
                    key,
                    np.repeat(profile.reshape(1, -1), environment.mesh.grid[0], axis=0),
                )
                for key, profile in initial_profiles.items()
            )
            solver = MPDATA_2D(
                advectees=advectees,
                stream_function=self.settings.stream_function,
                rhod_of_zZ=self.settings.rhod_of_zZ,
                dt=self.settings.dt,
                grid=self.settings.grid,
                size=self.settings.size,
                displacement=displacement,
                n_iters=self.settings.mpdata_iters,
                infinite_gauge=self.settings.mpdata_iga,
                nonoscillatory=self.settings.mpdata_fct,
                third_order_terms=self.settings.mpdata_tot,
            )
            builder.add_dynamic(EulerianAdvection(solver))
        if self.settings.processes["particle advection"]:
            builder.add_dynamic(displacement)
        if self.settings.processes["coalescence"]:
            builder.add_dynamic(
                Coalescence(
                    collision_kernel=self.settings.kernel,
                    adaptive=self.settings.coalescence_adaptive,
                    dt_coal_range=self.settings.coalescence_dt_coal_range,
                    substeps=self.settings.coalescence_substeps,
                    optimized_random=self.settings.coalescence_optimized_random,
                )
            )
        if self.settings.processes["freezing"]:
            builder.add_dynamic(Freezing(singular=self.settings.freezing_singular))

        attributes = environment.init_attributes(
            spatial_discretisation=spatial_sampling.Pseudorandom(),
            dry_radius_spectrum=self.settings.spectrum_per_mass_of_dry_air,
            kappa=self.settings.kappa,
        )

        if self.settings.processes["freezing"]:
            if self.settings.freezing_inp_spec is None:
                immersed_surface_area = formulae.trivia.sphere_surface(
                    diameter=2 * formulae.trivia.radius(volume=attributes["dry volume"])
                )
            else:
                immersed_surface_area = self.settings.freezing_inp_spec.percentiles(
                    np.random.random(self.settings.n_sd),  # TODO #599: seed
                )

            if self.settings.freezing_singular:
                attributes[
                    "freezing temperature"
                ] = formulae.freezing_temperature_spectrum.invcdf(
                    np.random.random(self.settings.n_sd),  # TODO #599: seed
                    immersed_surface_area,
                )
            else:
                attributes["immersed surface area"] = immersed_surface_area

        self.particulator = builder.build(attributes, tuple(products))

        if self.SpinUp is not None:
            self.SpinUp(self.particulator, self.settings.n_spin_up)
        if self.storage is not None:
            self.storage.init(self.settings)

    def run(self, controller=DummyController(), vtk_exporter=None):
        with controller:
            for step in self.settings.output_steps:
                if controller.panic:
                    break

                self.particulator.run(step - self.particulator.n_steps)

                self.store(step)

                if vtk_exporter is not None:
                    vtk_exporter.export_attributes(self.particulator)
                    vtk_exporter.export_products(self.particulator)

                controller.set_percent(step / self.settings.output_steps[-1])

    def store(self, step):
        for name, product in self.particulator.products.items():
            self.storage.save(product.get(), step, name)
