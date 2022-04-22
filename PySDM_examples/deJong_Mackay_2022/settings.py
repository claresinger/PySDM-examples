import numpy as np
from PySDM.dynamics.collisions.breakup_efficiencies import ConstEb
from PySDM.dynamics.collisions.breakup_fragmentations import ExponFrag
from PySDM.dynamics.collisions.coalescence_efficiencies import Berry1967
from PySDM.dynamics.collisions.collision_kernels import Geometric
from PySDM.formulae import Formulae
from PySDM.initialisation.spectra import Exponential
from PySDM.physics.constants import si
from pystrict import strict


@strict
class Settings:
    def __init__(self):
        self.formulae = Formulae()
        self.n_sd = 2**20
        self.n_part = 100 / si.cm**3
        self.X0 = self.formulae.trivia.volume(radius=30.531 * si.micrometres)
        self.dv = 1 * si.m**3
        self.norm_factor = self.n_part * self.dv
        self.rho = 1000 * si.kilogram / si.metre**3
        self.dt = 1 * si.seconds
        self.adaptive = False
        self.seed = 44
        self._steps = [0]
        self.kernel = Geometric()
        self.coal_eff = Berry1967()
        self.fragmentation = ExponFrag(scale=100.0 * si.micrometres)
        self.break_eff = ConstEb(1.0)  # no "bouncing"
        self.spectrum = Exponential(norm_factor=self.norm_factor, scale=self.X0)
        self.radius_bins_edges = np.logspace(
            np.log10(10 * si.um), np.log10(5000 * si.um), num=128, endpoint=True
        )
        self.radius_range = [0 * si.um, 1e6 * si.um]

    @property
    def output_steps(self):
        return [int(step / self.dt) for step in self._steps]
