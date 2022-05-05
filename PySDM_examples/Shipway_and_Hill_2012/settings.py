import numpy as np
from PySDM import Formulae
from PySDM.dynamics import condensation
from PySDM.initialisation import spectra
from PySDM.physics import si
from pystrict import strict
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


@strict
class Settings:
    def __init__(
        self,
        *,
        n_sd_per_gridbox: int,
        p0: float = 1007 * si.hPa,  # as used in Olesik et al. 2022 (GMD)
        particle_reservoir_depth: float = 0 * si.m,
        kappa: float = 1,
        rho_times_w_1: float = 2 * si.m / si.s * si.kg / si.m**3,
        dt: float = 1 * si.s,
        dz: float = 25 * si.m,
        precip: bool = True
    ):
        self.formulae = Formulae()
        self.n_sd_per_gridbox = n_sd_per_gridbox
        self.kappa = kappa
        self.wet_radius_spectrum_per_mass_of_dry_air = spectra.Lognormal(
            norm_factor=50 / si.cm**3 / self.formulae.constants.rho_STP,
            m_mode=0.08 / 2 * si.um,
            s_geom=1.4,
        )
        self.particle_reservoir_depth = particle_reservoir_depth
        self.dt = dt
        self.dz = dz
        self.precip = precip

        self.z_max = 3000 * si.metres
        self.t_max = 60 * si.minutes

        t_1 = 600 * si.s
        self.rho_times_w = (
            lambda t: rho_times_w_1 * np.sin(np.pi * t / t_1) if t < t_1 else 0
        )

        self._th = interp1d(
            (0.0 * si.m, 740.0 * si.m, 3260.00 * si.m),
            (297.9 * si.K, 297.9 * si.K, 312.66 * si.K),
            fill_value="extrapolate",
        )

        self.qv = interp1d(
            (-max(particle_reservoir_depth, 1), 0, 740, 3260),
            (0.015, 0.015, 0.0138, 0.0024),
            fill_value="extrapolate",
        )

        self.thd = (
            lambda z_above_reservoir: self.formulae.state_variable_triplet.th_dry(
                self._th(z_above_reservoir), self.qv(z_above_reservoir)
            )
        )

        g = self.formulae.constants.g_std
        self.rhod0 = self.formulae.state_variable_triplet.rho_d(
            p=p0,
            qv=self.qv(0 * si.m),
            theta_std=self._th(0 * si.m),
        )

        def drhod_dz(z_above_reservoir, rhod):
            if z_above_reservoir < 0:
                return 0
            qv = self.qv(z_above_reservoir)
            T = self.formulae.state_variable_triplet.T(
                rhod[0], self.thd(z_above_reservoir)
            )
            p = self.formulae.state_variable_triplet.p(rhod[0], T, qv)
            lv = self.formulae.latent_heat.lv(T)
            return self.formulae.hydrostatics.drho_dz(
                g, p, T, qv, lv
            )  # note: drho \approx drhod

        z_span = (-self.particle_reservoir_depth, self.z_max)
        z_points = np.linspace(*z_span, 2 * self.nz + 1)
        rhod_solution = solve_ivp(
            fun=drhod_dz,
            t_span=z_span,
            y0=np.asarray((self.rhod0,)),
            t_eval=z_points,
            max_step=dz / 2,
        )
        assert rhod_solution.success
        self.rhod = interp1d(z_points, rhod_solution.y[0])

        self.mpdata_settings = {"n_iters": 3, "iga": True, "fct": True, "tot": True}
        self.condensation_rtol_x = condensation.DEFAULTS.rtol_x
        self.condensation_rtol_thd = condensation.DEFAULTS.rtol_thd
        self.condensation_adaptive = True
        self.coalescence_adaptive = True

        self.r_bins_edges = np.logspace(
            np.log10(0.001 * si.um), np.log10(100 * si.um), 101, endpoint=True
        )
        self.cloud_water_radius_range = [1 * si.um, 50 * si.um]
        self.rain_water_radius_range = [50 * si.um, np.inf * si.um]

    @property
    def n_sd(self):
        return self.nz * self.n_sd_per_gridbox

    @property
    def nz(self):
        assert (
            self.particle_reservoir_depth / self.dz
            == self.particle_reservoir_depth // self.dz
        )
        nz = (self.z_max + self.particle_reservoir_depth) / self.dz
        assert nz == int(nz)
        return int(nz)

    @property
    def nt(self):
        nt = self.t_max / self.dt
        assert nt == int(nt)
        return int(nt)
