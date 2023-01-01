import numpy as np
from PySDM import Formulae
from PySDM.initialisation.aerosol_composition import DryAerosolMixture
from PySDM.initialisation.sampling import spectral_sampling as spec_sampling
from PySDM.physics import si
from pystrict import strict


@strict
class Settings:
    def __init__(
        self,
        dz: float,
        n_sd_per_mode: int,
        aerosol: DryAerosolMixture,
        model: str,
        spectral_sampling: type(spec_sampling.SpectralSampling),
        w: float = 0.32 * si.m / si.s,
        MAC: float = 1,
        HAC: float = 1,
        BDF: bool = False,
    ):
        assert model in (
            "Constant",
            "CompressedFilmOvadnevaite",
            "CompressedFilmRuehl",
            "SzyszkowskiLangmuir",
        )
        self.model = model
        self.n_sd_per_mode = n_sd_per_mode
        self.BDF = BDF

        if model in ("Constant", "CompressedFilmOvadnevaite"):
            form = Formulae(
                surface_tension=model,
                constants={
                    "sgm_org": 19 * si.mN / si.m,  # 40
                    "delta_min": 0.4 * si.nm,  # 0.1
                    "MAC": MAC,
                    "HAC": HAC,
                },
            )
        elif model == "SzyszkowskiLangmuir":
            form = Formulae(
                surface_tension=model,
                constants={
                    "RUEHL_nu_org": aerosol.modes[0]["nu_org"],
                    "RUEHL_A0": 7.5e-19 * si.m**2,  # 2.5e-19
                    "RUEHL_C0": 5e-7,  # 1e-5
                    "RUEHL_sgm_min": 21 * si.mN / si.m,  # 40
                    "MAC": MAC,
                    "HAC": HAC,
                },
            )
        elif model == "CompressedFilmRuehl":
            form = Formulae(
                surface_tension=model,
                constants={
                    "RUEHL_nu_org": aerosol.modes[0]["nu_org"],
                    "RUEHL_A0": 9.4e-19 * si.m**2,  # 2.5e-19
                    "RUEHL_C0": 10e-7,  # 1e-5
                    "RUEHL_sgm_min": 24 * si.mN / si.m,  # 40
                    "RUEHL_m_sigma": 1.8e17 * si.J / si.m**2,  # 1.3
                    "MAC": MAC,
                    "HAC": HAC,
                },
            )
        else:
            AssertionError()
        self.formulae = form
        const = self.formulae.constants

        self.aerosol = aerosol
        self.spectral_sampling = spectral_sampling

        max_altitude = 200 * si.m
        self.w = w
        self.t_max = max_altitude / self.w
        self.dt = dz / self.w
        self.output_interval = 1 * self.dt

        self.g = 9.81 * si.m / si.s**2

        self.p0 = 980 * si.mbar
        self.T0 = 280 * si.K
        pv0 = 0.99 * self.formulae.saturation_vapour_pressure.pvs_Celsius(
            self.T0 - const.T0
        )
        self.q0 = const.eps * pv0 / (self.p0 - pv0)

        self.cloud_radius_range = (0.5 * si.micrometre, np.inf)

        self.mass_of_dry_air = 44

        self.wet_radius_bins_edges = np.logspace(
            np.log10(4 * si.um), np.log10(12 * si.um), 128 + 1, endpoint=True
        )

    @property
    def rho0(self):
        const = self.formulae.constants
        rhod0 = self.formulae.trivia.p_d(self.p0, self.q0) / self.T0 / const.Rd
        return rhod0 * (1 + self.q0)

    @property
    def nt(self) -> int:
        nt = self.t_max / self.dt
        nt_int = round(nt)
        np.testing.assert_almost_equal(nt, nt_int)
        return nt_int

    @property
    def steps_per_output_interval(self) -> int:
        return int(self.output_interval / self.dt)

    @property
    def output_steps(self) -> np.ndarray:
        return np.arange(0, self.nt + 1, self.steps_per_output_interval)
