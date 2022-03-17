from chempy import Substance
from PySDM.initialisation import spectra
from PySDM.physics import si
from PySDM.physics.constants_defaults import Mv, rho_w
from pystrict import strict

compounds = ("(NH4)2SO4", "insoluble")

molar_masses = {
    "(NH4)2SO4": Substance.from_formula("(NH4)2SO4").mass * si.gram / si.mole,
    "insoluble": 44 * si.g / si.mole,
}

densities = {
    "(NH4)2SO4": 1.77 * si.g / si.cm ** 3,
    "insoluble": 1.77 * si.g / si.cm ** 3,
}

is_soluble = {"(NH4)2SO4": True, "insoluble": False}

ionic_dissociation_phi = {"(NH4)2SO4": 3, "insoluble": 0}


def volume_fractions(mass_fractions: dict):
    return {
        k: (mass_fractions[k] / densities[k])
        / sum(mass_fractions[i] / densities[i] for i in compounds)
        for k in compounds
    }


def f_soluble_volume(mass_fractions: dict):
    volfrac = volume_fractions(mass_fractions)
    return sum(is_soluble[k] * volfrac[k] for k in compounds)


def volfrac_just_soluble(volfrac: dict):
    _masked = {k: (is_soluble[k]) * volfrac[k] for k in compounds}

    _denom = sum(list(_masked.values()))
    if _denom == 0.0:
        x = {k: 0.0 for k in compounds}
    else:
        x = {k: _masked[k] / _denom for k in compounds}
    return x


def kappa(mass_fractions: dict):
    volfrac = volume_fractions(mass_fractions)
    molar_volumes = {i: molar_masses[i] / densities[i] for i in compounds}
    volume_fractions_of_just_soluble = volfrac_just_soluble(volfrac)

    ns_per_vol = f_soluble_volume(mass_fractions) * sum(
        ionic_dissociation_phi[i]
        * volume_fractions_of_just_soluble[i]
        / molar_volumes[i]
        for i in compounds
    )

    return ns_per_vol * Mv / rho_w


class Aerosol:
    pass


@strict
class AerosolARG(Aerosol):
    def __init__(
        self,
        M2_sol: float = 0,
        M2_N: float = 100 / si.cm ** 3,
        M2_rad: float = 50 * si.nm,
    ):
        mode1 = {
            "(NH4)2SO4": 1.0,
            "insoluble": 0,
        }
        mode2 = {
            "(NH4)2SO4": M2_sol,
            "insoluble": (1 - M2_sol),
        }

        self.aerosol_modes = (
            {
                "kappa": kappa(mode1),
                "spectrum": spectra.Lognormal(
                    norm_factor=100.0 / si.cm ** 3, m_mode=50.0 * si.nm, s_geom=2.0
                ),
            },
            {
                "kappa": kappa(mode2),
                "spectrum": spectra.Lognormal(
                    norm_factor=M2_N, m_mode=M2_rad, s_geom=2.0
                ),
            },
        )


@strict
class AerosolWhitby(Aerosol):
    def __init__(self):
        nuclei = {
            "(NH4)2SO4": 1.0,
        }
        accum = {"(NH4)2SO4": 1.0}
        coarse = {"(NH4)2SO4": 1.0}

        self.aerosol_modes = (
            {
                "kappa": kappa(nuclei),
                "spectrum": spectra.Lognormal(
                    norm_factor=1000.0 / si.cm ** 3, m_mode=0.008 * si.um, s_geom=1.6
                ),
            },
            {
                "kappa": kappa(accum),
                "spectrum": spectra.Lognormal(
                    norm_factor=800 / si.cm ** 3, m_mode=0.034 * si.um, s_geom=2.1
                ),
            },
            {
                "kappa": kappa(coarse),
                "spectrum": spectra.Lognormal(
                    norm_factor=0.72 / si.cm ** 3, m_mode=0.46 * si.um, s_geom=2.2
                ),
            },
        )
