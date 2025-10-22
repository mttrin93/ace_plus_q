from ace_plus_q.calculator.asecalculator import MagTPCalculator
from ace_plus_q import TensorPotential
from ace_plus_q.potentials import MACE
from pyace import *


def init_magace_from_yaml(path):
    bbasisconf = BBasisConfiguration(path)
    mace = MACE(potconfig=bbasisconf)

    return mace

def init_magace_calc_from_yaml(path, cutoff):
    mace = init_magace_from_yaml(path)
    tp = TensorPotential(mace, mode="evaluate", compute_stress=True)
    calc = MagTPCalculator(
        model=tp,
        cutoff=cutoff,
        model_properties=["energy", "forces", "stress", "free_energy"],
    )

    return calc

def init_magace_calc_from_saved_model(path, cutoff):
    calc = MagTPCalculator(
        model=path,
        cutoff=cutoff,
        model_properties=["energy", "forces", "stress", "free_energy"],
    )

    return calc

