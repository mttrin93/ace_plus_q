import numpy as np

from src.ace_plus_q.data.neighborlist import PrimitiveNeighborListWrapper
from ase.calculators.calculator import Calculator, all_changes
from typing import Union, Optional, List

from ace_plus_q import TensorPotential
from src.ace_plus_q.data import TPAtomsDataContainer, TPBatch
from src.ace_plus_q.graphspecs import SPEC_EVALUATE_MODE, SPEC_ENERGY_EVAL, SPEC_ENERGY_FORCE_STRESS_EVAL
from src.ace_plus_q.graphspecs import SPEC_TRAIN_SCF_MODE, SPEC_ENERGY_FORCE_TRAIN


class TPCalculator(Calculator):
    """
    TensorPotential ASE calculator
    """
    implemented_properties = ['energy', 'forces', 'stress', 'charges']

    def __init__(self, model: Union[str, TensorPotential], cutoff: float = 8, skin: float = 0,
                 model_properties: Optional[List[str]] = None, reset_nl: bool = False,
                 atomic_nelec=None, total_nelec=None, chi_0=None, j_0=None, qeq_radii=None , dpl=None, **kwargs):
        Calculator.__init__(self, **kwargs)
        self.skin = skin
        self.cutoff = cutoff
        self.model_type = None
        self.nl = None
        self.reset_nl = reset_nl
        self.atomic_nelec = atomic_nelec
        self.total_nelec = total_nelec
        self.chi_0 = chi_0
        self.j_0 = j_0
        self.qeq_radii = qeq_radii
        self.total_dpl_mom = dpl

        if model_properties is not None:
            for prop in model_properties:
                assert prop in self.implemented_properties, f'Property {prop} is not in the ' \
                                                            f'list of {self.implemented_properties=}'
            self.compute_properties = model_properties
        else:
            self.compute_properties = ['energy', 'forces', 'charges']

        if 'stress' in self.compute_properties:
            self.list_of_tensors = SPEC_ENERGY_FORCE_STRESS_EVAL
        elif model.mode == SPEC_TRAIN_SCF_MODE:
            self.list_of_tensors = SPEC_ENERGY_FORCE_TRAIN
        else:
            self.list_of_tensors = SPEC_ENERGY_EVAL

        if model is None:
            raise ValueError(f'"model" parameter is not provided')
        elif isinstance(model, str):
            self.model = TensorPotential.load_model(model)
            self.model_type = 'loaded'
        elif isinstance(model, TensorPotential):
            assert model.mode == SPEC_EVALUATE_MODE or model.mode == SPEC_TRAIN_SCF_MODE,\
                f'TensorPotential model must be configured in ' \
                f'the "{SPEC_EVALUATE_MODE}" mode, but the mode is "{model.mode}".'
            self.model = model
            self.model_type = 'dynamic'
        else:
            raise ValueError(f'provided "model" is not recognized. Expecting either path to the tf.saved_model or '
                             f'instance of TensorPotential class.')

    def get_data(self, atoms):
        if self.reset_nl:
            self.nl = PrimitiveNeighborListWrapper(cutoffs=[self.cutoff * 0.5] * len(atoms), skin=self.skin,
                                                   self_interaction=False, bothways=True, use_scaled_positions=False)
            self.nl.update(atoms.get_pbc(), atoms.get_cell(), atoms.get_positions())
        else:
            if self.nl is None:
                self.nl = PrimitiveNeighborListWrapper(cutoffs=[self.cutoff * 0.5] * len(atoms), skin=self.skin,
                                                       self_interaction=False, bothways=True,
                                                       use_scaled_positions=False)
                self.nl.update(atoms.get_pbc(), atoms.get_cell(), atoms.get_positions())
            else:
                self.nl.update(atoms.get_pbc(), atoms.get_cell(), atoms.get_positions())

        if self.model.mode == SPEC_TRAIN_SCF_MODE:
            # data = TPBatch([TPAtomsDataContainer(atoms, cutoff=self.cutoff, neighborlist=self.nl, atomic_nelec=
            #                self.atomic_nelec, total_nelec=self.total_nelec, chi_0=self.chi_0,
            #                j_0=self.j_0, qeq_radii=self.qeq_radii)], batch_size=1,
            #                list_of_elements=self.model.potential.get_chemical_symbols(),
            #                list_of_bond_symbol_combinations=self.model.potential.get_bond_symbol_combinations(),
            #                bond_indexing=self.model.potential.bond_indexing
            #                ).batches[0]            
            data = TPBatch([TPAtomsDataContainer(atoms, atomic_nelec=self.atomic_nelec,
                           total_nelec=self.total_nelec, chi_0=self.chi_0,
                           j_0=self.j_0, qeq_radii=self.qeq_radii, atomic_chrg=self.atomic_nelec,
                           total_dpl_mom=self.total_dpl_mom)], batch_size=1,
                           list_of_elements=self.model.potential.get_chemical_symbols(),
                           list_of_bond_symbol_combinations=self.model.potential.get_bond_symbol_combinations(),
                           bond_indexing=self.model.potential.bond_indexing
                           ).batches[0]
        else:
            data = TPBatch([TPAtomsDataContainer(atoms, cutoff=self.cutoff, neighborlist=self.nl)], batch_size=1,
                           list_of_elements=self.model.potential.get_chemical_symbols(),
                           list_of_bond_symbol_combinations=self.model.potential.get_bond_symbol_combinations(),
                           bond_indexing=self.model.potential.bond_indexing
                           ).batches[0]

        if self.model_type == 'loaded':
            converted = TensorPotential.convert_to_tensor(data)
            converted = {k: v for k, v in converted.items() if k in self.list_of_tensors}
            self.data = converted
        else:
            self.data = data

    def calculate(self, atoms=None, properties=['energy', 'forces', 'stress'],
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        self.forces = np.empty((len(atoms), 3))
        # self.charges = np.empty((len(atoms)))
        self.stress = np.empty((1, 3, 3))
        self.energy = 0.0
        results = {}

        self.get_data(atoms)

        if self.model.mode == SPEC_TRAIN_SCF_MODE:
            if self.model_type == 'loaded':
                _, output = self.model.evaluate_branch(self.data)
            else:
                _, output = self.model.evaluate(self.data)
        else:
            if self.model_type == 'loaded':
                output = self.model.evaluate_branch(self.data)
            else:
                output = self.model.evaluate(self.data)

        # self.charges = output[2].numpy()
        # results['charges'] = self.charges.reshape(-1).astype(np.float64)

        if 'energy' in self.compute_properties:
            self.energy = output[0].numpy()
            results['energy'] = np.float64(self.energy.reshape(-1, ))
        if 'forces' in self.compute_properties:
            self.forces = output[1].numpy()
            results['forces'] = self.forces.astype(np.float64)
        if 'stress' in self.compute_properties:
            self.stress = output[2].numpy()
            results['stress'] = self.stress.reshape(3, 3).astype(np.float64)

        self.results = results

