from ace_plus_q.data.datakeys import *

SPEC_EVALUATE_MODE: Final[str] = 'evaluate'
SPEC_TRAIN_MODE: Final[str] = 'train'
SPEC_TRAIN_SCF_MODE: Final[str] = 'scf_train'
SPEC_PAIRSTYLE_MODE: Final[str] = 'pairstyle'

SPEC_GRAPH_MODES: Final[List[str]] = [SPEC_EVALUATE_MODE, SPEC_TRAIN_MODE, SPEC_TRAIN_SCF_MODE, SPEC_PAIRSTYLE_MODE]

SPEC_ENERGY_EVAL: Final[List[str]] = [DATA_POSITIONS, DATA_NUM_OF_ATOMS, DATA_NUM_OF_STRUCTURES, DATA_IND_I,
                                      DATA_IND_J, DATA_MU_IJ, DATA_SLICE_MU_IJ, DATA_VECTOR_OFFSETS,
                                      DATA_ATOMIC_STRUCTURE_MAP, DATA_MU_I, DATA_MU_J, #]
                                      DATA_IND_AT_I, DATA_IND_AT_J, DATA_IND_AT_BATCH,
                                      DATA_IND_S_I, DATA_IND_S_J]
SPEC_ENERGY_FORCE_EVAL: Final[List[str]] = SPEC_ENERGY_EVAL
SPEC_ENERGY_FORCE_STRESS_EVAL: Final[List[str]] = [DATA_SCALED_POSITIONS, DATA_NUM_OF_ATOMS, DATA_NUM_OF_STRUCTURES,
                                                   DATA_IND_I, DATA_IND_J, DATA_MU_I, DATA_MU_J, DATA_MU_IJ,
                                                   DATA_SLICE_MU_IJ, DATA_CELL_OFFSETS, DATA_CELL,
                                                   DATA_ATOMIC_STRUCTURE_MAP, DATA_CELL_ATOM_MAP, DATA_CELL_BOND_MAP, #]
                                                   DATA_IND_AT_I, DATA_IND_AT_J, DATA_IND_AT_BATCH,
                                                   DATA_IND_S_I, DATA_IND_S_J, DATA_UNSCALED_POSITIONS]
SPEC_ENERGY_FORCE_PAIR: Final[List[str]] = [DATA_POSITIONS, DATA_NUM_OF_ATOMS, DATA_NUM_OF_STRUCTURES, DATA_IND_I,
                                            DATA_IND_J, DATA_MU_IJ, DATA_SLICE_MU_IJ,
                                            DATA_MU_I, DATA_MU_J, DATA_VECTOR_OFFSETS, #]
                                            DATA_IND_AT_I, DATA_IND_AT_J, DATA_IND_AT_BATCH,
                                            DATA_IND_S_I, DATA_IND_S_J]

SPEC_ENERGY_TRAIN: Final[List[str]] = [DATA_POSITIONS, DATA_NUM_OF_ATOMS, DATA_NUM_OF_STRUCTURES, DATA_IND_I,
                                       DATA_IND_J, DATA_MU_IJ, DATA_SLICE_MU_IJ, DATA_VECTOR_OFFSETS,
                                       DATA_ATOMIC_STRUCTURE_MAP, DATA_TOTAL_ENERGY, DATA_ENERGY_WEIGHTS,
                                       DATA_MU_I, DATA_MU_J, DATA_CENTERS,
                                       DATA_IND_AT_I, DATA_IND_AT_J, DATA_IND_AT_BATCH,
                                       DATA_IND_S_I, DATA_IND_S_J, DATA_UNSCALED_POSITIONS]
# SPEC_ENERGY_FORCE_TRAIN: Final[List[str]] = [DATA_POSITIONS, DATA_NUM_OF_ATOMS, DATA_NUM_OF_STRUCTURES, DATA_IND_I,
#                                              DATA_IND_J, DATA_MU_IJ, DATA_SLICE_MU_IJ, DATA_VECTOR_OFFSETS,
#                                              DATA_ATOMIC_STRUCTURE_MAP, DATA_TOTAL_ENERGY, DATA_ENERGY_WEIGHTS,
#                                              DATA_FORCES, DATA_FORCE_WEIGHTS, DATA_MU_I, DATA_MU_J]
SPEC_ENERGY_FORCE_TRAIN: Final[List[str]] = [DATA_POSITIONS, DATA_NUM_OF_ATOMS, DATA_NUM_OF_STRUCTURES, DATA_IND_I,
                                             DATA_IND_J, DATA_MU_IJ, DATA_SLICE_MU_IJ, DATA_VECTOR_OFFSETS,
                                             DATA_ATOMIC_STRUCTURE_MAP, DATA_TOTAL_ENERGY, DATA_ENERGY_WEIGHTS,
                                             DATA_FORCES, DATA_FORCE_WEIGHTS, DATA_MU_I, DATA_MU_J, DATA_CELL,
                                             DATA_CELL_ATOM_MAP, DATA_CENTERS, 
                                             DATA_IND_AT_I, DATA_IND_AT_J, DATA_IND_AT_BATCH,
                                             DATA_IND_S_I, DATA_IND_S_J, DATA_UNSCALED_POSITIONS]

SPEC_ENERGY_FORCE_STRESS_TRAIN: Final[List[str]] = [DATA_SCALED_POSITIONS, DATA_NUM_OF_ATOMS, DATA_NUM_OF_STRUCTURES,
                                                    DATA_IND_I, DATA_IND_J, DATA_MU_IJ, DATA_SLICE_MU_IJ,
                                                    DATA_CELL_OFFSETS, DATA_CELL, DATA_ATOMIC_STRUCTURE_MAP,
                                                    DATA_CELL_ATOM_MAP, DATA_CELL_BOND_MAP, DATA_TOTAL_ENERGY,
                                                    DATA_ENERGY_WEIGHTS, DATA_FORCES, DATA_FORCE_WEIGHTS, DATA_STRESS,
                                                    DATA_STRESS_WEIGHTS, DATA_MU_I, DATA_MU_J, #]
                                                    DATA_IND_AT_I, DATA_IND_AT_J, DATA_IND_AT_BATCH,
                                                    DATA_IND_S_I, DATA_IND_S_J, DATA_UNSCALED_POSITIONS]

SPEC_LIST_OF_INT_TENSORS: Final[List[str]] = [DATA_IND_I, DATA_IND_J, DATA_MU_I, DATA_MU_J, DATA_MU_IJ,
                                              DATA_SLICE_MU_IJ, DATA_NUM_OF_ATOMS, DATA_CELL_ATOM_MAP,
                                              DATA_CELL_BOND_MAP, DATA_ATOMIC_STRUCTURE_MAP, DATA_NUM_OF_STRUCTURES, #]
                                              DATA_IND_AT_I, DATA_IND_AT_J, DATA_IND_AT_BATCH,
                                              DATA_IND_S_I, DATA_IND_S_J]
SPEC_LIST_OF_FLOAT_TENSORS: Final[List[str]] = [DATA_POSITIONS, DATA_SCALED_POSITIONS, DATA_CELL, DATA_VECTOR_OFFSETS,
                                                DATA_CELL_OFFSETS, DATA_TOTAL_ENERGY, DATA_FORCES, DATA_ENERGY_WEIGHTS,
                                                DATA_FORCE_WEIGHTS, DATA_STRESS, DATA_STRESS_WEIGHTS,
                                                DATA_ATOMIC_NUM_ELEC, DATA_TOTAL_NUM_ELEC, DATA_MAG_MOM,
                                                DATA_ATOMIC_CHRG, DATA_TOTAL_CHRG, DATA_ATOMIC_DIPOLE_MOM,
                                                DATA_TOTAL_DIPOLE_MOM, DATA_CHI_0, DATA_J_0, DATA_RADII,
                                                DATA_ELECTRIC_FIELD, DATA_CENTERS, DATA_K_0, DATA_UNSCALED_POSITIONS]

SPEC_OPTIONAL_DATA_ENTRIES: Final[List[str]] = [DATA_ATOMIC_NUM_ELEC, DATA_TOTAL_NUM_ELEC, DATA_MAG_MOM,
                                                DATA_ATOMIC_CHRG, DATA_TOTAL_CHRG, DATA_ATOMIC_DIPOLE_MOM,
                                                DATA_TOTAL_DIPOLE_MOM, DATA_CHI_0, DATA_J_0, DATA_RADII,
                                                DATA_ELECTRIC_FIELD, DATA_CENTERS, DATA_K_0]

# Loss specification parameters
SPEC_LOSS_ENERGY_NORM_TYPE: Final[str] = 'energy_loss_norm'
SPEC_LOSS_ENERGY_NORM_PER_ATOM: Final[str] = 'per-atom'
SPEC_LOSS_ENERGY_NORM_PER_STRUC: Final[str] = 'per-structure'
SPEC_LOSS_VALID_ENERGY_NORM_TYPES: Final[List[str]] = [SPEC_LOSS_ENERGY_NORM_PER_ATOM, SPEC_LOSS_ENERGY_NORM_PER_STRUC]

SPEC_AUX_LOSS_FACTORS = 'aux_loss_factors'
SPEC_LOSS_L1_REG_FACTOR = 'l1_reg'
SPEC_LOSS_L2_REG_FACTOR = 'l2_reg'
SPEC_LOSS_ENERGY_FACTOR: Final[str] = 'loss_energy_factor'
SPEC_LOSS_FORCE_FACTOR: Final[str] = 'loss_force_factor'
SPEC_LOSS_STRESS_FACTOR: Final[str] = 'loss_stress_factor'
SPEC_LOSS_SCF_FACTOR: Final[str] = 'loss_scf_factor'
