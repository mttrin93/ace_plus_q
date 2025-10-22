from typing import List, Final

# Structure-defined variables
# Structure id/name
DATA_STRUCTURE_ID: Final[str] = 'id'
# Number of atoms in the structure
DATA_NUM_OF_ATOMS: Final[str] = 'nat'
# Cartesian atomic positions
DATA_POSITIONS: Final[str] = 'positions'
# Cell atomic positions
DATA_SCALED_POSITIONS: Final[str] = 'scaled_positions'
# Unscaled positions 
DATA_UNSCALED_POSITIONS: Final[str] = 'unscaled_positions'
# Cell matrix
DATA_CELL: Final[str] = 'cell'
# Index of a central atom one for each neighbor
DATA_IND_I: Final[str] = 'ind_i'
# Indices of central atom's neighbors
DATA_IND_J: Final[str] = 'ind_j'
# Atomic number of central atom
DATA_MU_I: Final[str] = 'mu_i'
# Atomic numbers of neighbors
DATA_MU_J: Final[str] = 'mu_j'
# Unique elements in the structure
DATA_UNIQUE_ELEMENTS: Final[str] = 'unique_elements'
# Cartesian vector offset for neighbors positions
DATA_VECTOR_OFFSETS: Final[str] = 'vector_offsets'
# Cell unit offset for neighbors positions
DATA_CELL_OFFSETS: Final[str] = 'cell_offsets'
# Total number of neighbors in the structure
DATA_NUM_NEIGHBORS: Final[str] = 'nneighbors'
# Total energy of a structure
DATA_TOTAL_ENERGY: Final[str] = 'energy'
# Atomic forces in a structure
DATA_FORCES: Final[str] = 'forces'
# Stress in the matrix form
DATA_STRESS: Final[str] = 'stress'
# Number of electrons on an atom
DATA_ATOMIC_NUM_ELEC: Final[str] = 'nelec'
# Total number of electrons in the structure
DATA_TOTAL_NUM_ELEC: Final[str] = 'total_nelec'
# Electric charge on an atom
DATA_ATOMIC_CHRG: Final[str] = 'atmc_q'
# Total electric charge of a structure
DATA_TOTAL_CHRG: Final[str] = 'tot_q'
# Dipole moment on an atom
DATA_ATOMIC_DIPOLE_MOM: Final[str] = 'atmc_dpl_mom'
# Total dipole moment of a structure
DATA_TOTAL_DIPOLE_MOM: Final[str] = 'tot_dpl_mom'
# Electronegativity of free atom
DATA_CHI_0: Final[str] = 'chi_0'
# Idempotential of free atom
DATA_J_0: Final[str] = 'j_0'
# Qeq radii
DATA_RADII: Final[str] = 'qeq_radii'
# Electric field
DATA_ELECTRIC_FIELD: Final[str] = 'efield'
# Electronic centers
DATA_CENTERS: Final[str] = 'centers'
# Elastic constant Drude
DATA_K_0: Final[str] = 'k_0'
# Index of the central atom in the cell
DATA_IND_AT_I : Final[str] = 'ind_at_i'
# Index of the neighboring atoms in the cell
DATA_IND_AT_J : Final[str] = 'ind_at_j'
# Structure index for each central atom in the cell
DATA_IND_AT_BATCH : Final[str] = 'ind_at_batch'

# Index of a central atom one for each neighbor
DATA_IND_S_I: Final[str] = 'ind_s_i'
# Indices of central atom's neighbors
DATA_IND_S_J: Final[str] = 'ind_s_j'

# Magnetic moment vector on atoms
DATA_MAG_MOM: Final[str] = 'mag_mom'

TPATOMS_DATA_COLLECTION_KEYS: Final[List[str]] = [DATA_STRUCTURE_ID, DATA_NUM_OF_ATOMS, DATA_POSITIONS,
                                                  DATA_SCALED_POSITIONS, DATA_CELL, DATA_IND_I, DATA_IND_J, DATA_MU_I,
                                                  DATA_MU_J, DATA_UNIQUE_ELEMENTS, DATA_VECTOR_OFFSETS, DATA_STRESS,
                                                  DATA_CELL_OFFSETS, DATA_NUM_NEIGHBORS, DATA_TOTAL_ENERGY, DATA_FORCES,
                                                  DATA_ATOMIC_NUM_ELEC, DATA_TOTAL_NUM_ELEC, DATA_MAG_MOM,
                                                  DATA_ATOMIC_CHRG, DATA_TOTAL_CHRG, DATA_ATOMIC_DIPOLE_MOM,
                                                  DATA_TOTAL_DIPOLE_MOM, DATA_CHI_0, DATA_J_0, DATA_RADII,
                                                  DATA_ELECTRIC_FIELD,
                                                  DATA_IND_AT_I, DATA_IND_AT_J, DATA_IND_AT_BATCH,
                                                  DATA_IND_S_I, DATA_IND_S_J,
                                                  DATA_CENTERS, DATA_K_0, DATA_UNSCALED_POSITIONS]
TPATOMS_ENV_KEYS: Final[List[str]] = [DATA_IND_I, DATA_IND_J, DATA_MU_I, DATA_MU_J, DATA_VECTOR_OFFSETS,
                                      DATA_CELL_OFFSETS, DATA_NUM_NEIGHBORS,#]
                                      DATA_IND_AT_I, DATA_IND_AT_J, DATA_IND_AT_BATCH,
                                      DATA_IND_S_I, DATA_IND_S_J]
# Batch-defined variables
# Bond type index contiguous array
DATA_MU_IJ: Final[str] = 'mu_ij'
# Mapping of bond types in the contiguous array
DATA_SLICE_MU_IJ: Final[str] = 'slice_mu_ij'
# Energy loss weights
DATA_ENERGY_WEIGHTS: Final[str] = 'w_energy'
# Force loss weights
DATA_FORCE_WEIGHTS: Final[str] = 'w_forces'
# Stress loss weights
DATA_STRESS_WEIGHTS: Final[str] = 'w_stress'
# Atomic property to structure mapping
DATA_ATOMIC_STRUCTURE_MAP: Final[str] = 'atomic_map'
# Cell to atom mapping
DATA_CELL_ATOM_MAP: Final[str] = 'cell_atom_map'
# Cell to bond mapping
DATA_CELL_BOND_MAP: Final[str] = 'cell_bond_map'
# Number of structures in the batch
DATA_NUM_OF_STRUCTURES: Final[str] = 'num_struc'

TPBATCH_DATA_COLLECTION_KEYS: Final[List[str]] = [DATA_IND_I, DATA_IND_J, DATA_MU_I, DATA_MU_J, DATA_VECTOR_OFFSETS,
                                                  DATA_CELL_OFFSETS, DATA_NUM_NEIGHBORS, DATA_ATOMIC_STRUCTURE_MAP,
                                                  DATA_CELL_ATOM_MAP, DATA_CELL_BOND_MAP, DATA_SCALED_POSITIONS,
                                                  DATA_POSITIONS, DATA_CELL, DATA_ENERGY_WEIGHTS, DATA_FORCE_WEIGHTS,
                                                  DATA_TOTAL_ENERGY, DATA_FORCES, DATA_STRESS, DATA_STRESS_WEIGHTS,
                                                  DATA_ATOMIC_NUM_ELEC, DATA_TOTAL_NUM_ELEC, DATA_MAG_MOM,
                                                  DATA_ATOMIC_CHRG, DATA_TOTAL_CHRG, DATA_ATOMIC_DIPOLE_MOM,
                                                  DATA_TOTAL_DIPOLE_MOM, DATA_CHI_0, DATA_J_0, DATA_RADII,
                                                  DATA_ELECTRIC_FIELD, DATA_CENTERS, DATA_K_0,
                                                  DATA_IND_AT_I, DATA_IND_AT_J, DATA_IND_AT_BATCH,
                                                  DATA_IND_S_I, DATA_IND_S_J, DATA_UNSCALED_POSITIONS]

# Dataframe column name for TPAtoms
DATA_TPATOMS_DF_KEY: Final[str] = 'tp_atoms'

