import numpy as np
import pandas as pd
import tensorflow as tf
import itertools

from typing import Union, Optional, Any
import numpy.typing as npt

from ase.neighborlist import NewPrimitiveNeighborList
from ase import Atoms
from ace_plus_q.data.datakeys import *
from ace_plus_q.data.symbols import symbol_to_atomic_number
from ace_plus_q.data.neighborlist import PrimitiveNeighborListWrapper


class TPAtomsDataContainer:
    __slots__ = TPATOMS_DATA_COLLECTION_KEYS

    def __init__(self,
                 ase_atoms: Atoms,
                 cutoff: float = 6.,
                 energy: npt.NDArray[np.float64] = None,
                 forces: npt.NDArray[np.float64] = None,
                 stress: npt.NDArray[np.float64] = None,
                 atomic_nelec: npt.NDArray[np.float64] = None,
                 total_nelec: float = None,
                 mag_mom: npt.NDArray[np.float64] = None,
                 atomic_chrg: npt.NDArray[np.float64] = None,
                 total_chrg: npt.NDArray[np.float64] = None,
                 atomic_dpl_mom: npt.NDArray[np.float64] = None,
                 total_dpl_mom: npt.NDArray[np.float64] = None,
                 chi_0: npt.NDArray[np.float64] = None,
                 j_0: npt.NDArray[np.float64] = None,
                 qeq_radii: npt.NDArray[np.float64] = None,
                 centers: npt.NDArray[np.float64] = None,
                 efield: npt.NDArray[np.float64] = None,
                 k_0: npt.NDArray[np.float64] = None,
                 neighborlist: PrimitiveNeighborListWrapper = None,
                 verbose: bool = True,
                 struc_id: Any = None):
        setattr(self, DATA_STRUCTURE_ID, struc_id)

        self.set_structure_property(energy, DATA_TOTAL_ENERGY, (1, 1))
        self.set_structure_property(forces, DATA_FORCES, (len(ase_atoms), 3))
        self.set_structure_property(stress, DATA_STRESS, (1, 3, 3))
        self.set_structure_property(atomic_nelec, DATA_ATOMIC_NUM_ELEC, (len(ase_atoms), 1))
        self.set_structure_property(total_nelec, DATA_TOTAL_NUM_ELEC, (1, 1))
        self.set_structure_property(atomic_chrg, DATA_ATOMIC_CHRG, (len(ase_atoms), 1))
        self.set_structure_property(total_chrg, DATA_TOTAL_CHRG, (1, 1))
        self.set_structure_property(atomic_dpl_mom, DATA_ATOMIC_DIPOLE_MOM, (len(ase_atoms), 3))
        self.set_structure_property(total_dpl_mom, DATA_TOTAL_DIPOLE_MOM, (1, 3))
        self.set_structure_property(chi_0, DATA_CHI_0, (len(ase_atoms), 1))
        self.set_structure_property(j_0, DATA_J_0, (len(ase_atoms), 1))
        self.set_structure_property(qeq_radii, DATA_RADII, (len(ase_atoms), 1))
        self.set_structure_property(centers, DATA_CENTERS, (len(ase_atoms), 3))
        self.set_structure_property(k_0, DATA_K_0, (len(ase_atoms), 1))
        self.set_structure_property(efield, DATA_ELECTRIC_FIELD, (len(ase_atoms), 3))

        self.set_structure_property(mag_mom, DATA_MAG_MOM, (len(ase_atoms), 3))

        setattr(self, DATA_NUM_OF_ATOMS, 0.)

        self.construct_tp_atoms(ase_atoms, cutoff, neighborlist, verbose)

    def set_structure_property(self, struc_property: Any, key: str, struc_property_shape: [list, tuple]):
        if struc_property is not None:
            struc_property = np.array(struc_property).reshape(struc_property_shape)
        else:
            struc_property = np.zeros(struc_property_shape)
        setattr(self, key, struc_property)

    def get_number_of_atoms(self) -> int:
        return getattr(self, DATA_NUM_OF_ATOMS)[0]

    def get_number_of_neighbors(self) -> List[int]:
        return getattr(self, DATA_NUM_NEIGHBORS)

    def get_unique_elements(self) -> List:
        return getattr(self, DATA_UNIQUE_ELEMENTS)

    def get_positions(self) -> npt.NDArray[np.float64]:
        return getattr(self, DATA_POSITIONS)

    def get_scaled_positions(self) -> npt.NDArray[np.float64]:
        return getattr(self, DATA_SCALED_POSITIONS)
        
    def get_unscaled_positions(self) -> npt.NDArray[np.float64]:
        return getattr(self, DATA_UNSCALED_POSITIONS)

    def get_energy(self) -> npt.NDArray[np.float64]:
        return getattr(self, DATA_TOTAL_ENERGY)

    def get_forces(self) -> npt.NDArray[np.float64]:
        return getattr(self, DATA_FORCES)

    def get_stress(self) -> npt.NDArray[np.float64]:
        return getattr(self, DATA_STRESS)

    def get_mag_mom(self) -> npt.NDArray[np.float64]:
        return getattr(self, DATA_MAG_MOM)

    def construct_tp_atoms(self, ase_atoms: Atoms, cutoff: float, neighborlist: Any, verbose: bool):
        ase_atoms = copy_atoms(ase_atoms)
        
        unscaled_pos = ase_atoms.get_positions().astype(np.float64)
        setattr(self, DATA_UNSCALED_POSITIONS, unscaled_pos)

        pbc_atoms = enforce_pbc(ase_atoms, cutoff)
        nat = len(pbc_atoms)
        setattr(self, DATA_NUM_OF_ATOMS, np.array([nat]))

        cell = pbc_atoms.get_cell().reshape(1, 3, 3).astype(np.float64)
        setattr(self, DATA_CELL, cell)

        scaled_pos = pbc_atoms.get_scaled_positions().astype(np.float64)
        setattr(self, DATA_SCALED_POSITIONS, scaled_pos)

        # positions = pbc_atoms.get_positions().astype(np.float64)
        positions = np.matmul(scaled_pos, cell.reshape(3, 3))
        setattr(self, DATA_POSITIONS, positions)

        if neighborlist is None:
            nghbrs_lst = PrimitiveNeighborListWrapper(cutoffs=[cutoff * 0.5] * nat, skin=0, self_interaction=False,
                                                      bothways=True, use_scaled_positions=False)
        else:
            nghbrs_lst = neighborlist
        nghbrs_lst.update(pbc_atoms.get_pbc(), pbc_atoms.get_cell(), pbc_atoms.get_positions())

        atomic_numbers = pbc_atoms.get_atomic_numbers()

        elems = [list(symbol_to_atomic_number.keys())[list(symbol_to_atomic_number.values()).index(z)]
                 for z in atomic_numbers]
        setattr(self, DATA_UNIQUE_ELEMENTS, list(np.unique(elems)))

        dcell = np.linalg.pinv(cell.reshape(3, 3))
        for key in TPATOMS_ENV_KEYS:
            setattr(self, key, [])
        for i in range(nat):
            ind, offset, dv = nghbrs_lst.get_neighbors(i)
            if len(ind) < 1:
                if verbose:
                    id = getattr(self, DATA_STRUCTURE_ID)
                    print(f'Found an atom with no neighbors within cutoff.'
                          f' Adding a fictitious neighbor beyond cutoff. Structure id: {id}')
                ind = np.array([0])
                dv = np.dot(cell, np.array([1, 1, 1]).reshape(3, 1)).reshape(1, 3) + cutoff
            sort = np.argsort(ind)
            ind = ind[sort]
            dv = dv[sort]  # pos_j = pos_i + dv
            dv_j = (np.take(positions, [i] * len(ind), axis=0) + dv) - np.take(positions, ind, axis=0)

            cell_offset = np.rint(np.dot(((dv + np.take(positions, [i] * len(ind), axis=0))
                                          - positions[ind]), dcell)).astype(np.int64)

            # tf.print('nat', i, nat, np.array([i] * len(ind)), summarize=-1)

            getattr(self, DATA_IND_I).extend(np.array([i] * len(ind)))
            getattr(self, DATA_MU_I).extend(np.array([atomic_numbers[i]] * len(ind)))  # Not substructing 1 here
            getattr(self, DATA_IND_J).extend(ind)
            getattr(self, DATA_MU_J).extend(np.take(atomic_numbers, ind))  # And here
            getattr(self, DATA_VECTOR_OFFSETS).extend(dv_j)
            getattr(self, DATA_CELL_OFFSETS).extend(cell_offset)
            getattr(self, DATA_NUM_NEIGHBORS).extend([len(ind)])

            getattr(self, DATA_IND_S_I).extend(np.array([i] * len(ind)))
            getattr(self, DATA_IND_S_J).extend(ind)

        combs_i = []
        combs_j = []
        for i in range(nat):
            for j in range(nat):
                if j >= i:
                    combs_i.append(i)
                    combs_j.append(j)

        setattr(self, DATA_IND_AT_I, np.array(combs_i))
        setattr(self, DATA_IND_AT_J, np.array(combs_j))

        for key in TPATOMS_ENV_KEYS:
            setattr(self, key, np.array(getattr(self, key)))

        del ase_atoms, nghbrs_lst


class TPBatch():
    def __init__(self, data: Union[List[TPAtomsDataContainer], pd.DataFrame], batch_size: int = 1, shuffle: bool = True,
                 energy_weights: List[npt.NDArray[np.float64]] = None,
                 force_weight: List[npt.NDArray[np.float64]] = None,
                 stress_weight: List[npt.NDArray[np.float64]] = None,
                 normalize_weights: Optional[bool] = True,
                 elements_sorting_type: str = 'alphabetic',
                 list_of_elements: List[str] = None,
                 bond_indexing: str = None,
                 list_of_bond_symbol_combinations: Optional[List[tuple[str, str]]] = None):
        self.batch_size = batch_size
        self.normalize_weights = normalize_weights
        self.elements_sorting = elements_sorting_type
        self.bond_indexing = bond_indexing
        self.list_of_elements = None
        self.list_of_bond_symbol_combinations = None
        if list_of_elements is not None:
            # It is possible that not all the elements will be present in one batch,
            # so the total list of necessary elements should come from the potential.
            self.list_of_elements = list_of_elements

        if list_of_bond_symbol_combinations is not None:
            self.list_of_bond_symbol_combinations = list_of_bond_symbol_combinations  # All possible combinations might not be necessary

        self.batches = self.process_data(data, energy_weights, force_weight, stress_weight, shuffle)

    def process_data(self, data, energy_weights, force_weight, stress_weight, shuffle):
        e_w = energy_weights
        f_w = force_weight
        s_w = stress_weight
        if isinstance(data, pd.DataFrame):
            data.dropna(subset=[DATA_TPATOMS_DF_KEY], inplace=True)
            data_list = data[DATA_TPATOMS_DF_KEY].to_list()
            if e_w is None:
                try:
                    e_w = list(data[DATA_ENERGY_WEIGHTS])
                except:
                    e_w = [np.ones_like(d.get_energy()) for d in data_list]
                    if self.normalize_weights:
                        e_norm = np.sum([np.sum(w) for w in e_w])
                        e_w = [w / e_norm for w in e_w]
            if f_w is None:
                try:
                    f_w = list(data[DATA_FORCE_WEIGHTS])
                except:
                    f_w = [np.ones(d.get_number_of_atoms()).reshape(-1, 1) for d in data_list]
                    if self.normalize_weights:
                        f_norm = np.sum([np.sum(w) for w in f_w])
                        f_w = [w / f_norm for w in f_w]
            if s_w is None:
                try:
                    s_w = list(data[DATA_STRESS_WEIGHTS])
                except:
                    s_w = [np.ones(1).reshape([1, 1, 1]) for _ in data_list]
                    if self.normalize_weights:
                        s_norm = np.sum([np.sum(w) for w in s_w])
                        s_w = [w / s_norm for w in s_w]
        elif isinstance(data, list):
            data_list = data
            if e_w is None:
                e_w = [np.ones_like(d.get_energy()) for d in data_list]
                if self.normalize_weights:
                    e_norm = np.sum([np.sum(w) for w in e_w])
                    e_w = [w / e_norm for w in e_w]
            if f_w is None:
                f_w = [np.ones(d.get_number_of_atoms()).reshape(-1, 1) for d in data_list]
                if self.normalize_weights:
                    f_norm = np.sum([np.sum(w) for w in f_w])
                    f_w = [w / f_norm for w in f_w]
                # f_w = [np.ones(d.get_number_of_atoms()).reshape(-1, 1)/d.get_number_of_atoms().reshape(-1, 1)/3
                #        for d in data_list]
            if s_w is None:
                s_w = [np.ones(1).reshape([1, 1, 1]) for _ in data_list]
                if self.normalize_weights:
                    s_norm = np.sum([np.sum(w) for w in s_w])
                    s_w = [w / s_norm for w in s_w]
        else:
            raise ValueError(f'Provided data is not recognized. Expected {list} or {pd.DataFrame}, got {type(data)}')

        if self.list_of_elements is None:
            elems = np.unique(list(itertools.chain.from_iterable([d.get_unique_elements() for d in data_list])))
        else:
            elems = self.list_of_elements
        self.elems = sort_elements(list(elems), self.elements_sorting)

        if self.list_of_bond_symbol_combinations is None:
            self.list_of_bond_symbol_combinations = [p for p in itertools.product(self.elems, repeat=2)]

        self.nat = [d.get_number_of_atoms() for d in data_list]
        self.total_nat = np.sum(self.nat).astype(np.int32)
        self.total_nneighbors = np.sum([np.sum(d.get_number_of_atoms()) for d in data_list])
        self.nneighbors = [d.get_number_of_neighbors() for d in data_list]

        index = np.arange(0, len(data_list))
        if shuffle:
            np.random.seed(42)
            np.random.shuffle(index)
        batch_index_split = chunks(index, self.batch_size)
        self.nbatches = len(batch_index_split)
        data_batches = []
        f_w_batches = []
        e_w_batches = []
        s_w_batches = []
        for batch_idx in batch_index_split:
            data_batches.append([data_list[i] for i in batch_idx])
            e_w_batches.append([e_w[i] for i in batch_idx])
            f_w_batches.append([f_w[i] for i in batch_idx])
            s_w_batches.append([s_w[i] for i in batch_idx])

        return [self.make_batch(data_batches[i], e_w_batches[i], f_w_batches[i], s_w_batches[i]) for i in
                range(self.nbatches)]

    def make_batch(self, data: List[TPAtomsDataContainer],
                   e_w: List[np.array], f_w: List[np.array], s_w: List[np.array]) -> dict[str, np.array]:
        batch = self.get_empty_batch()

        count_atoms = 0
        count_structures = 0
        for j, entry in enumerate(data):
            nat = entry.get_number_of_atoms()
            npair = np.sum(entry.get_number_of_neighbors())

            # tf.print('nat', j, nat, count_atoms)

            #for key in TPATOMS_ENV_KEYS:
            for key in TPBATCH_DATA_COLLECTION_KEYS:
                if (key == DATA_IND_I) or (key == DATA_IND_J):
                    batch[key].append(getattr(entry, key) + count_atoms)
                elif key == DATA_CELL_ATOM_MAP:
                    batch[DATA_CELL_ATOM_MAP].append(np.repeat(count_structures, nat))
                elif key == DATA_CELL_BOND_MAP:
                    batch[DATA_CELL_BOND_MAP].append(np.repeat(count_structures, npair))
                elif key == DATA_ENERGY_WEIGHTS:
                    batch[DATA_ENERGY_WEIGHTS].append(e_w[j])
                elif key == DATA_FORCE_WEIGHTS:
                    batch[DATA_FORCE_WEIGHTS].append(f_w[j])
                elif key == DATA_STRESS_WEIGHTS:
                    batch[DATA_STRESS_WEIGHTS].append(s_w[j])
                elif key == DATA_ATOMIC_STRUCTURE_MAP:
                    batch[DATA_ATOMIC_STRUCTURE_MAP].append(np.repeat(j, nat))
                elif key == DATA_IND_AT_BATCH:
                    #batch[DATA_IND_AT_BATCH].append(np.repeat(j, nat*nat))
                    batch[DATA_IND_AT_BATCH].append(np.repeat(j, nat*(nat+1)/2))
                else:
                    batch[key].append(getattr(entry, key))

            count_atoms += nat
            count_structures += 1

        batch = {k: np.concatenate(v, axis=0) for k, v in batch.items()}
        batch[DATA_NUM_OF_ATOMS] = np.reshape(count_atoms, [])
        batch[DATA_NUM_OF_STRUCTURES] = np.reshape(count_structures, [])
        batch = self.process_mu_ij(batch)

        return batch

    def process_mu_ij(self, batch: dict[str, np.array]) -> dict[str, np.array]:
        # bond_comb = [p for p in itertools.product(self.elems, repeat=2)]
        bond_comb = self.list_of_bond_symbol_combinations
        mu_i = batch[DATA_MU_I]
        mu_j = batch[DATA_MU_J]
        if self.bond_indexing == 'delta_all':
            collect = []
            sizes = [0]
            idx = np.arange(np.sum(batch[DATA_NUM_NEIGHBORS]))
            for i, c in enumerate(bond_comb):
                mask = np.logical_and((mu_i == symbol_to_atomic_number[c[0]]), ((mu_j == symbol_to_atomic_number[c[1]])))
                collect.append(idx[mask])
                sizes.append(len(idx[mask]))

            ind_i = batch[DATA_IND_I]
            at_mu_i = np.array([np.mean(mu_i[ind_i == i]).astype(np.int32) for i in range(batch[DATA_NUM_OF_ATOMS])])
            idx = np.arange(batch[DATA_NUM_OF_ATOMS])
            for e in self.elems:
                mask = (at_mu_i == symbol_to_atomic_number[e])
                collect.append(idx[mask])
                sizes.append(len(idx[mask]))
            ind_mu_ij, slices_mu_ij = np.concatenate(collect, axis=0), np.cumsum(sizes)

        elif self.bond_indexing == 'symmetric_bonds':
            ind_mu_ij = np.zeros(np.sum(batch[DATA_NUM_NEIGHBORS]))
            for i, c in enumerate(bond_comb):
                mask_1 = np.logical_and((mu_i == symbol_to_atomic_number[c[0]]),
                                      ((mu_j == symbol_to_atomic_number[c[1]])))
                mask_2 = np.logical_and((mu_i == symbol_to_atomic_number[c[1]]),
                                        ((mu_j == symbol_to_atomic_number[c[0]])))
                mask = np.logical_or(mask_1, mask_2)
                ind_mu_ij[mask] = i
            slices_mu_ij = [0]
            for i, e in enumerate(self.elems):
                mu_i[(mu_i == symbol_to_atomic_number[e])] = i
                mu_j[(mu_j == symbol_to_atomic_number[e])] = i
            batch[DATA_MU_I] = mu_i
            batch[DATA_MU_J] = mu_j
        else:
            ind_mu_ij = None
            slices_mu_ij = None
            raise ValueError(f'{self.bond_indexing=} is unknown')
        batch[DATA_MU_IJ] = ind_mu_ij
        batch[DATA_SLICE_MU_IJ] = slices_mu_ij

        return batch

    def get_empty_batch(self):
        batch_dict = {}
        for key in TPBATCH_DATA_COLLECTION_KEYS:
            batch_dict[key] = []

        return batch_dict


def sort_elements(list_of_elements: list, sorting_type: str = 'alphabetic') -> list:
    if sorting_type == 'alphabetic':
        list_of_elements.sort(key=str.lower)
        return list_of_elements
    elif sorting_type == 'atomic-number':
        pass
    else:
        raise ValueError('Unknown type of elements sorting "{}"'.format(sorting_type))


def get_nghbrs(atoms, list_at_ind=None, cutoff=8.7, skin=0, verbose=False):
    nghbrs_lst = NewPrimitiveNeighborList(cutoffs=[cutoff * 0.5] * len(atoms), skin=skin,
                                          self_interaction=False, bothways=True, use_scaled_positions=True)

    nghbrs_lst.update(atoms.get_pbc(), atoms.get_cell(), atoms.get_scaled_positions())
    # cell = atoms.get_cell()
    ind_i = []
    mu_i = []
    ind_j = []
    mu_j = []
    offsts = []
    at_nums = atoms.get_atomic_numbers()
    if list_at_ind is not None:
        list_of_atoms = list_at_ind
    else:
        list_of_atoms = np.arange(0, len(atoms))
    check = 0
    for i in list_of_atoms:
        ind, off = nghbrs_lst.get_neighbors(i)
        if len(ind) < 1:
            check += 1

        sort = np.argsort(ind)
        ind = ind[sort]
        off = off[sort]
        ind_i.append([i] * len(ind))
        mu_i.append([at_nums[i]] * len(ind))  # Not substructing 1 here
        ind_j.append(ind)
        mu_j.append(np.take(at_nums, ind))  # And here
        offsts.append(off)
    if check == 0:
        return np.hstack(ind_i), np.hstack(ind_j), np.hstack(mu_i), np.hstack(mu_j), np.vstack(offsts).astype(
            np.float64)
    else:
        if verbose:
            print('Found an atom with no neighbors within cutoff. This structure will be skipped')
        return None


def get_cart_nghbrs(atoms, list_at_ind=None, cutoff=8.7, skin=0, verbose=False):
    nghbrs_lst = PrimitiveNeighborListWrapper(cutoffs=[cutoff * 0.5] * len(atoms), skin=skin,
                                              self_interaction=False, bothways=True, use_scaled_positions=False)

    nghbrs_lst.update(atoms.get_pbc(), atoms.get_cell(), atoms.get_positions())

    ind_i = []
    mu_i = []
    ind_j = []
    mu_j = []
    offsts = []
    at_nums = atoms.get_atomic_numbers()
    if list_at_ind is not None:
        list_of_atoms = list_at_ind
    else:
        list_of_atoms = np.arange(0, len(atoms))
    check = 0
    for i in list_of_atoms:
        ind, offset, dv = nghbrs_lst.get_neighbors(i)
        if len(ind) < 1:
            check += 1

        sort = np.argsort(ind)
        ind = ind[sort]
        off = dv[sort]

        ind_i.append([i] * len(ind))
        mu_i.append([at_nums[i]] * len(ind))  # Not substructing 1 here
        ind_j.append(ind)
        mu_j.append(np.take(at_nums, ind))  # And here
        offsts.append(off)
    if check == 0:
        return np.hstack(ind_i), np.hstack(ind_j), np.hstack(mu_i), np.hstack(mu_j), np.vstack(offsts).astype(
            np.float64)
    else:
        if verbose:
            print('Found an atom with no neighbors within cutoff. This structure will be skipped')
        return None


def copy_atoms(atoms):
    if atoms.get_calculator() is not None:
        calc = atoms.get_calculator()
        new_atoms = atoms.copy()
        new_atoms.set_calculator(calc)
    else:
        new_atoms = atoms.copy()

    return new_atoms


def chunks(idx_list, chunksize):
    n = max(1, chunksize)
    return [idx_list[i:i + n] for i in range(0, len(idx_list), n)]


def enforce_pbc(atoms, cutoff):
    pos = atoms.get_positions()
    if (atoms.get_pbc() == 0).all():
        max_d = np.max(np.linalg.norm(pos - pos[0], axis=1))
        cell = np.eye(3) * ((max_d + cutoff) * 2)
        atoms.set_cell(cell)
        atoms.center()
    return atoms


def _set_gpu_config(config=None):
    conf_dict = {}
    conf_dict[GPU_INDEX] = 0
    conf_dict[GPU_MEMORY_LIMIT] = 0

    if config is not None:
        assert isinstance(config, dict), 'gpu_config must be a dict'
        if GPU_INDEX in config:
            assert isinstance(config[GPU_INDEX], int), '{} must be an integer'.format(GPU_INDEX)
            conf_dict[GPU_INDEX] = config[GPU_INDEX]

        if GPU_MEMORY_LIMIT in config:
            assert isinstance(config[GPU_MEMORY_LIMIT], int), \
                '{} must be an integer number of MB'.format(GPU_MEMORY_LIMIT)
            if config[GPU_MEMORY_LIMIT] < 0:
                raise ValueError('{} must be not negative'.format(GPU_MEMORY_LIMIT))
            if conf_dict[GPU_INDEX] >= 0 and config[GPU_MEMORY_LIMIT] > 0:
                try:
                    total_gpu_mem = get_gpu_memory(conf_dict[GPU_INDEX])
                except:
                    total_gpu_mem = 0
                assert config[GPU_MEMORY_LIMIT] <= total_gpu_mem, \
                    'Requested GPU memory limit is greater than total GPU memory'
                conf_dict[GPU_MEMORY_LIMIT] = config[GPU_MEMORY_LIMIT]

    return conf_dict


def init_gpu_config(gpu_config):
    gpu_config = _set_gpu_config(gpu_config)
    if gpu_config[GPU_INDEX] < 0:
        sel_gpus = []
        try:
            tf.config.set_visible_devices(sel_gpus, 'GPU')
            tf.config.list_logical_devices('GPU')
        except RuntimeError as e:
            print(e)
    elif gpu_config[GPU_INDEX] >= 0:
        avail_gpus = tf.config.list_physical_devices('GPU')
        if len(avail_gpus) > 0:
            assert gpu_config[GPU_INDEX] < len(avail_gpus), \
                'GPU ind {} is requested, but there are only {} GPUs'.format(gpu_config[GPU_INDEX], len(avail_gpus))
            sel_gpus = avail_gpus[gpu_config[GPU_INDEX]]
            if gpu_config[GPU_MEMORY_LIMIT] != 0:
                try:
                    tf.config.set_logical_device_configuration(
                        sel_gpus,
                        [tf.config.LogicalDeviceConfiguration(memory_limit=gpu_config[GPU_MEMORY_LIMIT])])
                    logical_gpus = tf.config.list_logical_devices('GPU')
                except RuntimeError as e:
                    print(e)
            else:
                try:
                    tf.config.set_visible_devices(sel_gpus, 'GPU')
                    logical_gpus = tf.config.list_logical_devices('GPU')
                except RuntimeError as e:
                    print(e)

    return gpu_config


def get_gpu_memory(gpu_id):
    import subprocess as sp

    command = f"nvidia-smi --id={gpu_id} --query-gpu=memory.total --format=csv"
    output_cmd = sp.check_output(command.split())
    memory = output_cmd.decode("ascii").split("\n")[1]
    memory = int(memory.split()[0])

    return memory
