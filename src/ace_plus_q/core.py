import tensorflow as tf

from pandas import DataFrame
from ace_plus_q.data import TPAtomsDataContainer
from ace_plus_q.graphspecs import *
from ace_plus_q.data.datakeys import *
from ace_plus_q.potentials.potential import Potential

from typing import Dict, List, Union, Tuple, Any, Literal


def prepare_tensor_specs(list_names: List[str], dtypes: any) -> Dict[str, tf.TensorSpec]:
    default_spec_list = {
        DATA_IND_I: tf.TensorSpec([None], dtype=dtypes.int, name=DATA_IND_I),
        DATA_IND_J: tf.TensorSpec([None], dtype=dtypes.int, name=DATA_IND_J),
        DATA_MU_I: tf.TensorSpec([None], dtype=dtypes.int, name=DATA_MU_I),
        DATA_MU_J: tf.TensorSpec([None], dtype=dtypes.int, name=DATA_MU_J),
        DATA_MU_IJ: tf.TensorSpec([None], dtype=dtypes.int, name=DATA_MU_IJ),
        DATA_SLICE_MU_IJ: tf.TensorSpec([None], dtype=dtypes.int, name=DATA_SLICE_MU_IJ),
        DATA_NUM_OF_ATOMS: tf.TensorSpec([], dtype=dtypes.int, name=DATA_NUM_OF_ATOMS),
        DATA_NUM_OF_STRUCTURES: tf.TensorSpec([], dtype=dtypes.int, name=DATA_NUM_OF_STRUCTURES),
        DATA_CELL_ATOM_MAP: tf.TensorSpec([None], dtype=dtypes.int, name=DATA_CELL_ATOM_MAP),
        DATA_CELL_BOND_MAP: tf.TensorSpec([None], dtype=dtypes.int, name=DATA_CELL_BOND_MAP),
        DATA_ATOMIC_STRUCTURE_MAP: tf.TensorSpec([None], dtype=dtypes.int, name=DATA_ATOMIC_STRUCTURE_MAP),

        DATA_POSITIONS: tf.TensorSpec([None, 3], dtype=dtypes.float, name=DATA_POSITIONS),
        DATA_SCALED_POSITIONS: tf.TensorSpec([None, 3], dtype=dtypes.float, name=DATA_SCALED_POSITIONS),
        DATA_UNSCALED_POSITIONS: tf.TensorSpec([None, 3], dtype=dtypes.float, name=DATA_UNSCALED_POSITIONS),
        DATA_CELL: tf.TensorSpec([None, 3, 3], dtype=dtypes.float, name=DATA_CELL),
        DATA_VECTOR_OFFSETS: tf.TensorSpec([None, 3], dtype=dtypes.float, name=DATA_VECTOR_OFFSETS),
        DATA_CELL_OFFSETS: tf.TensorSpec([None, 3], dtype=dtypes.float, name=DATA_CELL_OFFSETS),
        DATA_TOTAL_ENERGY: tf.TensorSpec([None, 1], dtype=dtypes.float, name=DATA_TOTAL_ENERGY),
        DATA_FORCES: tf.TensorSpec([None, 3], dtype=dtypes.float, name=DATA_FORCES),
        DATA_STRESS: tf.TensorSpec([None, 3, 3], dtype=dtypes.float, name=DATA_STRESS),
        DATA_ENERGY_WEIGHTS: tf.TensorSpec([None, 1], dtype=dtypes.float, name=DATA_ENERGY_WEIGHTS),
        DATA_FORCE_WEIGHTS: tf.TensorSpec([None, 1], dtype=dtypes.float, name=DATA_FORCE_WEIGHTS),
        DATA_ATOMIC_NUM_ELEC: tf.TensorSpec([None, 1], dtype=dtypes.float, name=DATA_ATOMIC_NUM_ELEC),
        DATA_TOTAL_NUM_ELEC: tf.TensorSpec([None, 1], dtype=dtypes.float, name=DATA_TOTAL_NUM_ELEC),
        DATA_ATOMIC_CHRG: tf.TensorSpec([None, 1], dtype=dtypes.float, name=DATA_ATOMIC_CHRG),
        DATA_TOTAL_CHRG: tf.TensorSpec([None, 1], dtype=dtypes.float, name=DATA_TOTAL_CHRG),
        DATA_CHI_0: tf.TensorSpec([None, 1], dtype=dtypes.float, name=DATA_CHI_0),
        DATA_J_0: tf.TensorSpec([None, 1], dtype=dtypes.float, name=DATA_J_0),
        DATA_RADII: tf.TensorSpec([None, 1], dtype=dtypes.float, name=DATA_RADII),
        DATA_ELECTRIC_FIELD: tf.TensorSpec([None, 3], dtype=dtypes.float, name=DATA_ELECTRIC_FIELD),

        # for matrix inversion in qeq only
        DATA_IND_S_I: tf.TensorSpec([None], dtype=dtypes.int, name=DATA_IND_S_I),
        DATA_IND_S_J: tf.TensorSpec([None], dtype=dtypes.int, name=DATA_IND_S_J),

        # for Ewald sum only
        DATA_IND_AT_I: tf.TensorSpec([None], dtype=dtypes.int, name=DATA_IND_AT_I),
        DATA_IND_AT_J: tf.TensorSpec([None], dtype=dtypes.int, name=DATA_IND_AT_J),
        DATA_IND_AT_BATCH: tf.TensorSpec([None], dtype=dtypes.int, name=DATA_IND_AT_BATCH),

        DATA_CENTERS: tf.TensorSpec([None, 3], dtype=dtypes.float, name=DATA_CENTERS),
        DATA_K_0: tf.TensorSpec([None, 1], dtype=dtypes.float, name=DATA_K_0),
        DATA_ATOMIC_DIPOLE_MOM: tf.TensorSpec([None, 3], dtype=dtypes.float, name=DATA_ATOMIC_DIPOLE_MOM),
        DATA_TOTAL_DIPOLE_MOM: tf.TensorSpec([None, 3], dtype=dtypes.float, name=DATA_TOTAL_DIPOLE_MOM),
        DATA_MAG_MOM: tf.TensorSpec([None, 3], dtype=dtypes.float, name=DATA_MAG_MOM)
    }

    return_spec = {tensor_name: default_spec_list[tensor_name] for tensor_name in list_names}

    return return_spec


class TensorPotential(tf.Module):

    def __init__(self,
                 potential: Potential,
                 mode: Literal['evaluate', 'train', 'scf_train', 'pairstyle'] = SPEC_EVALUATE_MODE,
                 compute_forces: bool = True,
                 compute_stress: bool = False,
                 eager_evaluate: bool = False,
                 loss_specs: Dict[str, Any] = None,
                 jit_compile: bool = False):
        super(TensorPotential, self).__init__()
        self.potential = potential
        self.dtypes = potential.dtypes
        self.mode = mode.lower()
        self.jit_compile = jit_compile
        self.fit_coefs = None
        self.list_of_tensors = []
        if self.mode == SPEC_EVALUATE_MODE:
            if compute_forces and not compute_stress:
                self.list_of_tensors.extend(SPEC_ENERGY_FORCE_EVAL)
                self.evaluate_branch = self.compute_energy_forces
            elif compute_forces and compute_stress:
                self.list_of_tensors.extend(SPEC_ENERGY_FORCE_STRESS_EVAL)
                self.evaluate_branch = self.compute_energy_forces_stress
            elif compute_stress and not compute_forces:
                raise ValueError(
                    f'{compute_stress=} but {compute_forces=}.'
                    f' There is no graph setup for such evaluation. Set them both either to True or False.')
            else:
                self.list_of_tensors.extend(SPEC_ENERGY_EVAL)
                self.evaluate_branch = self.compute_energy
        elif self.mode == SPEC_TRAIN_MODE:
            if loss_specs is None:
                self.loss_specs = self._set_default_loss_specs()
            else:
                self.loss_specs = self._set_loss_specs(loss_specs)

            if compute_forces and not compute_stress:
                self.list_of_tensors.extend(SPEC_ENERGY_FORCE_TRAIN)
                self.evaluate_branch = self.train_energy_forces
            elif compute_forces and compute_stress:
                self.list_of_tensors.extend(SPEC_ENERGY_FORCE_STRESS_TRAIN)
                self.evaluate_branch = self.train_energy_forces_stress
            elif compute_stress and not compute_forces:
                raise ValueError(
                    f'{compute_stress=} but {compute_forces=}.'
                    f' There is no graph setup for such evaluation. Set them both either to True or False.')
            else:
                self.list_of_tensors.extend(SPEC_ENERGY_TRAIN)
                self.evaluate_branch = self.train_energy
        elif self.mode == SPEC_TRAIN_SCF_MODE:
            if loss_specs is None:
                self.loss_specs = self._set_default_loss_specs()
            else:
                self.loss_specs = self._set_loss_specs(loss_specs)

            self.list_of_tensors.extend(SPEC_ENERGY_FORCE_TRAIN)
            self.evaluate_branch = self.train_energy_forces_scf
            # if compute_forces and not compute_stress:
            #     self.list_of_tensors.extend(SPEC_ENERGY_FORCE_TRAIN)
            #     self.evaluate_branch = self.train_energy_forces_scf
            # elif compute_forces and compute_stress:
            #     self.list_of_tensors.extend(SPEC_ENERGY_FORCE_TRAIN)
            #     self.evaluate_branch = self.train_energy_forces_scf_

        elif self.mode == SPEC_PAIRSTYLE_MODE:
            if not compute_forces or compute_stress:
                raise ValueError(f'{mode=} is only implemented with forces and no stress, but {compute_forces=} and '
                                 f'{compute_stress=}.')
            else:
                self.list_of_tensors.extend(SPEC_ENERGY_FORCE_PAIR)
                self.evaluate_branch = self.compute_pairstyle
        else:
            raise ValueError(f'Mode "{self.mode}" is not in the list of available modes: {SPEC_GRAPH_MODES}')

        if potential.required_optional_data_entries is not None:
            for data_key in potential.required_optional_data_entries:
                if data_key in SPEC_OPTIONAL_DATA_ENTRIES:
                    self.list_of_tensors.append(data_key)
                else:
                    print(f'{data_key} is not in the list of allowed optional entries: {SPEC_OPTIONAL_DATA_ENTRIES}')

        if not eager_evaluate:
            self.decorate_branch()

    def decorate_branch(self):
        self.evaluate_branch = tf.function(func=self.evaluate_branch,
                                           input_signature=[prepare_tensor_specs(self.list_of_tensors,
                                                                                 dtypes=self.dtypes)],
                                           jit_compile=self.jit_compile)

    def _set_default_loss_specs(self) -> Dict[str, Any]:
        spec_dict = {
            SPEC_LOSS_ENERGY_NORM_TYPE: SPEC_LOSS_ENERGY_NORM_PER_ATOM,
            SPEC_LOSS_FORCE_FACTOR: tf.constant(0., self.dtypes.float),
            SPEC_LOSS_ENERGY_FACTOR: tf.constant(1., self.dtypes.float),
        }

        return self._set_loss_specs(spec_dict)
    

    def _set_loss_specs(self, specs: Dict[str, Union[str, float]]) -> Dict[str, Any]:
        assert specs[SPEC_LOSS_ENERGY_NORM_TYPE] in SPEC_LOSS_VALID_ENERGY_NORM_TYPES,\
            f'{specs[SPEC_LOSS_ENERGY_NORM_TYPE]} is not in the list of supported energy loss normalization type:' \
            f' {SPEC_LOSS_VALID_ENERGY_NORM_TYPES}'
        
        spec_dict = {
            SPEC_LOSS_ENERGY_NORM_TYPE: specs[SPEC_LOSS_ENERGY_NORM_TYPE],
            SPEC_LOSS_FORCE_FACTOR: tf.constant(specs.get(SPEC_LOSS_FORCE_FACTOR, 0.), self.dtypes.float),
            SPEC_LOSS_ENERGY_FACTOR: tf.constant(specs.get(SPEC_LOSS_ENERGY_FACTOR, 1.), self.dtypes.float),
            SPEC_LOSS_STRESS_FACTOR: tf.constant(specs.get(SPEC_LOSS_STRESS_FACTOR, 0.), self.dtypes.float),
            SPEC_LOSS_SCF_FACTOR: tf.constant(specs.get(SPEC_LOSS_SCF_FACTOR, 0.), self.dtypes.float),
            SPEC_LOSS_L1_REG_FACTOR: tf.constant(specs.get(SPEC_LOSS_L1_REG_FACTOR, 0.), self.dtypes.float),
            SPEC_LOSS_L2_REG_FACTOR: tf.constant(specs.get(SPEC_LOSS_L2_REG_FACTOR, 0.), self.dtypes.float),
            SPEC_AUX_LOSS_FACTORS: tf.constant(specs.get(SPEC_AUX_LOSS_FACTORS, [0.]), self.dtypes.float)
        }

        return spec_dict

    def compute_regularization_loss(self):
        reg_loss = tf.constant(0, dtype=self.dtypes.float, name='Total_reg_loss')
        reg_components = []
        self.potential.compute_regularization()
        reg_loss += self.potential.reg_l1 * self.loss_specs[SPEC_LOSS_L1_REG_FACTOR]
        reg_loss += self.potential.reg_l2 * self.loss_specs[SPEC_LOSS_L2_REG_FACTOR]
        reg_components += [tf.reshape(self.potential.reg_l1, [1, 1]), tf.reshape(self.potential.reg_l2, [1, 1])]
        if self.potential.aux is not None:
            for i in range(self.loss_specs[SPEC_AUX_LOSS_FACTORS].shape[0]):
                reg_loss += tf.squeeze(self.potential.aux[i] * self.loss_specs[SPEC_AUX_LOSS_FACTORS][i])
                reg_components += [self.potential.aux[i]]
        self.reg_components = tf.stack(reg_components)

        return reg_loss

    def loss_component_energy(self, e: tf.Tensor, input: Dict[str, tf.Tensor]) -> tf.Tensor:
        total_loss = tf.constant(0, dtype=self.dtypes.float, name='Loss_energy_component')
        eweights = input[DATA_ENERGY_WEIGHTS]
        e_true = input[DATA_TOTAL_ENERGY]
        if self.loss_specs[SPEC_LOSS_ENERGY_NORM_TYPE] == 'per-structure':
            total_loss += tf.reduce_sum(eweights * (e - e_true) ** 2)
        elif self.loss_specs[SPEC_LOSS_ENERGY_NORM_TYPE] == 'per-atom':
            counter = tf.ones_like(input[DATA_ATOMIC_STRUCTURE_MAP], dtype=self.dtypes.float)
            natoms = tf.math.unsorted_segment_sum(counter, input[DATA_ATOMIC_STRUCTURE_MAP],
                                                  num_segments=tf.reduce_max(input[DATA_ATOMIC_STRUCTURE_MAP]) + 1)
            natoms = tf.reshape(tf.cast(natoms, self.dtypes.float), [-1, 1])
            total_loss += tf.reduce_sum(eweights * ((e - e_true) / natoms) ** 2)

        return total_loss

    def loss_component_force(self, f: tf.Tensor, input: Dict[str, tf.Tensor]) -> tf.Tensor:
        fweights = input[DATA_FORCE_WEIGHTS]
        f_true = input[DATA_FORCES]

        loss_f = tf.reduce_sum(fweights * (f - f_true) ** 2)

        return loss_f

    def loss_component_stress(self, s: tf.Tensor, input: Dict[str, tf.Tensor]) -> tf.Tensor:
        sweights = input[DATA_STRESS_WEIGHTS]
        s_true = input[DATA_STRESS]

        loss_f = tf.reduce_sum(sweights * (s - s_true) ** 2)

        return loss_f

    # def loss_component_scf(self, de_dn: tf.Tensor, input):
    #     tot_el = tf.zeros([input[DATA_NUM_OF_STRUCTURES], 1], dtype=self.dtypes.float, name='total_num_el')
    #     tot_el = tf.tensor_scatter_nd_add(tot_el, tf.reshape(input[DATA_ATOMIC_STRUCTURE_MAP], [-1, 1]),
    #                                      input[DATA_ATOMIC_NUM_ELEC])
    #     d_tot_num_el = tf.reduce_sum((tot_el - input[DATA_TOTAL_NUM_ELEC]) ** 2)
    #
    #     mu_n = tf.zeros([input[DATA_NUM_OF_STRUCTURES], 1], dtype=self.dtypes.float, name='total_mu_el')
    #     mu_n = tf.tensor_scatter_nd_add(mu_n, tf.reshape(input[DATA_ATOMIC_STRUCTURE_MAP], [-1, 1]), de_dn)
    #
    #     counter = tf.ones_like(input[DATA_ATOMIC_STRUCTURE_MAP], dtype=self.dtypes.float)
    #     natoms = tf.math.unsorted_segment_sum(counter, input[DATA_ATOMIC_STRUCTURE_MAP],
    #                                           num_segments=tf.reduce_max(input[DATA_ATOMIC_STRUCTURE_MAP]) + 1)
    #     natoms = tf.reshape(tf.cast(natoms, self.dtypes.float), [-1, 1])
    #
    #     mu_n = mu_n / natoms
    #     mean_mu = tf.gather(mu_n, input[DATA_ATOMIC_STRUCTURE_MAP])
    #     d_mu = tf.reduce_sum((de_dn - mean_mu) ** 2)
    #
    #     tf.print(d_mu, 'Delta mu')
    #     tf.print(d_tot_num_el, 'Electron conserve')
    #
    #     #for carbon chain
    #     # return d_tot_num_el + d_mu * 100, mu_n
    #     #for AuMgO and water
    #     return d_tot_num_el + d_mu, mu_n

    def loss_component_scf_(self, q: tf.Tensor, input):
        tot_el = tf.zeros([input[DATA_NUM_OF_STRUCTURES], 1], dtype=self.dtypes.float, name='total_num_el')
        tot_el = tf.tensor_scatter_nd_add(tot_el, tf.reshape(input[DATA_ATOMIC_STRUCTURE_MAP], [-1, 1]), q)
        d_tot_num_el = tf.reduce_sum((tot_el - input[DATA_TOTAL_CHRG]) ** 2)

        d_elec = tf.reduce_sum((input[DATA_ATOMIC_CHRG] - q) ** 2)
        tf.print(d_elec, 'diff elec')

        mu_n = tf.zeros([input[DATA_NUM_OF_STRUCTURES], 1], dtype=self.dtypes.float, name='total_mu_el')

        total_dipole = tf.zeros([input[DATA_NUM_OF_STRUCTURES], 3], dtype=self.dtypes.float, name='total_dipole')
        dipole = tf.math.multiply(tf.tile(q, [1, 3]), input[DATA_UNSCALED_POSITIONS])

        total_dipole = tf.tensor_scatter_nd_add(total_dipole, tf.reshape(input[DATA_ATOMIC_STRUCTURE_MAP], [-1, 1]), dipole)
        d_tot_dipole = tf.reduce_sum((total_dipole - input[DATA_TOTAL_DIPOLE_MOM]) ** 2)
        tf.print(d_tot_dipole, 'd_tot_dipole')

#        return d_tot_num_el + d_elec, mu_n
        return d_tot_num_el + d_tot_dipole + d_elec, mu_n


    def loss_component_scf(self, de_dn: tf.Tensor, input):
        #tot_el = tf.ones([input[DATA_NUM_OF_STRUCTURES], 1], dtype=self.dtypes.float, name='total_num_el')
        tot_el = tf.zeros([input[DATA_NUM_OF_STRUCTURES], 1], dtype=self.dtypes.float, name='total_num_el')
        #tot_el = tf.tensor_scatter_nd_add(tot_el, tf.reshape(input[DATA_ATOMIC_STRUCTURE_MAP], [-1, 1]),
        #                                  1 + tf.math.cos(input[DATA_ATOMIC_NUM_ELEC]*TF_PI + 1e-32))
        tot_el = tf.tensor_scatter_nd_add(tot_el, tf.reshape(input[DATA_ATOMIC_STRUCTURE_MAP], [-1, 1]),
                                         input[DATA_ATOMIC_NUM_ELEC])
        d_tot_num_el = tf.reduce_sum((tot_el - input[DATA_TOTAL_NUM_ELEC]) ** 2)
        # tf.print(tot_el, summarize=-1)
        # tf.print('input[DATA_TOTAL_NUM_ELEC]')
        # tf.print('atom_elec')
        # tf.print(input[DATA_ATOMIC_NUM_ELEC], summarize=-1)

        mu_n = tf.zeros([input[DATA_NUM_OF_STRUCTURES], 1], dtype=self.dtypes.float, name='total_mu_el')
        # mu_n = tf.tensor_scatter_nd_add(mu_n, tf.reshape(input[DATA_ATOMIC_STRUCTURE_MAP], [-1, 1]), tf.abs(de_dn))
        mu_n = tf.tensor_scatter_nd_add(mu_n, tf.reshape(input[DATA_ATOMIC_STRUCTURE_MAP], [-1, 1]), de_dn)

        counter = tf.ones_like(input[DATA_ATOMIC_STRUCTURE_MAP], dtype=self.dtypes.float)
        natoms = tf.math.unsorted_segment_sum(counter, input[DATA_ATOMIC_STRUCTURE_MAP],
                                              num_segments=tf.reduce_max(input[DATA_ATOMIC_STRUCTURE_MAP]) + 1)
        natoms = tf.reshape(tf.cast(natoms, self.dtypes.float), [-1, 1])

        mu_n = mu_n / natoms
        # mu_n = mu_n / input[DATA_TOTAL_NUM_ELEC]
        mean_mu = tf.gather(mu_n, input[DATA_ATOMIC_STRUCTURE_MAP])
        # d_mu = tf.reduce_sum(tf.abs(de_dn) ** 2)
        d_mu = tf.reduce_sum((de_dn - mean_mu) ** 2)

        # d_mu = tf.reduce_sum((tf.abs(de_dn)))
        #d_mu = tf.reduce_sum((de_dn) ** 2)
        # tf.print(d_mu, 'Delta mu', de_dn, 'de_dn', mean_mu, 'mean_mu')
        # tf.print(d_mu, 'Delta mu')
        # tf.print(tf.reduce_sum(tf.abs(de_dn)), 'Delta mu')
        # tf.print(d_tot_num_el, 'Electron conserve')
        #tf.print(d_mu, '!!!!!!!!!!!!!!!')
        #tf.print(de_dn, mean_mu, '!!!!!!!!!!!!!!!')


        d_elec = tf.reduce_sum((input[DATA_ATOMIC_CHRG] - input[DATA_ATOMIC_NUM_ELEC]) ** 2)
        # tf.print(d_elec, 'diff elec')

        # total_dipole = tf.zeros([input[DATA_NUM_OF_STRUCTURES], 3], dtype=self.dtypes.float, name='total_dipole')
        # total_dipole = tf.tensor_scatter_nd_add(total_dipole, tf.reshape(input[DATA_ATOMIC_STRUCTURE_MAP], [-1, 1]),
        #                                  tf.math.multiply(tf.tile(input[DATA_ATOMIC_NUM_ELEC], [1, 3]), input[DATA_POSITIONS]))
        # d_tot_dipole = tf.reduce_sum((total_dipole - input[DATA_TOTAL_DIPOLE_MOM]) ** 2)
        # tf.print(d_tot_dipole, 'Dipole conserve')
        # tf.print('pos', input[DATA_POSITIONS])

        # elec = input[DATA_ATOMIC_NUM_ELEC]

        #for carbon chain
        # return d_tot_num_el + d_mu * 100, mu_n
        #for AuMgO and water
        # return d_tot_num_el + d_mu + d_elec, mu_n
        # return d_tot_num_el + d_mu, mu_n
        return d_tot_num_el + d_elec, mu_n

        # this is for dipole fitting:
        # return d_tot_num_el + d_mu + d_tot_dipole, mu_n

        # return d_tot_num_el * 10
        # return d_tot_num_el + d_mu + 10*d_tot_dipole
        # return d_tot_num_el + d_mu + 0.1 * d_tot_dipole

    # def loss_component_scf(self, de_dn: tf.Tensor, input):
    #     tot_el = tf.zeros([input[DATA_NUM_OF_STRUCTURES], 1], dtype=self.dtypes.float, name='total_num_el')
    #     tot_el = tf.tensor_scatter_nd_add(tot_el, tf.reshape(input[DATA_ATOMIC_STRUCTURE_MAP], [-1, 1]),
    #                                      input[DATA_ATOMIC_NUM_ELEC])
    #     d_tot_num_el = tf.reduce_sum((tot_el - input[DATA_TOTAL_NUM_ELEC]) ** 2)
    #
    #     mu_n = tf.zeros([input[DATA_NUM_OF_STRUCTURES], 1], dtype=self.dtypes.float, name='total_mu_el')
    #     mean_mu = tf.zeros_like(input[DATA_ATOMIC_NUM_ELEC], dtype=self.dtypes.float, name='total_mean_mu_el')
    #     d_mu = tf.reduce_sum((de_dn - mean_mu) ** 2)
    #
    #     tf.print(d_mu, 'Delta mu')
    #     tf.print(d_tot_num_el, 'Electron conserve')
    #
    #     #for carbon chain
    #     return d_tot_num_el + d_mu * 100, mu_n
    #     #for AuMgO and water
    #     # return d_tot_num_el + d_mu, mu_n


    # def loss_component_scf_(self, de_dn: tf.Tensor, input):
    #
    #     mu_n = tf.zeros([input[DATA_NUM_OF_STRUCTURES], 3], dtype=self.dtypes.float, name='total_mu_el')
    #     # mu_n = tf.tensor_scatter_nd_add(mu_n, tf.reshape(input[DATA_ATOMIC_STRUCTURE_MAP], [-1, 1]), tf.abs(de_dn))
    #     mu_n = tf.tensor_scatter_nd_add(mu_n, tf.reshape(input[DATA_ATOMIC_STRUCTURE_MAP], [-1, 1]), tf.reshape(de_dn, [-1, 3]))
    #
    #     counter = tf.ones_like(input[DATA_ATOMIC_STRUCTURE_MAP], dtype=self.dtypes.float)
    #     natoms = tf.math.unsorted_segment_sum(counter, input[DATA_ATOMIC_STRUCTURE_MAP],
    #                                           num_segments=tf.reduce_max(input[DATA_ATOMIC_STRUCTURE_MAP]) + 1)
    #     natoms = tf.reshape(tf.cast(natoms, self.dtypes.float), [-1, 1])
    #
    #     mu_n = mu_n / natoms
    #     # mu_n = mu_n / input[DATA_TOTAL_NUM_ELEC]
    #     mean_mu = tf.gather(mu_n, input[DATA_ATOMIC_STRUCTURE_MAP])
    #     # d_mu = tf.reduce_sum(tf.abs(de_dn) ** 2)
    #     # d_mu = tf.reduce_sum((tf.reshape(de_dn, [-1, 3]) - mean_mu) ** 2)
    #     d_mu = tf.reduce_sum((tf.reshape(de_dn, [-1, 3])) ** 2)
    #     # d_mu = tf.reduce_sum((tf.reshape(de_dn, [-1, 3])) ** 2)
    #     tf.print(d_mu, 'Delta mu')
    #     # tf.print(tf.reshape(de_dn, [-1, 3]), 'dmu')
    #
    #     return d_mu

    def compute_energy_loss(self, e: tf.Tensor, input: Dict[str, tf.Tensor]) -> tf.Tensor:
        total_loss = tf.constant(0, dtype=self.dtypes.float, name='Total_energy_loss')

        total_loss += self.loss_specs[SPEC_LOSS_ENERGY_FACTOR] * self.loss_component_energy(e, input)

        total_loss += self.compute_regularization_loss()

        return total_loss

    def compute_energy_forces_loss(self, e: tf.Tensor, f: tf.Tensor, input: Dict[str, tf.Tensor]) -> tf.Tensor:
        total_loss = tf.constant(0, dtype=self.dtypes.float, name='Total_energy_forces_loss')

        total_loss += self.loss_specs[SPEC_LOSS_ENERGY_FACTOR] * self.loss_component_energy(e, input)
        total_loss += self.loss_specs[SPEC_LOSS_FORCE_FACTOR] * self.loss_component_force(f, input)

        # TODO: make it optional somehow
        total_loss += self.compute_regularization_loss()

        return total_loss

    def compute_scf_loss(self, e: tf.Tensor, f: tf.Tensor, q: tf.Tensor,
                         # input: Dict[str, tf.Tensor]) -> tf.Tensor:
                         input: Dict[str, tf.Tensor]) -> tuple[tf.Tensor, tf.Tensor]:
        total_loss = tf.constant(0, dtype=self.dtypes.float, name='Total_energy_forces_loss')

        total_loss += self.loss_specs[SPEC_LOSS_ENERGY_FACTOR] * self.loss_component_energy(e, input)
        total_loss += self.loss_specs[SPEC_LOSS_FORCE_FACTOR] * self.loss_component_force(f, input)
        # scf_loss, mean_mu = self.loss_component_scf(de_dn, input)
        scf_loss, mean_mu = self.loss_component_scf_(q, input)
        total_loss += self.loss_specs[SPEC_LOSS_SCF_FACTOR] * scf_loss
        # total_loss += self.loss_specs[SPEC_LOSS_SCF_FACTOR] * self.loss_component_scf(de_dn, input)

        total_loss += self.compute_regularization_loss()

        return total_loss, mean_mu

    # def compute_scf_loss_(self, e: tf.Tensor, f: tf.Tensor, de_dn: tf.Tensor,
    #                      input: Dict[str, tf.Tensor]) -> tf.Tensor:
    #     total_loss = tf.constant(0, dtype=self.dtypes.float, name='Total_energy_forces_loss')
    #
    #     total_loss += self.loss_specs[SPEC_LOSS_ENERGY_FACTOR] * self.loss_component_energy(e, input)
    #     total_loss += self.loss_specs[SPEC_LOSS_FORCE_FACTOR] * self.loss_component_force(f, input)
    #     total_loss += self.loss_specs[SPEC_LOSS_SCF_FACTOR] * self.loss_component_scf_(de_dn, input)
    #
    #     total_loss += self.compute_regularization_loss()
    #
    #     return total_loss

    def compute_energy_forces_stress_loss(self, e: tf.Tensor, f: tf.Tensor, s: tf.Tensor,
                                          input: Dict[str, tf.Tensor]) -> tf.Tensor:
        total_loss = tf.constant(0, dtype=self.dtypes.float, name='Total_energy_forces_stress_loss')

        total_loss += self.loss_specs[SPEC_LOSS_ENERGY_FACTOR] * self.loss_component_energy(e, input)
        total_loss += self.loss_specs[SPEC_LOSS_FORCE_FACTOR] * self.loss_component_force(f, input)
        total_loss += self.loss_specs[SPEC_LOSS_STRESS_FACTOR] * self.loss_component_stress(s, input)

        total_loss += self.compute_regularization_loss()

        return total_loss

    def compute_pairstyle(self, input: Dict[str, tf.Tensor]) -> tuple[tf.Tensor, tf.Tensor]:
        pos = input[DATA_POSITIONS]
        r_i = tf.gather(pos, input[DATA_IND_I])
        r_j = tf.gather(pos, input[DATA_IND_J]) + input[DATA_VECTOR_OFFSETS]
        r_ij = r_j - r_i
        self.fit_coefs = self.potential.fit_coefs
        with tf.GradientTape() as tape:
            tape.watch(r_ij)
            e_atomic = self.potential.compute_atomic_energy(r_ij, input)

        f = tf.negative(tape.gradient(e_atomic, r_ij))

        return e_atomic, f

    def compute_energy(self, input: Dict[str, tf.Tensor]) -> tf.Tensor:
        pos = input[DATA_POSITIONS]
        self.fit_coefs = self.potential.fit_coefs
        r_i = tf.gather(pos, input[DATA_IND_I])
        r_j = tf.gather(pos, input[DATA_IND_J]) + input[DATA_VECTOR_OFFSETS]
        r_ij = r_j - r_i

        e = tf.zeros([input[DATA_NUM_OF_STRUCTURES], 1], dtype=self.dtypes.float, name='e_total')
        e_atomic = self.potential.compute_atomic_energy(r_ij, input)
        e = tf.tensor_scatter_nd_add(e, tf.reshape(input[DATA_ATOMIC_STRUCTURE_MAP], [-1, 1]), e_atomic)

        return e

    def train_energy(self, input: Dict[str, tf.Tensor]) -> Tuple[Tuple[tf.Tensor, ...], Tuple[tf.Tensor, ...]]:
        pos = input[DATA_POSITIONS]
        self.fit_coefs = self.potential.fit_coefs
        with tf.GradientTape() as tape:
            r_i = tf.gather(pos, input[DATA_IND_I])
            r_j = tf.gather(pos, input[DATA_IND_J]) + input[DATA_VECTOR_OFFSETS]
            r_ij = r_j - r_i

            e = tf.zeros([input[DATA_NUM_OF_STRUCTURES], 1], dtype=self.dtypes.float, name='e_total')
            e_atomic = self.potential.compute_atomic_energy(r_ij, input)
            e = tf.tensor_scatter_nd_add(e, tf.reshape(input[DATA_ATOMIC_STRUCTURE_MAP], [-1, 1]), e_atomic)
            loss = self.compute_energy_loss(e, input)
        grad_loss = tape.gradient(loss, self.fit_coefs)

        return (loss, grad_loss, self.reg_components), (e,)

    def compute_energy_forces(self, input: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
        pos = input[DATA_POSITIONS]
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(pos)
            r_i = tf.gather(pos, input[DATA_IND_I])
            r_j = tf.gather(pos, input[DATA_IND_J]) + input[DATA_VECTOR_OFFSETS]
            r_ij = r_j - r_i

            e = tf.zeros([input[DATA_NUM_OF_STRUCTURES], 1], dtype=self.dtypes.float, name='e_total')
            e_atomic = self.potential.compute_atomic_energy(r_ij, input)
            e = tf.tensor_scatter_nd_add(e, tf.reshape(input[DATA_ATOMIC_STRUCTURE_MAP], [-1, 1]), e_atomic,
                                         name='Total_energy')
        f = tf.negative(tape.gradient(e, pos), name='Total_forces')

        return e, f

    def train_energy_forces(self, input: Dict[str, tf.Tensor]) -> Tuple[Tuple[tf.Tensor, ...], Tuple[tf.Tensor, ...]]:
        pos = input[DATA_POSITIONS]
        self.fit_coefs = self.potential.fit_coefs
        with tf.GradientTape() as tape0:
            with tf.GradientTape() as tape:
                tape.watch(pos)
                r_i = tf.gather(pos, input[DATA_IND_I])
                r_j = tf.gather(pos, input[DATA_IND_J]) + input[DATA_VECTOR_OFFSETS]
                r_ij = r_j - r_i

                e = tf.zeros([input[DATA_NUM_OF_STRUCTURES], 1], dtype=self.dtypes.float, name='e_total')
                e_atomic = self.potential.compute_atomic_energy(r_ij, input)
                e = tf.tensor_scatter_nd_add(e, tf.reshape(input[DATA_ATOMIC_STRUCTURE_MAP], [-1, 1]), e_atomic)
                f = tf.negative(tape.gradient(e, pos))
            loss = self.compute_energy_forces_loss(e, f, input)
        grad_loss = tape0.gradient(loss, self.fit_coefs)

        return (loss, grad_loss, tf.convert_to_tensor(self.reg_components)), (e, f)

    def train_energy_forces_scf_(self, input: Dict[str, tf.Tensor]) -> \
            Tuple[Tuple[tf.Tensor, ...], Tuple[tf.Tensor, ...]]:
        pos = input[DATA_POSITIONS]
        self.fit_coefs = self.potential.fit_coefs
        with tf.GradientTape() as tape0, tf.GradientTape() as tape_dl_dn:
            tape_dl_dn.watch(input[DATA_ATOMIC_NUM_ELEC])
            with tf.GradientTape() as tape, tf.GradientTape() as tape_de_dn:
                tape.watch(pos)
                tape_de_dn.watch(input[DATA_ATOMIC_NUM_ELEC])

                r_i = tf.gather(pos, input[DATA_IND_I])
                r_j = tf.gather(pos, input[DATA_IND_J]) + input[DATA_VECTOR_OFFSETS]
                r_ij = r_j - r_i

                e = tf.zeros([input[DATA_NUM_OF_STRUCTURES], 1], dtype=self.dtypes.float, name='e_total')
                e_atomic = self.potential.compute_atomic_energy(r_ij, input)
                e = tf.tensor_scatter_nd_add(e, tf.reshape(input[DATA_ATOMIC_STRUCTURE_MAP], [-1, 1]), e_atomic)
                f = tf.negative(tape.gradient(e, pos))
            de_dn = tape_de_dn.gradient(e, input[DATA_ATOMIC_NUM_ELEC])
            #loss = self.compute_scf_loss(e, f, de_dn, input)
            loss = self.compute_scf_loss(e, f, tf.negative(de_dn), input)
        grad_loss = tape0.gradient(loss, self.fit_coefs)
        grad_loss_elec = tf.reshape(tape_dl_dn.gradient(loss, input[DATA_ATOMIC_NUM_ELEC]), [-1])
        #tot_grad = tf.concat([grad_loss, grad_loos_elec], axis=0)
        #tf.print(grad_loos_elec, 'grad_loos_elecgrad_loos_elecgrad_loos_elecgrad_loos_elec')

        return (loss, grad_loss, grad_loss_elec, tf.convert_to_tensor(self.reg_components)), (e, f)

    # def train_energy_forces_scf_(self, input: Dict[str, tf.Tensor]) -> \
    #         Tuple[Tuple[tf.Tensor, ...], Tuple[tf.Tensor, ...]]:
    #     pos = input[DATA_POSITIONS]
    #     self.fit_coefs = self.potential.fit_coefs
    #     with tf.GradientTape() as tape0, tf.GradientTape() as tape_dl_dn:
    #         tape_dl_dn.watch(input[DATA_CENTERS])
    #         # with tf.GradientTape() as tape, tf.GradientTape() as tape_de_dn:
    #         with tf.GradientTape() as tape:
    #             tape.watch(pos)
    #             # tape_de_dn.watch(input[DATA_ATOMIC_NUM_ELEC])
    #
    #             r_i = tf.gather(pos, input[DATA_IND_I])
    #             r_j = tf.gather(pos, input[DATA_IND_J]) + input[DATA_VECTOR_OFFSETS]
    #             r_ij = r_j - r_i
    #
    #             e = tf.zeros([input[DATA_NUM_OF_STRUCTURES], 1], dtype=self.dtypes.float, name='e_total')
    #             e_atomic, de_dq = self.potential.compute_atomic_energy(r_ij, input)
    #             #e_atomic = self.potential.compute_atomic_energy(r_ij, input)
    #             e = tf.tensor_scatter_nd_add(e, tf.reshape(input[DATA_ATOMIC_STRUCTURE_MAP], [-1, 1]), e_atomic)
    #             f = tf.negative(tape.gradient(e, pos))
    #         # de_dn = tape_de_dn.gradient(e, input[DATA_ATOMIC_NUM_ELEC])
    #         loss = self.compute_scf_loss_(e, f, de_dq, input)
    #         #loss = self.compute_scf_loss(e, f, tf.negative(de_dn), input)
    #         # loss = self.compute_scf_loss(e, f, de_dq, input)
    #     grad_loss = tape0.gradient(loss, self.fit_coefs)
    #     grad_loss_elec = tf.reshape(tape_dl_dn.gradient(loss, input[DATA_CENTERS]), [-1])
    #     #tot_grad = tf.concat([grad_loss, grad_loos_elec], axis=0)
    #     #tf.print(grad_loos_elec, 'grad_loos_elecgrad_loos_elecgrad_loos_elecgrad_loos_elec')
    #
    #     # return (loss, grad_loss, grad_loss_elec, tf.convert_to_tensor(self.reg_components)), (e, f, mu, de_dq)
    #     return (loss, grad_loss, grad_loss_elec, tf.convert_to_tensor(self.reg_components)), (e, f)

    def train_energy_forces_scf(self, input: Dict[str, tf.Tensor]) -> \
            Tuple[Tuple[tf.Tensor, ...], Tuple[tf.Tensor, ...]]:
        pos = input[DATA_POSITIONS]
        self.fit_coefs = self.potential.fit_coefs
        with tf.GradientTape() as tape0:
            with tf.GradientTape() as tape:
                tape.watch(pos)

                r_i = tf.gather(pos, input[DATA_IND_I])
                r_j = tf.gather(pos, input[DATA_IND_J]) + input[DATA_VECTOR_OFFSETS]
                r_ij = r_j - r_i

                e = tf.zeros([input[DATA_NUM_OF_STRUCTURES], 1], dtype=self.dtypes.float, name='e_total')
                e_atomic, q = self.potential.compute_atomic_energy(r_ij, input)
                #e_atomic = self.potential.compute_atomic_energy(r_ij, input)
                e = tf.tensor_scatter_nd_add(e, tf.reshape(input[DATA_ATOMIC_STRUCTURE_MAP], [-1, 1]), e_atomic)
                # e_qeq = tf.tensor_scatter_nd_add(e_qeq, tf.reshape(input[DATA_ATOMIC_STRUCTURE_MAP], [-1, 1]), e_atomic_qeq)
                f = tf.negative(tape.gradient(e, pos))

            # de_dn = tape_de_dn.gradient(e, input[DATA_ATOMIC_NUM_ELEC])
            loss, mu = self.compute_scf_loss(e, f, q, input)
            #loss = self.compute_scf_loss(e, f, tf.negative(de_dn), input)
            # loss = self.compute_scf_loss(e, f, de_dq, input)
        grad_loss = tape0.gradient(loss, self.fit_coefs)
        #grad_loss_elec = tf.reshape(tape_dl_dn.gradient(loss, c), [-1])
        grad_loss_elec = tf.reshape(tf.zeros_like(input[DATA_ATOMIC_CHRG]), [-1])

        #return (loss, grad_loss, grad_loss_elec, tf.convert_to_tensor(self.reg_components)), (e, f, chi)
        #return (loss, grad_loss, grad_loss_elec, tf.convert_to_tensor(self.reg_components)), (e, f, q)
        return (loss, grad_loss, grad_loss_elec, tf.convert_to_tensor(self.reg_components)), (e, f)

    def compute_energy_forces_stress(self, input: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, ...]:
        cell = input[DATA_CELL]
        cells_pos = tf.gather(cell, input[DATA_CELL_ATOM_MAP])
        pos = tf.reshape(input[DATA_SCALED_POSITIONS], [-1, 1, 3])
        pos = tf.reshape(tf.matmul(pos, cells_pos), [-1, 3])
        self.fit_coefs = self.potential.fit_coefs
        with tf.GradientTape() as tapeF, tf.GradientTape() as tapeS:
            tapeF.watch(pos)
            tapeS.watch(cell)
            cells = tf.gather(cell, input[DATA_CELL_BOND_MAP])
            r_i = tf.gather(pos, input[DATA_IND_I])
            j_ofst = tf.reshape(input[DATA_CELL_OFFSETS], [-1, 1, 3])
            j_ofst = tf.reshape(tf.matmul(j_ofst, cells), [-1, 3])
            r_j = tf.gather(pos, input[DATA_IND_J]) + j_ofst
            r_ij = r_j - r_i

            e = tf.zeros([input[DATA_NUM_OF_STRUCTURES], 1], dtype=self.dtypes.float, name='e_total')
            e_atomic = self.potential.compute_atomic_energy(r_ij, input)
            e = tf.tensor_scatter_nd_add(e, tf.reshape(input[DATA_ATOMIC_STRUCTURE_MAP], [-1, 1]), e_atomic)
        f = tf.negative(tapeF.gradient(e, pos))
        s = tapeS.gradient(e, cell)
        v = tf.reshape(tf.linalg.det(cell), [-1, 1, 1])
        s = tf.linalg.matmul(s, cell, transpose_a=True) / v

        return e, f, s

    def train_energy_forces_stress(self, input: Dict[str, tf.Tensor]) -> Tuple[
        Tuple[tf.Tensor, ...], Tuple[tf.Tensor, ...]]:

        cell = input[DATA_CELL]
        cells_pos = tf.gather(cell, input[DATA_CELL_ATOM_MAP])
        pos = tf.reshape(input[DATA_SCALED_POSITIONS], [-1, 1, 3])
        pos = tf.reshape(tf.matmul(pos, cells_pos), [-1, 3])
        self.fit_coefs = self.potential.fit_coefs
        with tf.GradientTape() as tape0:
            with tf.GradientTape() as tapeF, tf.GradientTape() as tapeS:
                tapeF.watch(pos)
                tapeS.watch(cell)
                cells = tf.gather(cell, input[DATA_CELL_BOND_MAP])
                r_i = tf.gather(pos, input[DATA_IND_I])
                j_ofst = tf.reshape(input[DATA_CELL_OFFSETS], [-1, 1, 3])
                j_ofst = tf.reshape(tf.matmul(j_ofst, cells), [-1, 3])
                r_j = tf.gather(pos, input[DATA_IND_J]) + j_ofst
                r_ij = r_j - r_i

                e = tf.zeros([input[DATA_NUM_OF_STRUCTURES], 1], dtype=self.dtypes.float, name='e_total')
                e_atomic = self.potential.compute_atomic_energy(r_ij, input)
                e = tf.tensor_scatter_nd_add(e, tf.reshape(input[DATA_ATOMIC_STRUCTURE_MAP], [-1, 1]), e_atomic)
                f = tf.negative(tapeF.gradient(e, pos))
                s = tapeS.gradient(e, cell)
                v = tf.reshape(tf.linalg.det(cell), [-1, 1, 1])
                s = tf.linalg.matmul(s, cell, transpose_a=True) / v
            loss = self.compute_energy_forces_stress_loss(e, f, s, input)
        grad_loss = tape0.gradient(loss, self.fit_coefs)

        return (loss, grad_loss, self.reg_components), (e, f, s)

    def evaluate(self, data: Dict[str, Any]) -> Union[
        tf.Tensor, Tuple[tf.Tensor, ...], Tuple[Tuple[tf.Tensor, ...], ...]]:
        input_dict = self.convert_to_tensor({k: data[k] for k in self.list_of_tensors}, dtypes=self.dtypes)

        return self.evaluate_branch(input_dict)

    def train(self, data: Union[list[TPAtomsDataContainer], DataFrame], **kwargs):
        from src.ace_plus_q.fit import FitTensorPotential

        assert self.mode in [SPEC_TRAIN_MODE, SPEC_TRAIN_SCF_MODE], f'Fitting can only be done with a graph' \
                                                                    f' configured in mode={SPEC_TRAIN_MODE} or ' \
                                                                    f'mode={SPEC_TRAIN_SCF_MODE}. Graph is' \
                                                                    f' in mode={self.mode} instead.'


        self.tpf = FitTensorPotential(self)
        self.tpf.fit(df = data, **kwargs)

    def external_fit(self, coefs: Any, data: Dict[str, Any]) -> Union[
        tf.Tensor, Tuple[tf.Tensor, ...], Tuple[Tuple[tf.Tensor, ...], ...]]:

        self.potential.set_coefs(coefs)

        return self.evaluate(data)

    def native_fit(self, data: Dict[str, Any]) -> Any:
        # if eager:
        #     loss, grad_loss, e, f, self.reg_components = self._eager_evaluate_loss(*input2evaloss(data))
        # else:
        #     loss, grad_loss, e, f, self.reg_components = self._evaluate_loss(*input2evaloss(data))
        #
        # return loss, grad_loss, e, f
        return self.evaluate(data)

    # def external_vector_fit(self, coefs, data):
    #     self.potential.set_coefs(coefs)
    #     loss, grad_loss, e, f, self.reg_components = self._evaluate_vector_loss(*input2evaloss(data))
    #
    #     return loss, grad_loss, e, f

    def save_model(self, path: str):
        tf.saved_model.save(self, path)
        #TODO: save additional potential information

    @staticmethod
    def load_model(path: str) -> tf.saved_model:
        imported = tf.saved_model.load(path)

        return imported


    @staticmethod
    def convert_to_tensor(input, dtypes=None):
        if dtypes is None:
            from src.ace_plus_q.precision import ModelDataTypes, DEFAULT_INT_TYPE, DEFAULT_FLOAT_TYPE
            dtypes = ModelDataTypes(float=DEFAULT_FLOAT_TYPE, int=DEFAULT_INT_TYPE)

        converted = {}
        for k, v in input.items():
            if k in SPEC_LIST_OF_INT_TENSORS:
                converted[k] = tf.convert_to_tensor(v, dtype=dtypes.int, name=k)
            elif k in SPEC_LIST_OF_FLOAT_TENSORS:
                converted[k] = tf.convert_to_tensor(v, dtype=dtypes.float, name=k)
            else:
                raise ValueError(f'{k} is not in any of the datatype lists')

        return converted

