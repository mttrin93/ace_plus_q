import tensorflow as tf
import numpy as np
import pandas as pd

from ace_plus_q.functions.radial_functions import radial_function
from ace_plus_q.data.datakeys import *
from ace_plus_q.precision import ModelDataTypes, DEFAULT_INT_TYPE, DEFAULT_FLOAT_TYPE

from abc import ABC, abstractmethod
from itertools import combinations_with_replacement, product
from typing import List, Tuple


class Potential(ABC):
    '''
    Base abstract class for all TensorPotential potentials.
        Attribures:
            fit_coefs (tf.Variable[None]): adjustable coefficients of the potential. It must be a 1-D array that
                                           includes all the parameters of the potential that must be optimized for.
            bond_indexing (str):  'delta_all' or 'symmetric_bonds'. Affects the DATA_MU_IJ indexing in TPBatch.
                                  'delta_all': DATA_MU_IJ index elements of the batch belonging to a particular bond
                                   combination (combinations are given by Potential.get_bond_symbol_combinations())
                                   and to every kind of the central atom (given by Potential.get_chemical_symbols()).
                                   This is used if potential treats each bond separately (possibly unique) and if
                                   the type of the central atom is important as well
                                   'symmetric_bonds': DATA_MU_IJ maps every bond to an index. It assumes that bonds
                                   are symmetric, i.e., bonds AB and BA are the same.

            reg_l1 (tf.float64): L1 regularization of the "fit_coefs"

            reg_l2 (tf.float64): L2 regularization of the "fit_coefs"

            aux (List[tf.float64]): collection of the auxiliary regularization parameters. Any additional
                                    regularizations (that should be passed to the loass function and optimized for)
                                    a potential might have must be collected into this list, otherwise
                                    it should be None
            dtypes (ModelDataTypes): defines float and int data types precision that the model will be using
            required_optional_data_entries (list[str]): list of additional datakeys required for potential evaluation

    '''
    def __init__(self,
                 float_type: tf.dtypes = DEFAULT_FLOAT_TYPE,
                 int_type: tf.dtypes = DEFAULT_INT_TYPE,
                 required_optional_data_entries: list[str] = None):
        self.fit_coefs = None # tf.Variable(None, dtype=tf.float64, name='adjustable coefficients')
        self.bond_indexing = 'delta_all'
        self.reg_l1 = 0
        self.reg_l2 = 0
        self.aux = None
        self.tensor_input = None
        self.dtypes = ModelDataTypes(float_type, int_type)
        if required_optional_data_entries is None:
            self.required_optional_data_entries = None
        else:
            self.required_optional_data_entries = required_optional_data_entries

    def init_fit_coefs(self, coefs):
        '''
        Initialize adjustable coefficients. Should only be used to set initial values of the coefficients

        :param coefs: 1-D array of the coefficients to be set as the potential's adjustable coefficients

        :return: None
        '''
        self.fit_coefs = tf.Variable(coefs, dtype=self.dtypes.float, name='adjustable_coefficients')

    def set_coefs(self, coefs):
        '''
        Assign "coefs" to the adjustable coefficients "fit_coefs". This method should be used to change the values
        of the "fit_coefs". Size of the "coefs" must match the size of the "fit_coefs"

        :param coefs: 1-D array of the coefficients that will be assigned
        :return: None
        '''

        self.fit_coefs.assign(tf.Variable(coefs, dtype=self.dtypes.float))

    def get_coefs(self) -> tf.Variable:
        '''
        Returns adjustable coefficients of the potential

        :return: tf.Variable: adjustable coefficients of the potential
        '''

        return self.fit_coefs

    @abstractmethod
    def get_number_of_coefficients(self) -> int:
        '''
        Returns the total number of adjustable coefficients

        :return: (int): total number of coefficients
        '''

        pass

    @abstractmethod
    def get_chemical_symbols(self) -> List[str]:
        '''
        Returns a full list of unique chemical symbols for which the potential is supposed to work.
        Sorting might be important for mapping bonds to a particular type, therefore
        make sure to use consistent element sorting here and in the TPBatch.elements_sorting

        :return: list[str]
        '''

        pass

    @abstractmethod
    def get_bond_symbol_combinations(self) -> List[Tuple[str, str]]:
        '''
        Returns the list of bond combinations that are treated by the potential.

        :return: List[Tuple[str, str]]: list of bond combinations
        '''
        elements  = self.get_chemical_symbols()
        nelements = len(elements)
        comb = [p for p in product(np.arange(nelements), repeat=2)]

        return [(elements[p[0]], elements[p[1]]) for p in comb]

    @abstractmethod
    def compute_regularization(self):
        '''
        Computes L1 and L2 regularization for the adjustable coefficients "fit_coefs"

        :return: None
        '''

        self.reg_l1 = tf.math.reduce_sum(tf.abs(self.fit_coefs))
        self.reg_l2 = tf.math.reduce_sum(self.fit_coefs ** 2)

    @abstractmethod
    def compute_atomic_energy(self, r_ij, input):
        '''
        Computes atomic energy for every atom in the batch. Neighbor information and additional info on atomic
        properties is provided in the "input" dictionary.

            Parameters:
                r_ij (tf.Tensor[NumberOfPairs, 3]): pair vector distances;

                input (dict[str, tf.Tensor]): dictionary with additional structural information necessary
                                              for computing atomic energy;
            Returns:
                e_atom (tf.Tensor[NumberOfAtoms, 1]): atomic energies
        '''

        return None

    @abstractmethod
    def save(self, *args, **kwargs):
        '''
        Saves current Potential state. It is called during fitting process to save necessary information
        for rebuilding a Potential at the current state. Information that is saved is specific for the Potential
        and must be defined for each Potential separately. This is different from
        ace_plus_q.core.save_model() which saves the graph of the entire model and could be run without
        knowledge about specific Potential that is used inside.

        :return: Any
        '''
        pass

    @staticmethod
    def flat_gather_nd(params, indices):
        '''Helper function for flattening tf.gather function along multiple axis'''

        idx_shape = tf.shape(indices, name='shape_idx_shape')
        params_shape = tf.shape(params, name='shape_params_shape')
        idx_dims = idx_shape[-1]
        gather_shape = params_shape[idx_dims:]
        params_flat = tf.reshape(params, tf.concat([[-1], gather_shape], axis=0))
        axis_step = tf.math.cumprod(params_shape[:idx_dims], exclusive=True, reverse=True)
        indices_flat = tf.reduce_sum(indices * axis_step, axis=-1)
        result_flat = tf.gather(params_flat, indices_flat, name='flat_gather_nd')

        return tf.reshape(result_flat, tf.concat([idx_shape[:-1], gather_shape], axis=0))

    def sum_neighbors(self, tensor):
        if self.bond_indexing == 'symmetric_bonds':
            return tf.math.unsorted_segment_sum(tensor, segment_ids=self.tensor_input[DATA_IND_I],
                                                num_segments=self.tensor_input[DATA_NUM_OF_ATOMS])
        else:
            raise NotImplementedError('Only works for symmetric_bonds indexing of the batch dimension')

class BACE(Potential):
    '''Base class for ACE potentials'''

    def __init__(self, potconfig, nelements=1, rcut=5, lmbda=5.25, nradmax=5, lmax=4, nradbase=12,
                 rankmax=2, ndensity=2, core_pre=0., core_lmbda=1., core_cut=100000.0, core_dcut=250.0,
                 fs_parameters=None, compute_smoothness=False, compute_orthogonality=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rcut = rcut
        self.nradmax = nradmax
        self.nradbase = nradbase
        self.lmbda = lmbda
        self.lmax = lmax
        self.rankmax = rankmax
        self.ndensity = ndensity
        self.core_pre = core_pre
        self.core_lmbda = core_lmbda
        self.core_cut = core_cut
        self.core_dcut = core_dcut
        self.nelements = nelements
        self.ranks_sizes = []
        self.tmp_coefs = None
        self.index = None
        self.element_symbols = None
        self.bond_combs = None
        self.deltaSplineBins = 0.001
        self.embedingtype = 'FinnisSinclairShiftedScaled'
        # radbase
        self.radbasetype = "ChebPow"  # "ChebExpCos"
        self.compute_smoothness = compute_smoothness
        self.compute_orthogonality = compute_orthogonality
        if fs_parameters is not None:
            self.fs_parameters = fs_parameters
        else:
            self.fs_parameters = [1., 1., 1., 0.5][:2 * self.ndensity]

        self.BbasisFuncs = None
        # Basis configuration
        if isinstance(potconfig, str):
            raise NotImplementedError('Initialization from anything but BBasisConf'
                                      ' is not implemented for multicomponent ACE')
        else:
            self._init_basis_configuration_from_bbasisconf(potconfig)

        self.ncoef = 0
        if self.tmp_coefs is not None:
            self.ncoef = len(self.tmp_coefs)

        self.init_fit_coefs(self.tmp_coefs)

        if self.compute_smoothness or self.compute_orthogonality:
            self.aux = []
        else:
            self.aux = None

        self.tensor_input = {}
        self.factor4pi = tf.sqrt(4 * tf.constant(np.pi, dtype=self.dtypes.float))

    def compute_regularization(self):
        basis_coefs = self.fit_coefs[self.total_num_crad:]
        self.reg_l1 = tf.reduce_sum(tf.abs(basis_coefs))
        self.reg_l2 = tf.reduce_sum(basis_coefs ** 2)

    def _init_basis_configuration_from_df(self, potconfile):
        if isinstance(potconfile, str):
            df = pd.read_pickle(potconfile)
        else:
            df = potconfile

        self.config = ConfigBasis(df, self.rankmax, self.lmax)

    def get_chemical_symbols(self) -> List[str]:
        return self.element_symbols

    def get_bond_symbol_combinations(self) -> List[Tuple[str, str]]:
        return self.bond_symbol_combs

    def get_number_of_coefficients(self):
        return self.ncoef

    def _init_basis_configuration_from_bbasisconf(self, bbasisfunconf):
        from pyace.basis import BBasisConfiguration, ACEBBasisSet

        assert isinstance(bbasisfunconf, BBasisConfiguration), \
            'provided configuration is not an instance of BBasisConfiguration'

        self.bbasisset = ACEBBasisSet(bbasisfunconf)
        self.nelements = self.bbasisset.nelements
        self.element_symbols = self.bbasisset.elements_name

        self.nradmax = self.bbasisset.nradmax
        self.lmax = self.bbasisset.lmax
        self.nradbase = self.bbasisset.nradbase
        self.bond_specs = self.bbasisset.map_bond_specifications
        self.embed_spec = self.bbasisset.map_embedding_specifications

        # TODO: build not via product, but from bbasisset
        # self.bond_combs = [p for p in product(np.arange(self.nelements), repeat=2)]
        self.bond_combs = list(self.bond_specs.keys())
        self.bond_symbol_combs = [(self.element_symbols[c[0]], self.element_symbols[c[1]]) for c in self.bond_combs]
        self.ncombs = len(self.bond_combs)

        self.ndensity = self.bbasisset.ndensitymax
        self.rcut = self.bbasisset.cutoffmax

        self.tmp_coefs = self.bbasisset.all_coeffs
        self.tmp_coefs[self.tmp_coefs == 0] += 1e-32

        self.total_num_crad = 0
        bond_mus_unique = [k for k in combinations_with_replacement(range(self.nelements), 2)]
        unique_bond_to_slice = {}
        for mus in bond_mus_unique:
            if mus not in self.bond_specs:
                continue
            bond = self.bond_specs[mus]
            n_crad = np.prod(np.shape(bond.radcoefficients))
            unique_bond_to_slice[mus] = [self.total_num_crad, self.total_num_crad + n_crad]
            self.total_num_crad += n_crad
        self.bond_to_slice = {k: unique_bond_to_slice[tuple(sorted(k))] for k in self.bond_specs.keys()}

        self.coefs_part = {}
        self.coefs_r1_part = {c: [] for c in self.bond_combs}
        self.coefs_rk_part = {c: [] for c in self.bond_combs}
        count = 0
        self.ranksmax = []
        for ne in range(self.nelements):
            self.coefs_part[ne] = []
            basis = self.bbasisset.basis_rank1[ne] + self.bbasisset.basis[ne]
            rank = 0
            for f in basis:
                r = f.rank
                for _ in f.coeffs:
                    if r == 1:
                        self.coefs_r1_part[tuple([f.mu0, f.mus[0]])] += [count]
                        self.coefs_rk_part[tuple([f.mu0, f.mus[0]])] += [count + self.total_num_crad]
                        count += 1
                    else:
                        self.coefs_part[ne] += [count]
                        self.coefs_rk_part[tuple([f.mu0, f.mus[0]])] += [count + self.total_num_crad]
                        count += 1
                rank = max([rank, r])
            self.coefs_part[ne] = tf.convert_to_tensor(self.coefs_part[ne], dtype=self.dtypes.int,
                                                       name='index_coefs_{}'.format(ne))
            self.ranksmax += [rank]
        self.coefs_r1_part = {c: tf.convert_to_tensor(self.coefs_r1_part[c], dtype=self.dtypes.int,
                                                      name='index_coefs_r1_{}'.format(c)) for c in self.bond_combs}
        self.rankmax = max(self.ranksmax)

        self.config = self.init_bbasis_configs(nelements=self.nelements)

    def save(self, prefix=None):
        pass

    def init_bbasis_configs(self, nelements):
        ranksmax = []
        #for ne in range(1):
        for ne in range(nelements):
            basis = self.bbasisset.basis_rank1[ne] + self.bbasisset.basis[ne]
            rank = 0
            for f in basis:
                r = f.rank
                rank = max([rank, r])
            ranksmax += [rank]
        # bbasisfuncspecs = [[BBasisFunc(f) for f in self.bbasisset.basis[ne]] for ne in range(1)]
        bbasisfuncspecs = [[BBasisFunc(f) for f in self.bbasisset.basis[ne]] for ne in range(nelements)]
        config = ConfigBasis(bbasisfuncspecs, ranksmax, nelements)
        # config = ConfigBasis(bbasisfuncspecs, ranksmax, 1)

        return config

    def get_updated_config(self, updating_coefs=None, prefix=None):
        if updating_coefs is not None:
            self.set_coefs(updating_coefs)
            self.bbasisset.all_coeffs = self.fit_coefs.numpy()

            return self.bbasisset.to_BBasisConfiguration()
        else:
            if self.fit_coefs is not None and self.bbasisset is not None:
                self.bbasisset.all_coeffs = self.fit_coefs.numpy()

                return self.bbasisset.to_BBasisConfiguration()
            else:
                ValueError("Can't update configuration, no coefficients or nothing to update.")

    @staticmethod
    def complexmul(r1, im1, r2, im2):
        real_part = r1 * r2 - im1 * im2
        imag_part = im2 * r1 + im1 * r2

        return real_part, imag_part

    @staticmethod
    def integrate(func, x, dx, rcut, axis=None):
        if axis is None:
            reduce_axis = [1, 2]
        else:
            reduce_axis = axis
        f = tf.reshape(tf.reduce_sum(tf.abs(func), axis=reduce_axis), [tf.shape(func)[0], -1])
        trapz = tf.reduce_sum(x ** 2 * f, axis=0) * dx
        trapz /= rcut ** 2

        return tf.reshape(trapz, [-1, 1])

    def compute_smoothness_reg(self):
        start = tf.constant(1e-5, dtype=self.dtypes.float)
        stop = tf.constant(self.rcut, dtype=self.dtypes.float)
        d_cont = tf.reshape(tf.linspace(start, stop - 1e-5, 100), [-1, 1])
        delta_d = d_cont[1] - d_cont[0]
        radial_coefs = self.fit_coefs[:self.total_num_crad]
        r_nl = []
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(d_cont)
            with tf.GradientTape(persistent=True) as tape2:
                tape2.watch(d_cont)
                rcuts = []
                for c_i, c in enumerate(self.bond_combs):
                    bond_spec = self.bond_specs[c]
                    nradmaxi = bond_spec.nradmax
                    lmaxi = bond_spec.lmax
                    nradbasei = bond_spec.nradbasemax
                    crad = radial_coefs[slice(*self.bond_to_slice[c])]
                    crad = tf.reshape(crad, [nradmaxi, lmaxi + 1, nradbasei])
                    rcuts += [tf.constant(bond_spec.rcut, dtype=self.dtypes.float)]
                    g_cont = radial_function(d_cont,
                                             nfunc=nradbasei,
                                             ftype=bond_spec.radbasename,
                                             cutoff=rcuts[c_i],
                                             lmbda=tf.constant(bond_spec.radparameters[0],
                                                               dtype=self.dtypes.float))  # [None, nradbase]
                    r_nl += [tf.einsum('jk,nlk->jnl', g_cont, crad)]  # [None, nradmax, lmax+1]
                self.aux += [tf.reshape(tf.reduce_mean(
                    [self.integrate(r_nl_i, d_cont, delta_d, rcuts[i]) for i, r_nl_i in enumerate(r_nl)]),
                    [-1, 1])]
            drnl_dr = [tf.squeeze(tape2.batch_jacobian(r_nl_i, d_cont, experimental_use_pfor=True), axis=-1) for r_nl_i
                       in r_nl]
            self.aux += [tf.reshape(tf.reduce_mean(
                [self.integrate(drnl_dr_i, d_cont, delta_d, rcuts[i]) for i, drnl_dr_i in enumerate(drnl_dr)]),
                [-1, 1])]
        d2rnl_dr2 = [tf.squeeze(tape1.batch_jacobian(drnl_dr_i, d_cont, experimental_use_pfor=True), axis=-1) for
                     drnl_dr_i in drnl_dr]
        self.aux += [tf.reshape(tf.reduce_mean(
            [self.integrate(d2rnl_dr2_i, d_cont, delta_d, rcuts[i]) for i, d2rnl_dr2_i in enumerate(d2rnl_dr2)]),
            [-1, 1])]
        # tf.print(self.aux, "!!!!!!!!!")

    def compute_smoothness_reg_(self):
        start = tf.constant(1e-5, dtype=self.dtypes.float)
        stop = tf.constant(self.rcut, dtype=self.dtypes.float)
        d_cont = tf.reshape(tf.linspace(start, stop - 1e-5, 100), [-1, 1])
        delta_d = d_cont[1] - d_cont[0]
        radial_coefs = self.fit_coefs[:self.total_num_crad]
        w_0 = []
        w_1 = []
        w_2 = []
        for c_i, c in enumerate(self.bond_combs):
            bond_spec = self.bond_specs[c]
            nradmaxi = bond_spec.nradmax
            lmaxi = bond_spec.lmax
            nradbasei = bond_spec.nradbasemax
            crad = radial_coefs[slice(*self.bond_to_slice[c])]
            crad = tf.reshape(crad, [nradmaxi, lmaxi + 1, nradbasei])
            # rcuts += [tf.constant(bond_spec.rcut, dtype=tf.float64)]
            rcut = tf.constant(bond_spec.rcut, dtype=self.dtypes.float)
            with tf.GradientTape(persistent=False) as tape1:
                tape1.watch(d_cont)
                with tf.GradientTape(persistent=False) as tape2:
                    tape2.watch(d_cont)
                    #rcuts = []
                    g_cont = radial_function(d_cont,
                                             nfunc=nradbasei,
                                             ftype=bond_spec.radbasename,
                                             cutoff=rcut,
                                             lmbda=tf.constant(bond_spec.radparameters[0], dtype=self.dtypes.float))
                    r_nl = tf.einsum('jk,nlk->jnl', g_cont, crad)  # [None, nradmax, lmax+1]
                w_0 += [self.integrate(r_nl, d_cont, delta_d, rcut)]
                grad = tape2.batch_jacobian(r_nl, d_cont)
                #drnl_dr = tf.squeeze(tape2.batch_jacobian(r_nl, d_cont, experimental_use_pfor=True), axis=-1)
                drnl_dr = tf.squeeze(grad, axis=-1)
            w_1 += [self.integrate(drnl_dr, d_cont, delta_d, rcut)]
            grad = tape1.batch_jacobian(drnl_dr, d_cont)
            d2rnl_dr2 = tf.squeeze(grad, axis=-1)
            w_2 += [self.integrate(d2rnl_dr2, d_cont, delta_d, rcut)]
        self.aux += [w_0, w_1, w_2]

    def embedding_function(self, rho, mexp, ftype='FinnisSinclairShiftedScaled'):
        if ftype == 'FinnisSinclairShiftedScaled':
            return self.f_exp_shsc(rho, mexp)
        elif ftype == 'FinnisSinclair':
            return self.f_exp_old(rho, mexp)

    def f_exp_old(self, rho, mexp):
        return tf.where(tf.less(tf.abs(rho), tf.constant(1e-10, dtype=self.dtypes.float)), mexp * rho,
                        self.en_func_old(rho, mexp))

    def en_func_old(self, rho, mexp):
        w = tf.constant(10., self.dtypes.float)
        y1 = w * rho ** 2
        g = tf.where(tf.less(tf.constant(30., dtype=self.dtypes.float), y1), 0. * rho, tf.exp(tf.negative(y1)))

        omg = 1. - g
        a = tf.abs(rho)
        y3 = tf.pow(omg * a + 1e-20, mexp)
        # y3 = tf.pow(omg * a, mexp)
        y2 = mexp * g * a
        f = tf.sign(rho) * (y3 + y2)
        return f

    def f_exp_shsc(self, rho, mexp):
        eps = tf.constant(1e-10, dtype=self.dtypes.float)
        cond = tf.abs(tf.ones_like(rho, dtype=self.dtypes.float) * mexp - tf.constant(1., dtype=self.dtypes.float))
        mask = tf.where(tf.less(cond, eps), tf.ones_like(rho, dtype=tf.bool), tf.zeros_like(rho, dtype=tf.bool))

        arho = tf.abs(rho)
        # func = tf.where(mask, rho, tf.sign(rho) * (tf.sqrt(tf.abs(arho + 0.25 * tf.exp(-arho))) - 0.5 * tf.exp(-arho)))
        exprho = tf.exp(-arho)
        nx = 1. / mexp
        xoff = tf.pow(nx, (nx / (1.0 - nx))) * exprho
        yoff = tf.pow(nx, (1 / (1.0 - nx))) * exprho
        func = tf.where(mask, rho, tf.sign(rho) * (tf.pow(xoff + arho, mexp) - yoff))

        return func


class BBasisFunc():
    def __init__(self, bbasisfunc, lmax=None):
        self.rank = bbasisfunc.rank
        self.ns = bbasisfunc.ns
        self.ls = bbasisfunc.ls
        self.mu0 = bbasisfunc.mu0
        self.mus = bbasisfunc.mus
        self.genCG = np.reshape(bbasisfunc.gen_cgs, [-1, 1])
        self.ms = bbasisfunc.ms_combs
        self.coefs = bbasisfunc.coeffs

        self.munlm = self.get_munlm()
        self.msum = np.zeros(len(self.ms)).astype(np.int32)

        if lmax is not None:
            self.adjust_m_index(lmax)

    def get_munlm(self):
        munlm = [np.zeros((len(self.ms), 4)).astype(np.int32) for _ in range(self.rank)]
        for c in range(len(self.ms)):
            for r in range(self.rank):
                munlm[r][c] = np.array([self.mus[r], self.ns[r] - 1, self.ls[r], self.ms[c][r]])

        return munlm

    def adjust_m_index(self, lmax):
        for r in range(self.rank):
            self.munlm[r][:, -1] += lmax


class BBasisFuncSet():
    def __init__(self, list_of_bbasisfunc, rank, ):
        self.rank = rank
        self.munlm = self.get_munlm(list_of_bbasisfunc)
        self.msum = self.get_msum(list_of_bbasisfunc)
        self.genCG = self.get_gen_cg(list_of_bbasisfunc)
        self.coefs = self.get_coefs(list_of_bbasisfunc)

        for r in range(self.rank):
            self.munlm[r][:, -2] = merge_lm(self.munlm[r][:, -2], self.munlm[r][:, -1])
            self.munlm[r] = self.munlm[r][:, :-1]

    def get_munlm(self, list_of_bbasisfunc):
        total_munlm = []
        for i in range(self.rank):
            total_munlm.append(np.vstack([bbasisfunc.munlm[i] for bbasisfunc in list_of_bbasisfunc]))

        return total_munlm

    def get_gen_cg(self, list_of_bbasisfunc):
        return np.vstack([bbasisfunc.genCG for bbasisfunc in list_of_bbasisfunc])

    def get_msum(self, list_of_bbasisfunc):
        return np.hstack([bbasisfunc.msum + i for i, bbasisfunc in enumerate(list_of_bbasisfunc)])

    def get_coefs(self, list_of_bbasisfunc):
        return np.vstack([bbasisfunc.coefs for bbasisfunc in list_of_bbasisfunc])

    def rearrange_msum(self, msum):
        new_msum = []
        j = 0
        for i in range(len(msum)):
            if i == 0:
                new_msum += [j]
            else:
                if msum[i] == msum[i - 1]:
                    new_msum += [j]
                else:
                    j += 1
                    new_msum += [j]

        return np.array(new_msum).astype(np.int32)

    def apply_constrains(self, nmax=2, lmax=0):
        mask = np.ones_like(self.munlm[0][:, 0])
        for i in range(self.rank):
            cond = (self.munlm[i][:, 0] < nmax) & (self.munlm[i][:, 1] <= lmax)
            mask = np.logical_and(mask, cond)

        for i in range(self.rank):
            self.munlm[i] = self.munlm[i][mask]

        self.msum = self.rearrange_msum(self.msum[mask] - np.min(self.msum[mask]))
        self.genCG = self.genCG[mask]
        # self.coefs = self.coefs[mask]


class ConfigBasis():
    def __init__(self, bbasisfunc, ranksmax, nelem):
        self.central_atom = []

        if isinstance(bbasisfunc, pd.DataFrame):
            self.set_basis_from_df(bbasisfunc, ranksmax, nelem)
        elif isinstance(bbasisfunc, list):
            self.set_basis_from_list(bbasisfunc, ranksmax, nelem)

    def set_basis_from_df(self, bbasisfunc, rankmax, nelem):
        for r in range(rankmax):
            rank_data = bbasisfunc.loc[bbasisfunc['rank'] == r + 1]
            rank_data = rank_data['func']
            # basisfuncs = rank_data.apply(BBasisFunc, lmax=lmax)
            basisfuncs = rank_data.tolist()
            basisset = BBasisFuncSet(basisfuncs, rank=r + 1)
            self.central_atom.append(basisset)

    def set_basis_from_list_(self, bbasisfunc, ranksmax, nelem):
        for r in range(1, rankmax):
            total_rank = [[] for _ in range(nelem)]
            for elem in range(nelem):
                funcs_of_rank = [f for f in bbasisfunc[elem] if f.rank == r + 1]
                total_rank[elem] = BBasisFuncSet(funcs_of_rank, rank=r + 1)
            self.central_atom.append(total_rank)

    def set_basis_from_list(self, bbasisfunc, ranksmax, nelem):
        for elem in range(nelem):
            # total_rank = [[] for _ in range(2, ranksmax[elem] + 1)]
            total_rank = []
            for r in range(2, ranksmax[elem] + 1):  # We only collect starting from rank2
                funcs_of_rank = [f for f in bbasisfunc[elem] if f.rank == r]
                total_rank += [BBasisFuncSet(funcs_of_rank, rank=r)]
            self.central_atom.append(total_rank)


class ConfigFromBBasisConf():
    def __init__(self, df, rankmax, lmax):
        self.ranks = []
        for r in range(rankmax):
            rank_data = df.loc[df['rank'] == r + 1]
            rank_data = rank_data['func']
            # basisfuncs = rank_data.apply(BBasisFunc, lmax=lmax)
            basisfuncs = rank_data.tolist()
            basisset = BBasisFuncSet(basisfuncs, rank=r + 1, lmax=lmax)
            self.ranks.append(basisset)


def merge_lm(l, m):
    return l * (l + 1) + m