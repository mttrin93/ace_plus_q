import numpy as np
import tensorflow as tf
import pathlib

from tensorpotential.functions import radial_functions, spherical_harmonics
from tensorpotential.data.datakeys import *
from tensorpotential.potentials.potential import BACE
from itertools import product, combinations_with_replacement
from tensorpotential.data.symbols import symbol_to_atomic_number
# tf.debugging.set_log_device_placement(True)

# TF_PI = tf.constant(np.pi, dtype=tf.float64)
# COUL_CONST = tf.constant(14.3996, dtype=tf.float64)

class QACE(BACE):
    def __init__(self, n_e_atomic_prop, max_at, nkopints, g_ewald, name_out, invert_matrix=False, *args, **kwargs):
        self.n_e_atomic_prop = n_e_atomic_prop
        self.invert_matrix = invert_matrix
        self.max_at = max_at
        self.nkpoints = nkopints
        self.g_ewald = g_ewald
        self.name_out = name_out
        super().__init__(*args, **kwargs)

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

    def compute_core_repulsion(self, d_ij, core_lmbda, core_pre, core_cut, core_dcut):
        phi_core = core_pre * tf.math.exp(-core_lmbda * d_ij ** 2) / d_ij

        return phi_core * radial_functions.cutoff_func_poly(d_ij, core_cut, core_dcut)

    def f_cut_core_rep(self, rho_core, rho_core_cut, drho_core_cut):
        condition1 = tf.less_equal(rho_core_cut, rho_core)
        condition2 = tf.less(rho_core, rho_core_cut - drho_core_cut)

        res = tf.where(condition1, tf.zeros_like(rho_core, dtype=self.dtypes.float),
                       tf.where(condition2, tf.ones_like(rho_core, dtype=self.dtypes.float),
                                (radial_functions.cutoff_func_poly(rho_core, rho_core_cut, drho_core_cut))))

        return res

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

        self.bond_combs = list(self.bond_specs.keys())
        self.bond_symbol_combs = [(self.element_symbols[c[0]], self.element_symbols[c[1]]) for c in self.bond_combs]

        self.ndensity = self.bbasisset.ndensitymax
        self.rcut = self.bbasisset.cutoffmax

        epsilon = 1e-5
        self.tmp_coefs = self.bbasisset.all_coeffs
        self.tmp_coefs[self.tmp_coefs == 0] += epsilon
        self.total_num_crad = len(self.bbasisset.crad_coeffs)

        self.ncombs = len(self.bond_combs)

        self.u_bond_symbol_combs = [p for p in combinations_with_replacement(self.element_symbols, r=2)]
        self.bond_symbol_combs = [p for p in product(self.element_symbols, repeat=2)]
        self.bond_map = []
        for all_b in self.bond_symbol_combs:
            for j, u_b in enumerate(self.u_bond_symbol_combs):
                if (np.sort(all_b) == np.sort(u_b)).all():
                    self.bond_map.append(j)

        self.ranksmax = []
        for ne in range(self.nelements):
            basis = self.bbasisset.basis_rank1[ne] + self.bbasisset.basis[ne]
            rank = 0
            for f in basis:
                r = f.rank
                rank = max([rank, r])
            self.ranksmax += [rank]
        self.rankmax = np.max(self.ranksmax)

        self.crad_slices = {-1: 0}
        crads = []
        self.total_num_crad = 0

        self.b1_slices = {-1: 0}
        b1 = []
        self.total_num_b1 = 0

        combs_crad = []

        for c_i, c in enumerate(self.bond_combs):
            bond_spec = self.bond_specs[c]
            nradmaxi = bond_spec.nradmax
            lmaxi = bond_spec.lmax
            nradbasei = bond_spec.nradbasemax

            bond_ind = self.bond_map[c_i]
            if bond_ind not in combs_crad:
                crad_shape = [nradmaxi, lmaxi + 1, nradbasei]
                try:
                    crad = self.bbasisset.auxdata.double_arr_data[f'crad_{bond_ind}']
                    crad = np.array(crad)
                except:
                    crad = np.zeros(crad_shape) + epsilon
                    for n in range(nradmaxi):
                        crad[n, :, n] = 1.0
                self.total_num_crad += np.prod(crad_shape)
                self.crad_slices[bond_ind] = self.total_num_crad
                crads.extend(crad.flatten())
            combs_crad.append(bond_ind)

            try:
                basis_coefs = self.bbasisset.auxdata.double_arr_data[f'basis_r1_{c_i}']
                basis_coefs = np.array(basis_coefs)
            except:
                basis_coefs = np.zeros((nradbasei, self.ndensity)) + epsilon
            size = np.prod([nradbasei, self.ndensity])
            self.total_num_b1 += size
            self.b1_slices[c_i] = self.total_num_b1
            b1.extend(basis_coefs.flatten())

        self.total_num_chg_coefs = 2
        try:
            chg_int = self.bbasisset.auxdata.double_arr_data[f'chg_int']
            chg_int = np.array(chg_int)
        except:
            chg_int = np.ones((2,)) + epsilon

        self.bn_slices = {-1: 0}
        bn = []
        self.total_num_bn = 0
        for c_i in range(self.nelements):
            nbasisfuncs = len(self.bbasisset.basis[c_i])
            try:
                basis_coefs = self.bbasisset.auxdata.double_arr_data[f'basis_rn_{c_i}']
                basis_coefs = np.array(basis_coefs)
            except:
                basis_coefs = np.zeros((nbasisfuncs, self.ndensity)) + epsilon
            size = np.prod([nbasisfuncs, self.ndensity])
            self.total_num_bn += size
            self.bn_slices[c_i] = self.total_num_bn
            bn.extend(basis_coefs.flatten())

        self.tmp_coefs = np.concatenate([crads, chg_int, b1, bn])
        self.config = self.init_bbasis_configs(nelements=self.nelements)

    def save(self, prefix=None):
        radial_coefs = self.fit_coefs[:self.total_num_crad]
        chg_coefs = self.fit_coefs[self.total_num_crad:self.total_num_crad + self.total_num_chg_coefs]
        basis_coefs = self.fit_coefs[self.total_num_crad + self.total_num_chg_coefs:]
        b1_coefs = basis_coefs[:self.total_num_b1]
        bn_coefs = basis_coefs[self.total_num_b1:]

        cfs = {}

        combs_crad = []

        for c_i, c in enumerate(self.bond_combs):
            bond_ind = self.bond_map[c_i]

            if bond_ind not in combs_crad:
                crad = radial_coefs[self.crad_slices[bond_ind - 1]:self.crad_slices[bond_ind]]
                cfs[f'crad_{bond_ind}'] = crad
            combs_crad.append(bond_ind)

            c_b1 = b1_coefs[self.b1_slices[c_i - 1]:self.b1_slices[c_i]]
            cfs[f'basis_r1_{c_i}'] = c_b1
        for c_i in range(self.nelements):
            c_bn = bn_coefs[self.bn_slices[c_i - 1]:self.bn_slices[c_i]]

            cfs[f'basis_rn_{c_i}'] = c_bn
        cfs[f'chg_int'] = chg_coefs
        self.bbasisset.auxdata.double_arr_data = cfs

        conf = self.bbasisset.to_BBasisConfiguration()
        path = './saves/{}_{}/'.format(self.name_out, prefix)
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        conf.save(path + 'q_ace.yaml')

    def compute_bbasis_funcs(self, a_munlm_r, a_munlm_i, cntrl_at, rank):
        a_r = tf.transpose(a_munlm_r, [1, 2, 3, 0])
        a_i = tf.transpose(a_munlm_i, [1, 2, 3, 0])

        a_1_r = self.flat_gather_nd(a_r, self.config.central_atom[cntrl_at][rank - 2].munlm[0])
        a_2_r = self.flat_gather_nd(a_r, self.config.central_atom[cntrl_at][rank - 2].munlm[1])
        a_1_i = self.flat_gather_nd(a_i, self.config.central_atom[cntrl_at][rank - 2].munlm[0])
        a_2_i = self.flat_gather_nd(a_i, self.config.central_atom[cntrl_at][rank - 2].munlm[1])

        prod_r, prod_i = self.complexmul(a_1_r, a_1_i, a_2_r, a_2_i)

        if rank > 2:
            for k in range(2, rank):
                a_k_r = self.flat_gather_nd(a_r, self.config.central_atom[cntrl_at][rank - 2].munlm[k])
                a_k_i = self.flat_gather_nd(a_i, self.config.central_atom[cntrl_at][rank - 2].munlm[k])

                prod_r, prod_i = self.complexmul(prod_r, prod_i, a_k_r, a_k_i)

        b_base = prod_r * tf.convert_to_tensor(self.config.central_atom[cntrl_at][rank - 2].genCG, dtype=self.dtypes.float)
        # TODO: scatter update
        b_base = tf.transpose(tf.math.unsorted_segment_sum(b_base, self.config.central_atom[cntrl_at][rank - 2].msum,
                                                           num_segments=tf.reduce_max(
                                                               self.config.central_atom[cntrl_at][rank - 2].msum) + 1),
                              [1, 0])

        return b_base


    def eval_atomic_energy(self, r_ij, pos):
        d_ij = tf.reshape(tf.linalg.norm(r_ij, axis=1), [-1, 1])
        rhat = r_ij / d_ij  # [None, 3]/[None, 1] -> [None, 3]

        sh = spherical_harmonics.SphericalHarmonics(self.lmax, prec=self.dtypes)
        ylm_r, ylm_i = sh.compute_ylm(rhat)
        ynlm_r = tf.expand_dims(ylm_r, 1) * self.factor4pi  # [None, 1, (lmax+1) * (lmax+1)]
        ynlm_i = tf.expand_dims(ylm_i, 1) * self.factor4pi  # [None, 1, (lmax+1) * (lmax+1)]

        radial_coefs = self.fit_coefs[:self.total_num_crad]
        chg_coefs = self.fit_coefs[self.total_num_crad:self.total_num_crad + self.total_num_chg_coefs]
        basis_coefs = self.fit_coefs[self.total_num_crad + self.total_num_chg_coefs:]
        b1_coefs = basis_coefs[:self.total_num_b1]
        bn_coefs = basis_coefs[self.total_num_b1:]

        a_munlm_r = tf.zeros([self.tensor_input[DATA_NUM_OF_ATOMS], self.nelements, self.nradmax, (self.lmax + 1) ** 2],
                               dtype=self.dtypes.float)
        a_munlm_i = tf.zeros([self.tensor_input[DATA_NUM_OF_ATOMS], self.nelements, self.nradmax, (self.lmax + 1) ** 2],
                               dtype=self.dtypes.float)

        total_phi_core = tf.zeros([self.tensor_input[DATA_NUM_OF_ATOMS], 1], dtype=self.dtypes.float, name='total_phi_core')
        inner_cut = tf.zeros([self.tensor_input[DATA_NUM_OF_ATOMS], 1], dtype=self.dtypes.float, name='core_cut')
        rho_r1 =0
        dsor = 0
        for c_i, c in enumerate(self.bond_combs):
            bond_ind = self.bond_map[c_i]
            bond_spec = self.bond_specs[c]
            nradmaxi = bond_spec.nradmax
            lmaxi = bond_spec.lmax
            nradbasei = bond_spec.nradbasemax
            ndensity = self.embed_spec[c[0]].ndensity

            a_1 = tf.zeros([self.tensor_input[DATA_NUM_OF_ATOMS], nradbasei],
                           dtype=self.dtypes.float, name=f'a_1_{c_i}')
            phi_core = tf.zeros([self.tensor_input[DATA_NUM_OF_ATOMS], 1], dtype=self.dtypes.float,
                                name=f'phi_core_{c_i}')

            crad = tf.reshape(radial_coefs[self.crad_slices[bond_ind - 1]:self.crad_slices[bond_ind]],
                              [nradmaxi, lmaxi + 1, nradbasei])

            part_gather = self.tensor_input[DATA_MU_IJ][
                          self.tensor_input[DATA_SLICE_MU_IJ][c_i]:self.tensor_input[DATA_SLICE_MU_IJ][c_i + 1]]
            part_d_ij = tf.gather(d_ij, part_gather)
            part_ylm_r = tf.gather(ynlm_r, part_gather)
            part_ylm_i = tf.gather(ynlm_i, part_gather)

            # if self.rankmax > 1:
            #     imat = tf.eye(nradmaxi, batch_shape=[lmaxi + 1], dtype=tf.float64)
            #     wwt = tf.matmul(tf.transpose(crad, [1, 0, 2]), tf.transpose(crad, [1, 2, 0]))
            #     dsor += tf.reduce_mean((wwt - imat) ** 2)
            #
            #     imat = tf.eye(nradbasei, batch_shape=[lmaxi + 1], dtype=tf.float64)
            #     wtw = tf.matmul(tf.transpose(crad, [1, 2, 0]), tf.transpose(crad, [1, 0, 2]))
            #     dsor += tf.reduce_mean((wtw - imat) ** 2)

            comb_gjk = radial_functions.radial_function(part_d_ij,
                                            nfunc=nradbasei,
                                            ftype=bond_spec.radbasename,
                                            cutoff=tf.constant(bond_spec.rcut, dtype=self.dtypes.float),
                                            lmbda=tf.constant(bond_spec.radparameters[0],
                                                              dtype=self.dtypes.float))  # [None, nradbase]
            comb_gjk = comb_gjk * (1 - radial_functions.cutoff_func_poly(part_d_ij,
                                                                         tf.constant(bond_spec.rcut_in,
                                                                                     dtype=self.dtypes.float),
                                                                         tf.constant(bond_spec.dcut_in,
                                                                                     dtype=self.dtypes.float)))

            atomic_ind = tf.reshape(tf.gather(self.tensor_input[DATA_IND_I], part_gather), [-1, 1])
            a_1 = tf.tensor_scatter_nd_add(a_1, atomic_ind, comb_gjk)

            # Compute core repulsion
            phi_ij = self.compute_core_repulsion(part_d_ij,
                                                 core_lmbda=tf.constant(bond_spec.lambdahc, dtype=self.dtypes.float),
                                                 core_pre=tf.constant(bond_spec.prehc, dtype=self.dtypes.float),
                                                 core_cut=tf.constant(bond_spec.rcut_in, dtype=self.dtypes.float),
                                                 core_dcut=tf.constant(bond_spec.dcut_in, dtype=self.dtypes.float))
            phi_core = tf.tensor_scatter_nd_add(phi_core, atomic_ind, phi_ij)
            total_phi_core += phi_core
            f_cut_core_ij = self.f_cut_core_rep(phi_core,
                                                rho_core_cut=tf.constant(self.embed_spec[c[0]].rho_core_cutoff,
                                                                         dtype=self.dtypes.float),
                                                drho_core_cut=tf.constant(self.embed_spec[c[0]].drho_core_cutoff,
                                                                          dtype=self.dtypes.float))
            inner_cut += f_cut_core_ij / self.ncombs

            c_b1 = tf.reshape(b1_coefs[self.b1_slices[c_i - 1]:self.b1_slices[c_i]], [nradbasei, ndensity])
            e = tf.einsum('jn,nd->jd', a_1, c_b1)
            rho_r1 += tf.pad(e, [[0, 0], [0, self.ndensity - ndensity]])

            if self.rankmax > 1:
                rj_nl = tf.einsum('jk,nlk->jnl', comb_gjk, crad)  # [None, nradmax, lmax+1]
                rj_nl = tf.pad(rj_nl, [[0, 0], [0, self.nradmax - nradmaxi], [0, self.lmax - lmaxi]])
                rj_nlm = tf.gather(rj_nl, sh.l_tile, axis=2)
                phij_nlm_r = rj_nlm * part_ylm_r  # [None, nradmax, (lmax+1)*(lmax+1)]
                phij_nlm_i = rj_nlm * part_ylm_i  # [None, nradmax, (lmax+1)*(lmax+1)]

                update_index = tf.zeros_like(atomic_ind) + tf.constant(c[1], dtype=self.dtypes.int)
                update_index = tf.concat([atomic_ind, update_index], axis=1)
                a_munlm_r = tf.tensor_scatter_nd_add(a_munlm_r, update_index, phij_nlm_r)
                a_munlm_i = tf.tensor_scatter_nd_add(a_munlm_i, update_index, phij_nlm_i)

        #
        # if self.compute_orthogonality:
        #     self.aux += [tf.reshape(dsor, [-1, 1])]

        at_nrgs = tf.zeros([self.tensor_input[DATA_NUM_OF_ATOMS], 1], dtype=self.dtypes.float)
        at_nrgs_chiJ = tf.zeros([self.tensor_input[DATA_NUM_OF_ATOMS], self.n_e_atomic_prop], dtype=self.dtypes.float)
        if self.rankmax > 1:
            collect_basis = [[] for _ in range(self.nelements)]
            for i, cent_at in enumerate(collect_basis):
                ind_c_at_i = self.tensor_input[DATA_MU_IJ][self.tensor_input[DATA_SLICE_MU_IJ][self.ncombs + i]:
                                                           self.tensor_input[DATA_SLICE_MU_IJ][self.ncombs + i + 1]]
                a_munlm_r_ci = tf.gather(a_munlm_r, ind_c_at_i)
                a_munlm_i_ci = tf.gather(a_munlm_i, ind_c_at_i)

                for k in range(2, self.ranksmax[i] + 1):
                    cent_at += [self.compute_bbasis_funcs(a_munlm_r_ci, a_munlm_i_ci, i, k)]
            basis = [tf.concat(cent_at, axis=1) for cent_at in collect_basis]

            for c_i in range(self.nelements):
                ind_c_at_i = self.tensor_input[DATA_MU_IJ][self.tensor_input[DATA_SLICE_MU_IJ][self.ncombs + c_i]:
                                                           self.tensor_input[DATA_SLICE_MU_IJ][self.ncombs + c_i + 1]]
                ndensity = self.embed_spec[c_i].ndensity
                fs_parameters = self.embed_spec[c_i].FS_parameters

                coefs = tf.reshape(bn_coefs[self.bn_slices[c_i - 1]:self.bn_slices[c_i]],
                                  [-1, ndensity])
                rho_r1_part = tf.gather(rho_r1, ind_c_at_i)  # this is "energy" already
                rho = tf.einsum('jb,bd->jd', basis[c_i], coefs) + rho_r1_part
                safe_rho = tf.where(tf.not_equal(rho, 0.), rho, rho + 1e-32)
                en_sum = 0
                for dens in range(self.ndensity - self.n_e_atomic_prop):
                    en_sum += tf.constant(fs_parameters[2 * dens], dtype=self.dtypes.float) \
                              * self.embedding_function(safe_rho[:, dens],
                                                        tf.constant(fs_parameters[2 * dens + 1], dtype=self.dtypes.float),
                                                        ftype=self.embed_spec[c_i].npoti)
                at_nrgs = tf.tensor_scatter_nd_update(at_nrgs, tf.reshape(ind_c_at_i, [-1, 1]),
                                                      tf.reshape(en_sum, [-1, 1]))

                at_nrgs_chiJ = tf.tensor_scatter_nd_update(at_nrgs_chiJ, tf.reshape(ind_c_at_i, [-1, 1]),
                                                           safe_rho[:, self.ndensity - self.n_e_atomic_prop:])
        else:
            for c_i in range(self.nelements):
                ndensity = self.embed_spec[c_i].ndensity
                fs_parameters = self.embed_spec[c_i].FS_parameters
                ind_c_at_i = self.tensor_input[DATA_MU_IJ][self.tensor_input[DATA_SLICE_MU_IJ][self.ncombs + c_i]:
                                                           self.tensor_input[DATA_SLICE_MU_IJ][self.ncombs + c_i + 1]]
                rho = tf.gather(rho_r1, ind_c_at_i)
                safe_rho = tf.where(tf.not_equal(rho, 0.), rho, rho + 1e-32)
                en_sum = 0
                for dens in range(self.ndensity - self.n_e_atomic_prop):
                    en_sum += tf.constant(fs_parameters[2 * dens], dtype=self.dtypes.float) \
                              * self.embedding_function(safe_rho[:, dens],
                                                        tf.constant(fs_parameters[2 * dens + 1], dtype=self.dtypes.float),
                                                        ftype=self.embed_spec[c_i].npoti)
                at_nrgs = tf.tensor_scatter_nd_update(at_nrgs, tf.reshape(ind_c_at_i, [-1, 1]),
                                                      tf.reshape(en_sum, [-1, 1]))

                at_nrgs_chiJ = tf.tensor_scatter_nd_update(at_nrgs_chiJ, tf.reshape(ind_c_at_i, [-1, 1]),
                                                           safe_rho[:, self.ndensity - self.n_e_atomic_prop:])

        e_atom = tf.math.add(tf.reshape(at_nrgs, [-1, 1]) * inner_cut, total_phi_core, 'atomic_energies')

        chi = tf.reshape(self.f_exp_shsc(at_nrgs_chiJ[:, 0], tf.constant(0.5, dtype=self.dtypes.float)),
                       [-1, 1]) + self.tensor_input[DATA_CHI_0]
        if self.n_e_atomic_prop > 1:
            J = tf.reshape(self.f_exp_shsc(at_nrgs_chiJ[:, 1], tf.constant(0.5, dtype=self.dtypes.float)),
                             [-1, 1]) + self.tensor_input[DATA_J_0]
        else:
            J = self.tensor_input[DATA_J_0]

        COUL_CONST = tf.constant(14.3996, dtype=self.dtypes.float)

        sigma = self.tensor_input[DATA_RADII]
        pisqrt = tf.math.sqrt(tf.constant(np.pi, dtype=self.dtypes.float))
        sqrttwo = tf.math.sqrt(tf.constant(2, dtype=self.dtypes.float))
        eta = 1. / sqrttwo / self.g_ewald

        energy_temp_larger_than_zero, kmesh_larger_than_zero, energy_temp_with_zeros, kmesh_with_zeros, direct_vol = self.get_energy_temp(eta)

        if self.invert_matrix:
            nat = self.max_at
            H = tf.zeros([self.tensor_input[DATA_NUM_OF_STRUCTURES], nat, nat], dtype=self.dtypes.float)

            diagonal = tf.reshape(J - COUL_CONST * sqrttwo / eta / pisqrt + COUL_CONST / sigma / pisqrt,
                                  [-1, 1])

            ca_ind = tf.stack([self.tensor_input[DATA_IND_AT_BATCH], self.tensor_input[DATA_IND_AT_I],
                             self.tensor_input[DATA_IND_AT_J]], axis=1)
            mask = tf.math.equal(self.tensor_input[DATA_IND_AT_I], self.tensor_input[DATA_IND_AT_J])
            inds = tf.reshape(tf.boolean_mask(ca_ind, mask, axis=0), [-1, 1, 3])

            H = tf.tensor_scatter_nd_update(H, inds, diagonal)

            idx1 = tf.gather(self.tensor_input[DATA_ATOMIC_STRUCTURE_MAP], self.tensor_input[DATA_IND_I])
            up = tf.reshape(tf.stack([idx1, self.tensor_input[DATA_IND_S_I], self.tensor_input[DATA_IND_S_J]], axis=-1),
                            [-1, 1, 3])

            sigma_i = tf.gather(sigma, self.tensor_input[DATA_IND_I])

            ups = tf.reshape(COUL_CONST * tf.math.erfc(d_ij / eta / sqrttwo) / d_ij -
                             COUL_CONST * tf.math.erfc(d_ij / sigma_i / sqrttwo) / d_ij, [-1, 1])
            H = tf.tensor_scatter_nd_add(H, up, ups)

            pos_ext = tf.zeros([self.tensor_input[DATA_NUM_OF_STRUCTURES], nat, 3], dtype=self.dtypes.float)

            idx = tf.boolean_mask(tf.stack([self.tensor_input[DATA_IND_AT_BATCH], self.tensor_input[DATA_IND_AT_I]],
                                           axis=1), mask)
            pos_ext = tf.tensor_scatter_nd_update(pos_ext, idx, pos)

            updates, indices, updates_down, indices_down = self.get_reciprocal_matrix_elements(pos_ext,
                            energy_temp_larger_than_zero, energy_temp_with_zeros, direct_vol, kmesh_larger_than_zero, kmesh_with_zeros)

            updates = tf.reshape(COUL_CONST * updates, [-1, 1])
            updates_down = tf.reshape(COUL_CONST * updates_down, [-1, 1])

            H = tf.tensor_scatter_nd_add(H, indices, updates)
            H = tf.tensor_scatter_nd_add(H, indices_down, updates_down)

#            tf.print('TYPESSSSS', nat.dtype, self.tensor_input[DATA_NUM_OF_STRUCTURES].dtype, summarize=-1)
            num_structures = tf.cast(self.tensor_input[DATA_NUM_OF_STRUCTURES], tf.int64)
#            dtype_nat = tf.dtypes.as_dtype(nat).base_dtype  # Get the dtype of nat
#            num_structures = tf.cast(self.tensor_input[DATA_NUM_OF_STRUCTURES], dtype_nat)

            indx = tf.stack([tf.range(num_structures),
                             tf.fill([self.tensor_input[DATA_NUM_OF_STRUCTURES]], nat),
                             tf.fill([self.tensor_input[DATA_NUM_OF_STRUCTURES]], nat)], axis=-1)
            H_pad = tf.tensor_scatter_nd_update(tf.pad(H, [[0, 0], [0, 1], [0, 1]], constant_values=1),
                                                indx, tf.zeros(self.tensor_input[DATA_NUM_OF_STRUCTURES],
                                                               dtype=self.dtypes.float))

            chi_ext = tf.zeros([self.tensor_input[DATA_NUM_OF_STRUCTURES], nat], dtype=self.dtypes.float)
            chi_ext = tf.tensor_scatter_nd_update(chi_ext, idx, tf.reshape(-chi, [-1]))

            b = tf.concat([chi_ext, tf.reshape(self.tensor_input[DATA_TOTAL_NUM_ELEC], [-1, 1])], axis=1)

            q_pred = tf.einsum('lij,lj->li', tf.linalg.pinv(H_pad), b)[:, :-1]

            idx_q = tf.boolean_mask(tf.math.add(self.tensor_input[DATA_IND_AT_I],
                                                tf.math.scalar_mul(nat, self.tensor_input[DATA_IND_AT_BATCH])), mask)

            q = tf.reshape(q_pred, [-1, 1])

        e_qeq = tf.reshape(0.5 * tf.einsum('ik,ikj->ij', q_pred, H) * q_pred, [-1, 1]) - tf.reshape(chi_ext, [-1, 1]) * q

        return e_atom + tf.gather(e_qeq, idx_q), tf.gather(q, idx_q)

    def f_exp_shsc_pos(self, rho, mexp):
        eps = tf.constant(1e-10, dtype=self.dtypes.float)
        cond = tf.abs(tf.ones_like(rho, dtype=self.dtypes.float) * mexp - tf.constant(1., dtype=self.dtypes.float))
        mask = tf.where(tf.less(cond, eps), tf.ones_like(rho, dtype=tf.bool), tf.zeros_like(rho, dtype=tf.bool))

        arho = tf.abs(rho)
        # func = tf.where(mask, rho, tf.sign(rho) * (tf.sqrt(tf.abs(arho + 0.25 * tf.exp(-arho))) - 0.5 * tf.exp(-arho)))
        exprho = tf.exp(-arho)
        nx = 1. / mexp
        xoff = tf.pow(nx, (nx / (1.0 - nx))) * exprho
        yoff = tf.pow(nx, (1 / (1.0 - nx))) * exprho
        func = tf.where(mask, rho, (tf.pow(xoff + arho, mexp) - yoff))
        return func

    def compute_atomic_energy(self, r_ij: tf.Tensor, input: dict[str, tf.Tensor]) -> [tf.Tensor, tf.Tensor]:
#    def compute_atomic_energy(self, r_ij_pos: [tf.Tensor, tf.Tensor], input: dict[str, tf.Tensor]) -> [tf.Tensor, tf.Tensor]:
        self.tensor_input = input
#        r_ij = r_ij_pos[0]
        pos = self.tensor_input[DATA_POSITIONS]
        # e_atom, de_dq, e_qeq = self.eval_atomic_energy(r_ij, pos)
        e_atom, de_dq = self.eval_atomic_energy(r_ij, pos)

        if self.compute_smoothness:
            self.compute_smoothness_reg()

        # return e_atom, de_dq, e_qeq
        return e_atom, de_dq

    def get_real_space_cell(self):
        cell = self.tensor_input[DATA_CELL]
        a1, a2, a3 = cell[:, 0, :], cell[:, 1, :], cell[:, 2, :]
        cross = tf.linalg.cross(a2, a3)
        vol = tf.stop_gradient(tf.reshape(tf.einsum('ij,ij->i', a1, cross), [-1, 1]))

        return a1, a2, a3, vol

    def reduce_number_kpoints_by_symmetry(self, kvecs_nozero):
        # select combinations with no zeros
        mask_no_zeros = tf.where(tf.reduce_any(tf.math.equal(kvecs_nozero, 0.), axis=1), 0, 1)
        kvecs_no_zeros = tf.boolean_mask(kvecs_nozero, mask_no_zeros)

        # select combinations that contain one or two zeros
        mask_zeros = tf.where(tf.reduce_any(tf.math.equal(kvecs_nozero, 0.), axis=1), 1, 0)
        kvecs_with_zeros = tf.boolean_mask(kvecs_nozero, mask_zeros)

        # select only m1 >0 , m2 > 0, m3 > 0
        larger_than_zero = tf.math.greater(kvecs_no_zeros, 0.)
        mask = tf.where(tf.reduce_all(larger_than_zero, axis=1), 1, 0)
        kvecs_larger_than_zero = tf.boolean_mask(kvecs_no_zeros, mask)

        return kvecs_larger_than_zero, kvecs_with_zeros

    def compute_energy_temp(self, kvecs_nozero, rec_lattice, eta):

        kmesh = tf.stop_gradient(tf.einsum('ij,jlkm->ilkm', kvecs_nozero, rec_lattice))

        k_sq = tf.linalg.norm(kmesh, axis=-1) ** 2
        energy_temp = tf.stop_gradient(tf.math.exp(-(k_sq * eta ** 2) / 2) / k_sq)

        return energy_temp, kmesh

    def get_energy_temp(self, eta):
        twopi = tf.constant(2. * np.pi, dtype=self.dtypes.float)
        a1, a2, a3, direct_vol = self.get_real_space_cell()

        b1 = twopi * tf.linalg.cross(a2, a3) / direct_vol
        b2 = twopi * tf.linalg.cross(a3, a1) / direct_vol
        b3 = twopi * tf.linalg.cross(a1, a2) / direct_vol
        rec_lattice = tf.expand_dims(tf.stack([b1, b2, b3]), axis=-2)

        n_kvecs = tf.constant(self.nkpoints, dtype=self.dtypes.int)
        r_n = tf.range(-n_kvecs, n_kvecs + 1)
        kvecs = tf.cast(tf.reshape(tf.transpose(tf.meshgrid(r_n, r_n, r_n)), [-1, 3]), dtype=self.dtypes.float)

        # get rid of the origin
        equal = tf.math.equal(kvecs, tf.constant([[0., 0., 0.]], dtype=self.dtypes.float))
        mask = tf.where(tf.reduce_all(equal, axis=1), 0, 1)
        kvecs_nozero = tf.boolean_mask(kvecs, mask)

        kvecs_larger_than_zero, kvecs_with_zeros = self.reduce_number_kpoints_by_symmetry(kvecs_nozero)

        energy_temp_larger_than_zero, kmesh_larger_than_zero = self.compute_energy_temp(kvecs_larger_than_zero, rec_lattice, eta)
        energy_temp_with_zeros, kmesh_with_zeros = self.compute_energy_temp(kvecs_with_zeros, rec_lattice, eta)

        return energy_temp_larger_than_zero, kmesh_larger_than_zero, energy_temp_with_zeros, kmesh_with_zeros, direct_vol

    def square_structure_factor_matrix_elements_with_zeros(self, r_ij, kmesh, energy_temp, direct_vol):
        twopi = tf.constant(2. * np.pi, dtype=self.dtypes.float)

        direct_vol_at = tf.gather(direct_vol, self.tensor_input[DATA_IND_AT_BATCH], axis=0)

        kmesh_at = tf.gather(kmesh, self.tensor_input[DATA_IND_AT_BATCH], axis=1)
        energy_temp_at = tf.gather(energy_temp, self.tensor_input[DATA_IND_AT_BATCH], axis=1)

        kpos = tf.einsum('ijlk,jk->ijl', kmesh_at, r_ij)
        real = tf.math.cos(kpos)

        energy_reciprocal = 2 * (twopi / direct_vol_at) * tf.reduce_sum(real * energy_temp_at, axis=0)

        return energy_reciprocal

    def square_structure_factor_matrix_elements(self, r_ij, kmesh, energy_temp, direct_vol):
        twopi = tf.constant(2. * np.pi, dtype=self.dtypes.float)

        direct_vol_at = tf.gather(direct_vol, self.tensor_input[DATA_IND_AT_BATCH], axis=0)

        kmesh_at = tf.gather(kmesh, self.tensor_input[DATA_IND_AT_BATCH], axis=1)
        energy_temp_at = tf.gather(energy_temp, self.tensor_input[DATA_IND_AT_BATCH], axis=1)

        kpos = tf.einsum('ijlk,jk->ijlk', kmesh_at, r_ij)
        real = 8. * tf.math.reduce_prod(tf.math.cos(kpos), axis=-1)

        energy_reciprocal = 2 * (twopi / direct_vol_at) * tf.reduce_sum(real * energy_temp_at, axis=0)

        return energy_reciprocal

    def get_reciprocal_matrix_elements(self, pos, energy_temp_larger_than_zero, energy_temp_with_zeros, direct_vol,
                                       kmesh_larger_than_zero, kmesh_with_zeros):

        idx1 = tf.stack([self.tensor_input[DATA_IND_AT_BATCH], self.tensor_input[DATA_IND_AT_I]], axis=1)
        idx2 = tf.stack([self.tensor_input[DATA_IND_AT_BATCH], self.tensor_input[DATA_IND_AT_J]], axis=1)

        diff = tf.gather_nd(pos, idx2) - tf.gather_nd(pos, idx1)

        ind = tf.stack([self.tensor_input[DATA_IND_AT_BATCH], self.tensor_input[DATA_IND_AT_I],
                               self.tensor_input[DATA_IND_AT_J]], axis=-1)

        energy_reciprocal_larger_than_zero = self.square_structure_factor_matrix_elements(diff, kmesh_larger_than_zero,
                                                                                 energy_temp_larger_than_zero, direct_vol)

        energy_reciprocal_with_zeros = self.square_structure_factor_matrix_elements_with_zeros(diff, kmesh_with_zeros,
                                                                         energy_temp_with_zeros, direct_vol)

        energy_reciprocal = tf.math.add(energy_reciprocal_larger_than_zero, energy_reciprocal_with_zeros)

        up = tf.reshape(ind, [-1, 1, 3])

        mask = tf.math.not_equal(self.tensor_input[DATA_IND_AT_I], self.tensor_input[DATA_IND_AT_J])
        id_i = tf.expand_dims(tf.boolean_mask(self.tensor_input[DATA_IND_AT_I], mask), axis=0)
        id_j = tf.expand_dims(tf.boolean_mask(self.tensor_input[DATA_IND_AT_J], mask), axis=0)
        id_0 = tf.expand_dims(tf.boolean_mask(self.tensor_input[DATA_IND_AT_BATCH], mask), axis=0)
        down = tf.reshape(tf.stack([id_0, id_j, id_i], axis=-1), [-1, 1, 3])
        energy_reciprocal_down = tf.boolean_mask(energy_reciprocal, mask)

        return energy_reciprocal, up, energy_reciprocal_down, down

    def selective_fitting(self, list_of_bonds, basis_factor=1., rad_coefs_factor=1.):
        assert len(list_of_bonds) <= len(self.bond_combs), \
            ValueError('Number of requested bond types ({}) exceeds the number specified in the potential ({})'.format(
                len(list_of_bonds), len(self.bond_combs)))
        for bond in list_of_bonds:
            assert bond in self.bond_combs, \
                ValueError("Bond type {} is not in the potential's list of bond combinations".format(bond))
        assert basis_factor in (0., 1.), ValueError('Only 0 and 1 are allowed values for a factor')
        assert rad_coefs_factor in (0., 1.), ValueError('Only 0 and 1 are allowed values for a factor')

        factor_list = np.zeros_like(self.tmp_coefs).astype(np.float64)
        for bond in list_of_bonds:
            factor_list[self.coefs_rk_part[bond]] = basis_factor
            factor_list[slice(*self.bond_to_slice[bond])] = rad_coefs_factor

        return factor_list
