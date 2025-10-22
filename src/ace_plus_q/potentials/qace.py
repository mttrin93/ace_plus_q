import numpy as np
import tensorflow as tf
import pathlib

from ace_plus_q.functions import spherical_harmonics, radial_functions
from ace_plus_q.data.datakeys import *
from ace_plus_q.potentials.potential import BACE
from itertools import product, combinations_with_replacement
from ace_plus_q.data.symbols import symbol_to_atomic_number
from ace_plus_q.data.tpatoms import sort_elements
# tf.debugging.set_log_device_placement(True)

# TF_PI = tf.constant(np.pi, dtype=tf.float64)
# COUL_CONST = tf.constant(14.3996, dtype=tf.float64)

class QACE(BACE):
    def __init__(self, n_e_atomic_prop, max_at, save_path,
                 chi0_dict, J0_dict, sigma_dict, invert_matrix=False,
                 *args, **kwargs):
        self.n_e_atomic_prop = n_e_atomic_prop
        self.invert_matrix = invert_matrix
        self.max_at = max_at
        self.name_out = save_path
        super().__init__(*args, **kwargs)

        elms = self.get_chemical_symbols()
        elms = sort_elements(elms)

        self.chi0_dict = chi0_dict
        # self.chi0_arr = tf.constant([self.chi0_dict[e] for e in elms], dtype=self.dtypes.float, name='chi0_arr')
        # ensure it's a Python list (not a tf tensor)
        elms = [str(e) for e in elms]

        # compute maximum atomic number among elements present (or among keys in chi0_dict)
        max_z_data = max(symbol_to_atomic_number[s] for s in elms)  # max Z in your data
        max_z_params = max(symbol_to_atomic_number[s] for s in self.chi0_dict)  # max Z in your chi0_dict
        max_z = max(max_z_data, max_z_params)

        # create numpy array length max_z+1 (index by atomic number)
        chi0_np = np.zeros((max_z + 1,), dtype=np.float32)  # use float32 if TensorFlow float32
        # fill from chi0_dict (skip unknown symbols if any)
        for sym, val in self.chi0_dict.items():
            atomic_number = symbol_to_atomic_number[sym]  # atomic number, e.g. 'O' -> 8
            chi0_np[atomic_number] = float(val)

        # convert to TF constant
        self.chi0_arr = tf.constant(chi0_np, dtype=self.dtypes.float, name='chi0_arr')

        self.J0_dict = J0_dict
        J0_np = np.zeros((max_z + 1,), dtype=np.float32)
        for sym, val in self.J0_dict.items():
            atomic_number = symbol_to_atomic_number[sym]
            J0_np[atomic_number] = float(val)

        self.J0_arr = tf.constant(J0_np, dtype=self.dtypes.float, name='J0_arr')
        # self.J0_arr = tf.constant([self.J0_dict[e] for e in elms], dtype=self.dtypes.float, name="J0_arr")

        self.sigma_dict = sigma_dict
        sigma_np = np.zeros((max_z + 1,), dtype=np.float32)
        for sym, val in self.J0_dict.items():
            atomic_number = symbol_to_atomic_number[sym]
            sigma_np[atomic_number] = float(val)

        self.sigma_arr = tf.constant(sigma_np, dtype=self.dtypes.float, name='sigma_arr')

        # self.sigma_arr = tf.constant([self.sigma_dict[e] for e in elms], dtype=self.dtypes.float, name="sigma_arr")

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


    def eval_atomic_energy(self, r_ij):
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

        atomic_mu_index = tf.cast(tf.math.unsorted_segment_mean(self.tensor_input[DATA_MU_I],
                                                                self.tensor_input[DATA_IND_I],
                                                                num_segments=self.tensor_input[DATA_NUM_OF_ATOMS]),
                                  dtype=self.dtypes.int)

        chi0 = tf.reshape(tf.gather(self.chi0_arr, atomic_mu_index, axis=0), (-1, 1))
        chi = tf.reshape(self.f_exp_shsc(at_nrgs_chiJ[:, 0], tf.constant(0.5, dtype=self.dtypes.float)),
                         [-1, 1]) + chi0   #TODO: remove DATA_CHI_0 key from everywhere
        self.chi = chi

        J0 = tf.reshape(tf.gather(self.J0_arr, atomic_mu_index, axis=0), (-1, 1))
        if self.n_e_atomic_prop > 1:
            J = tf.reshape(self.f_exp_shsc(at_nrgs_chiJ[:, 1], tf.constant(0.5, dtype=self.dtypes.float)),
                           [-1, 1]) + J0
        else:
            J = J0
        self.J = J

        COUL_CONST = tf.constant(14.399645, dtype=self.dtypes.float)

        sigma = tf.reshape(tf.gather(self.sigma_arr, atomic_mu_index, axis=0), (-1, 1))
        pisqrt = tf.math.sqrt(tf.constant(np.pi, dtype=self.dtypes.float))
        twosqrt = tf.math.sqrt(tf.constant(2, dtype=self.dtypes.float))

        if self.invert_matrix:
            nat = tf.cast(self.max_at, dtype=self.dtypes.int)   # max number of atoms
            H = tf.zeros([self.tensor_input[DATA_NUM_OF_STRUCTURES], nat, nat], dtype=self.dtypes.float)

            diagonal = tf.reshape(J + COUL_CONST / sigma / pisqrt, [-1, 1])

            ca_ind = tf.stack([self.tensor_input[DATA_IND_AT_BATCH], self.tensor_input[DATA_IND_AT_I],
                             self.tensor_input[DATA_IND_AT_J]], axis=1)
            mask = tf.math.equal(self.tensor_input[DATA_IND_AT_I], self.tensor_input[DATA_IND_AT_J])
            inds = tf.reshape(tf.boolean_mask(ca_ind, mask, axis=0), [-1, 1, 3])

            H = tf.tensor_scatter_nd_update(H, inds, diagonal)

            idx1 = tf.gather(self.tensor_input[DATA_ATOMIC_STRUCTURE_MAP], self.tensor_input[DATA_IND_I])
            up = tf.reshape(tf.stack([idx1, self.tensor_input[DATA_IND_S_I], self.tensor_input[DATA_IND_S_J]], axis=-1),
                            [-1, 1, 3])

            sigma_i = tf.gather(sigma, self.tensor_input[DATA_IND_I])
            sigma_j = tf.gather(sigma, self.tensor_input[DATA_IND_J])
            gamma = tf.math.sqrt(sigma_i ** 2 + sigma_j ** 2)

            ups = tf.reshape(COUL_CONST * tf.math.erf(d_ij / gamma / twosqrt) / d_ij, [-1, 1])

            H = tf.tensor_scatter_nd_update(H, up, ups)

            H_pad = tf.pad(H, [[0, 0], [0, 1], [0, 1]], constant_values=0)

            n_real_at = tf.unique_with_counts(self.tensor_input[DATA_ATOMIC_STRUCTURE_MAP])[2]
            id0 = tf.boolean_mask(self.tensor_input[DATA_IND_AT_BATCH], mask, axis=0)

            # indices to fill the needed elements with ones both vertically and horizontally
            idx_v = tf.stack([id0, tf.repeat(n_real_at, repeats=n_real_at),
                              tf.boolean_mask(self.tensor_input[DATA_IND_AT_I], mask, axis=0)], axis=-1)
            idx_o = tf.reshape(tf.gather(idx_v, [[0, 2, 1]], axis=1), [-1, 3])

            H_pad = tf.tensor_scatter_nd_update(H_pad, idx_v, tf.ones_like(id0, dtype=self.dtypes.float))
            H_pad = tf.tensor_scatter_nd_update(H_pad, idx_o, tf.ones_like(id0, dtype=self.dtypes.float))

            idx = tf.boolean_mask(tf.stack([self.tensor_input[DATA_IND_AT_BATCH], self.tensor_input[DATA_IND_AT_I]],
                                           axis=1), mask)

            b = tf.zeros([self.tensor_input[DATA_NUM_OF_STRUCTURES], nat + 1], dtype=self.dtypes.float)
            b = tf.tensor_scatter_nd_update(b, idx, tf.reshape(-chi, [-1]))

            indx = tf.stack([tf.range(self.tensor_input[DATA_NUM_OF_STRUCTURES]), n_real_at], axis=1)
            b = tf.tensor_scatter_nd_update(b, indx, tf.reshape(self.tensor_input[DATA_TOTAL_CHRG], [-1]))

            q_pred = tf.einsum('lij,lj->li', tf.linalg.pinv(H_pad), b)

            # indices to remove fake charges if Nat < self.max_at
            idx_q = tf.boolean_mask(tf.math.add(self.tensor_input[DATA_IND_AT_I],
                                                tf.math.scalar_mul(nat + 1, self.tensor_input[DATA_IND_AT_BATCH])),
                                    mask)

            # indices to remove energy contrinutions from fake atoms if Nat < self.max_at
            idx_e = tf.boolean_mask(tf.math.add(self.tensor_input[DATA_IND_AT_I],
                                                tf.math.scalar_mul(nat, self.tensor_input[DATA_IND_AT_BATCH])), mask)

            q = tf.reshape(q_pred, [-1, 1])

            q_ext = tf.zeros([self.tensor_input[DATA_NUM_OF_STRUCTURES], nat], dtype=self.dtypes.float)
            q_ext = tf.tensor_scatter_nd_update(q_ext, idx, tf.reshape(tf.gather(q, idx_q), [-1]))

        e_qeq = tf.reshape(0.5 * tf.einsum('ik,ikj->ij', q_ext, H) * q_ext, [-1, 1])
        e_qeq = tf.gather(e_qeq, idx_e) + chi * tf.gather(q, idx_q)
        
        return e_atom + e_qeq, tf.gather(q, idx_q)

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
        self.tensor_input = input
        e_atom, de_dq = self.eval_atomic_energy(r_ij)

        if self.compute_smoothness:
            self.compute_smoothness_reg()

        return e_atom, de_dq

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
