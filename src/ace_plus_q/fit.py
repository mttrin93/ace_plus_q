import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import time, os
import warnings
from scipy.optimize import minimize, dual_annealing
from typing import Any, Tuple, Dict, Union, Optional, Callable

from ace_plus_q import TensorPotential
from src.ace_plus_q.data import TPBatch, TPAtomsDataContainer
from src.ace_plus_q.graphspecs import *

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

NUMPY_OPTIMIZERS = ['BFGS', 'L-BFGS-B', 'CG', 'trf', 'dogbox', 'lm', 'dual_annealing']
TF_OPTIMIZERS = ['Adam', 'SGD']


class FitTensorPotential:
    def __init__(self, tensorpot: TensorPotential):
        self.tensorpot = tensorpot
        self.ncoefs = self.tensorpot.potential.get_number_of_coefficients()
        self.fit_coefs = None
        self.jacobian_factor = None
        self.loss_history = []
        self.iter_num = 0
        self.eval_count = 0
        self.global_callback = None
        self.display_step = None
        self.data_df = None
        self.test_df = None
        self.res_opt = None
        self.batch_size = 1
        self.n_batches = None
        self.metrics = None
        self.test_metrics = None
        self.optmzr = None
        self.fit_metrics_data_dict = {}
        self.last_fit_metric_data = None
        self.last_test_metric_data = None
        self.fit_metric_callback = None
        self.test_metric_callback = None
        self.nfuncs = None
        self.opt = None

    def fit(self, df: Union[list[TPAtomsDataContainer], pd.DataFrame],
            test_df: Optional[Union[list[TPAtomsDataContainer], pd.DataFrame]] = None,
            display_step: int = 20, batch_size: int = 1, optimizer: str = 'BFGS', niter: int = 10,
            coefs: Optional[Any] = None, jacobian_factor: Optional[Any] = None, callback: Optional[Callable] = None,
            save_coefs: Optional[bool] = False,
            optimizer_options: Optional[dict[str, Any]] = None,
            fit_metric_callback: Optional[Callable] = None,
            test_metric_callback: Optional[Callable] = None):

        if fit_metric_callback is not None:
            self.fit_metric_callback = fit_metric_callback
        if test_metric_callback is not None:
            self.test_metric_callback = test_metric_callback
        self.display_step = display_step

        self.data_df = df
        self.batch_size = batch_size
        self.potentials_elements = self.tensorpot.potential.get_chemical_symbols()
        self.potentials_list_of_bonds = self.tensorpot.potential.get_bond_symbol_combinations()
        self.potentials_bond_indexing = self.tensorpot.potential.bond_indexing

        batches = TPBatch(self.data_df, batch_size=batch_size, list_of_elements=self.potentials_elements,
                          list_of_bond_symbol_combinations=self.potentials_list_of_bonds,
                          bond_indexing=self.potentials_bond_indexing, shuffle=False).batches
        self.n_batches = len(batches)

        w_e = np.vstack([b[DATA_ENERGY_WEIGHTS] for b in batches])
        w_f = np.vstack([b[DATA_FORCE_WEIGHTS] for b in batches])
        # force_scale = float(self.tensorpot.loss_force_factor)
        force_scale = float(self.tensorpot.loss_specs[SPEC_LOSS_FORCE_FACTOR])
        energy_scale = float(self.tensorpot.loss_specs[SPEC_LOSS_ENERGY_FACTOR])
        self.metrics = FitMetrics(w_e, w_f, energy_scale, force_scale, self.ncoefs)
        self.metrics.nfuncs = self.nfuncs

        if test_df is not None:
            self.test_df = test_df
            test_batches = TPBatch(self.test_df, batch_size=batch_size, list_of_elements=self.potentials_elements,
                                   list_of_bond_symbol_combinations=self.potentials_list_of_bonds,
                                   bond_indexing=self.potentials_bond_indexing).batches
            test_w_e = np.vstack([b[DATA_ENERGY_WEIGHTS] for b in test_batches])
            test_w_f = np.vstack([b[DATA_FORCE_WEIGHTS] for b in test_batches])
            self.test_metrics = FitMetrics(test_w_e, test_w_f, energy_scale, force_scale, self.ncoefs)
            self.test_metrics.nfuncs = self.nfuncs

        self.global_callback = callback
        if jacobian_factor is not None:
            self.jacobian_factor = np.array(jacobian_factor)

        if coefs is None:
            coefs = self.tensorpot.potential.get_coefs().numpy()
        elif coefs is not None:
            try:
                assert len(coefs) == self.ncoefs
            except:
                raise ValueError(f'Number of provided coefficients {len(coefs)} is '
                                 f'different from the numbers used in model {self.ncoefs}')

        t0 = time.perf_counter()
        if optimizer in NUMPY_OPTIMIZERS:
            # TODO: Make args passable
            if optimizer in ['trf', 'dogbox', 'lm']:
                from scipy.optimize import least_squares
                warnings.warn(f'You have specified an {optimizer=}. LSTSQ optimization is not very well tested,'
                              f' do not expect it to work stable if at all!', Warning)
                # if self.tensorpot.potential.__class__.__name__ == 'ACE':
                #     assert self.tensorpot.potential.rankmax == 1, 'Only pair potential of Atomic Cluster Expansion is' \
                #                                                   ' supported for the least squares algorithms'

                ls_opt_func = LMfunc(self, w_e)

                res_opt = least_squares(ls_opt_func.func, coefs, jac=ls_opt_func.jac,
                                        method=optimizer, args=(batches,), ftol=1e-8, verbose=1, max_nfev=niter)
            elif optimizer == 'dual_annealing':
                warnings.warn(f'You have specified an {optimizer=}. Dual annealing optimization is not very well tested,'
                              f' do not expect it to work stable if at all!', Warning)
                da_opt_func = DAfunc(self)

                bounds = [(-10000, 10000) for _ in coefs]
                options = {'disp': True, 'maxiter': niter, 'jac': False}
                minimizer_kwargs = {'method': 'BFGS',
                                    # 'args':(batches,),
                                    'jac': da_opt_func.jac,
                                    'options': {'gtol': 1e-8, 'disp': True, 'maxiter': niter}}
                res_opt = dual_annealing(da_opt_func.func, bounds, args=(batches,), maxiter=niter,
                                         local_search_options=minimizer_kwargs, callback=self.callback, x0=coefs)
                #     minimizer_kwargs = minimizer_kwargs, options=options, )

            else:
                if optimizer_options is None:
                    optimizer_options = {'gtol': 1e-8, 'disp': True, 'maxiter': niter}
                optimizer_options["maxiter"] = niter
                log.info("Minimizer options: {}".format(optimizer_options))
                # reset fit_metrics_data_dict here and every time in self.callback. No need to store complete history
                self.fit_metrics_data_dict = {}
                if self.tensorpot.mode == 'train':
                    res_opt = minimize(self.fit_func, coefs, method=optimizer, jac=True,
                                       args=(batches,), options=optimizer_options,
                                       callback=self.callback)
                    self.process_test_metric()
                elif self.tensorpot.mode == 'scf_train':
                    collect_nel = []
                    scf_param_size = [0]
                    for b in batches:
                        scf_param = b[DATA_ATOMIC_NUM_ELEC].flatten()
                        # scf_param = b[DATA_CENTERS].flatten()
                        scf_param_size.append(len(scf_param))
                        collect_nel.extend(scf_param)
                    # self.save_batch(batches)
                    total_opt_param = np.concatenate([coefs, collect_nel])
                    res_opt = minimize(self.scf_fit_func, total_opt_param, method=optimizer, jac=True,
                                       args=(batches, scf_param_size), options=optimizer_options,
                                       callback=self.callback)
                    self.process_test_metric()
                    # res_opt = minimize(self.fit_func, coefs, method=optimizer, jac=True,
                    #                    args=(batches,), options=optimizer_options,
                    #                    callback=self.callback)


            self.res_opt = res_opt
            self.fit_coefs = res_opt.x
            self.tensorpot.potential.set_coefs(self.fit_coefs[:self.ncoefs])

        elif optimizer in TF_OPTIMIZERS:
            self.opt = self.get_optimzer(optimizer, optimizer_options)
            for epoch in range(niter):
                loss = self.tf_fit_func(batches)
                self.fit_coefs = self.tensorpot.potential.get_coefs().numpy()
                self.callback(self.fit_coefs)
        # TODO: else

        if save_coefs:
            self.save_coefs()
        # x = res_opt.x
        x = self.fit_coefs
        self.update_last_metrics(x)
        print('Fitting took {0:6.2f} seconds'.format(time.perf_counter() - t0))

    def save_batch(self, batches):
        for i, b in enumerate(batches):
            np.savetxt('opt_chgs/batch_{}'.format(i), b[DATA_ATOMIC_NUM_ELEC])
            np.savetxt('opt_chgs/ind_{}'.format(i), b[DATA_ATOMIC_STRUCTURE_MAP])

    def restart_fit(self, df, batch_size=100, optimizer='BFGS', niter=100, jacobian_factor=None):
        if self.eval_count == 0:
            raise AttributeError('Cannot restart, no fit has been done so far')
        else:
            self.fit(df, batch_size=batch_size, optimizer=optimizer, coefs=self.fit_coefs, niter=niter,
                     jacobian_factor=jacobian_factor)

    def get_optimzer(self, opt_name, optimizer_options):
        if opt_name == 'Adam':
            optmzr = tf.keras.optimizers.Adam(**optimizer_options)
            # optmzr = tf.keras.optimizers.Nadam(**optimizer_options)
        elif opt_name == 'SGD':
            optmzr = tf.keras.optimizers.SGD(**optimizer_options)
        else:
            optmzr = None
        log.info("Minimizer options: {}".format(optmzr.get_config()))

        return optmzr

    def tf_fit_func(self, batches):
        tot_loss = 0
        de, de_pa, df = ([] for _ in range(3))
        total_na = []
        t0 = time.perf_counter()
        idx = np.arange(len(batches))
        np.random.shuffle(idx)
        #log.info("Minimizer options: {}".format(self.opt.get_weights()))
        for ind in idx:
            b = batches[ind]
            loss_tup, data_tup = self.tensorpot.native_fit(b)
            self.opt.apply_gradients(zip([loss_tup[1]], [self.tensorpot.potential.fit_coefs]))
            tot_loss += loss_tup[0].numpy()
            if len(data_tup) == 3:
                e = data_tup[0].numpy()
                f = data_tup[1].numpy()
                s = data_tup[2].numpy()
            elif len(data_tup) == 2:
                e = data_tup[0].numpy()
                f = data_tup[1].numpy()
                s = 0
            else:
                e = data_tup[0].numpy()
                f = s = 0

            errors = self.compute_batch_errors(b, e, f)
            de.append(errors[0])
            de_pa.append(errors[1])
            df.append(errors[2])
            total_na.append(errors[3])

        self.eval_time = time.perf_counter() - t0
        self.fit_coefs = self.tensorpot.potential.get_coefs().numpy()
        self.eval_count += 1
        # collect all relevant info into FitMetrics
        self.metrics.regs = loss_tup[2].numpy().reshape(-1, 1).astype(np.float64)
        self.metrics.reg_weights = self.get_reg_weights()
        self.metrics.compute_metrics(np.vstack(de), np.vstack(de_pa), np.vstack(df),
                                     np.vstack(total_na), dataframe=self.data_df)
        self.metrics.loss = tot_loss
        self.metrics.eval_time = self.eval_time
        # do a snapshot of  FitMetrics into FitMetricsData: loss, e_loss, f_loss, reg_loss, RMSE, MAE, MAX_E (E,F), timing
        curr_fit_metrics_data = self.metrics.to_FitMetricsDict()
        curr_fit_metrics_data["eval_count"] = self.eval_count
        curr_fit_metrics_data["iter_num"] = self.iter_num
        # store metrics_data into dict (x-> curr_fit_metrics_data)
        self.fit_metrics_data_dict[hash(self.fit_coefs.tobytes())] = curr_fit_metrics_data
        self.last_fit_metric_data = curr_fit_metrics_data

        if self.iter_num == 0:
            if self.fit_metric_callback is not None:
                self.fit_metric_callback(curr_fit_metrics_data)
            else:
                print_detailed_metrics(curr_fit_metrics_data, title='Initial state:')
                print_extended_metrics(curr_fit_metrics_data, title='INIT STATS')

            self.process_test_metric(title="TEST INIT STATS")
            self.iter_num += 1

        self.loss_history.append(tot_loss)
        self.last_loss = self.loss_history[-1]

        return tot_loss

    def fit_func(self, x: np.array, batches: List[Dict[str, Any]]) -> Tuple[np.array, np.array]:
        tot_loss = 0
        tot_jac = 0
        de, de_pa, df = ([] for _ in range(3))
        total_na = []
        t0 = time.perf_counter()
        for b in batches:
            loss_tup, data_tup = self.tensorpot.external_fit(x, b)
            tot_jac += loss_tup[1].numpy().astype(np.float64)
            tot_loss += loss_tup[0].numpy().astype(np.float64)
            if len(data_tup) == 3:
                e = data_tup[0].numpy().astype(np.float64)
                f = data_tup[1].numpy().astype(np.float64)
                s = data_tup[2].numpy().astype(np.float64)
            elif len(data_tup) == 2:
                e = data_tup[0].numpy().astype(np.float64)
                f = data_tup[1].numpy().astype(np.float64)
                s = 0
            else:
                e = data_tup[0].numpy().astype(np.float64)
                f = s = 0

            errors = self.compute_batch_errors(b, e, f)
            de.append(errors[0])
            de_pa.append(errors[1])
            df.append(errors[2])
            total_na.append(errors[3])

        self.eval_time = time.perf_counter() - t0
        self.fit_coefs = x
        self.eval_count += 1
        # collect all relevant info into FitMetrics
        #self.metrics.regs = self.get_reg_components()
        self.metrics.regs = loss_tup[2].numpy().reshape(-1, 1).astype(np.float64)
        self.metrics.reg_weights = self.get_reg_weights()
        self.metrics.compute_metrics(np.vstack(de), np.vstack(de_pa), np.vstack(df),
                                     np.vstack(total_na), dataframe=self.data_df)
        self.metrics.loss = tot_loss
        self.metrics.eval_time = self.eval_time
        # do a snapshot of  FitMetrics into FitMetricsData: loss, e_loss, f_loss, reg_loss, RMSE, MAE, MAX_E (E,F), timing
        curr_fit_metrics_data = self.metrics.to_FitMetricsDict()
        curr_fit_metrics_data["eval_count"] = self.eval_count
        curr_fit_metrics_data["iter_num"] = self.iter_num
        # store metrics_data into dict (x-> curr_fit_metrics_data)
        self.fit_metrics_data_dict[hash(x.tobytes())] = curr_fit_metrics_data
        self.last_fit_metric_data = curr_fit_metrics_data

        if self.iter_num == 0:
            if self.fit_metric_callback is not None:
                self.fit_metric_callback(curr_fit_metrics_data)
            else:
                print_detailed_metrics(curr_fit_metrics_data, title='Initial state:')
                print_extended_metrics(curr_fit_metrics_data, title='INIT STATS')

            self.process_test_metric(title="TEST INIT STATS")
            self.iter_num += 1

        self.loss_history.append(tot_loss)
        self.last_loss = self.loss_history[-1]

        if self.jacobian_factor is not None:
            try:
                assert len(self.jacobian_factor) == tot_jac.shape[0]
                tot_jac *= self.jacobian_factor
            except:
                raise ValueError(
                    'Size of the provided Jacobian factor ({0}) is not compatible with Jacobian size ({1}).' \
                        .format(len(self.jacobian_factor), tot_jac.shape[0]))

        return tot_loss, tot_jac

    def scf_fit_func(self, x: np.array, batches: List[Dict[str, Any]],
                     scf_param_sizes: List[int]) -> Tuple[np.array, np.array]:
        tot_loss = 0
        #tot_jac = 0
        jac_coefs = 0
        jac_scf = []
        de, de_pa, df = ([] for _ in range(3))
        total_na = []
        t0 = time.perf_counter()
        slices = np.cumsum(scf_param_sizes)
        coefs = x[:self.ncoefs]
        scf_params = x[self.ncoefs:]
        for i, b in enumerate(batches):
            b[DATA_ATOMIC_NUM_ELEC] = scf_params[slices[i]:slices[i + 1]].reshape(-1, 1)
            # b[DATA_CENTERS] = scf_params[slices[i]:slices[i + 1]].reshape(-1, 3)
            loss_tup, data_tup = self.tensorpot.external_fit(coefs, b)
            jac_coefs += loss_tup[1].numpy().astype(np.float64)
            # jac_scf.extend(loss_tup[2].numpy().astype(np.float64))
            jac_scf.extend(loss_tup[2].numpy().astype(np.float64)*0)
            tot_loss += loss_tup[0].numpy().astype(np.float64)
            if len(data_tup) == 3:
                e = data_tup[0].numpy().astype(np.float64)
                f = data_tup[1].numpy().astype(np.float64)
                s = data_tup[2].numpy().astype(np.float64)
            elif len(data_tup) == 2:
                e = data_tup[0].numpy().astype(np.float64)
                f = data_tup[1].numpy().astype(np.float64)
                s = 0
            else:
                e = data_tup[0].numpy().astype(np.float64)
                f = s = 0

            errors = self.compute_batch_errors(b, e, f)
            de.append(errors[0])
            de_pa.append(errors[1])
            df.append(errors[2])
            total_na.append(errors[3])
        tot_jac = np.concatenate([jac_coefs, jac_scf], axis=0)

        self.fit_coefs = x
        self.eval_count += 1
        self.eval_time = time.perf_counter() - t0
        # collect all relevant info into FitMetrics
        #self.metrics.regs = self.get_reg_components()
        self.metrics.regs = loss_tup[2].numpy().reshape(-1, 1).astype(np.float64)
        self.metrics.reg_weights = self.get_reg_weights()
        self.metrics.compute_metrics(np.vstack(de), np.vstack(de_pa), np.vstack(df),
                                     np.vstack(total_na), dataframe=self.data_df)
        self.metrics.loss = tot_loss
        self.metrics.eval_time = self.eval_time
        # do a snapshot of  FitMetrics into FitMetricsData: loss, e_loss, f_loss, reg_loss, RMSE, MAE, MAX_E (E,F), timing
        curr_fit_metrics_data = self.metrics.to_FitMetricsDict()
        curr_fit_metrics_data["eval_count"] = self.eval_count
        curr_fit_metrics_data["iter_num"] = self.iter_num
        # store metrics_data into dict (x-> curr_fit_metrics_data)
        self.fit_metrics_data_dict[hash(x.tobytes())] = curr_fit_metrics_data
        self.last_fit_metric_data = curr_fit_metrics_data

        if self.iter_num == 0:
            if self.fit_metric_callback is not None:
                self.fit_metric_callback(curr_fit_metrics_data)
            else:
                print_detailed_metrics(curr_fit_metrics_data, title='Initial state:')
                print_extended_metrics(curr_fit_metrics_data, title='INIT STATS')

            self.process_test_metric(title="TEST INIT STATS")

            self.iter_num += 1

        self.loss_history.append(tot_loss)
        self.last_loss = self.loss_history[-1]

        if self.jacobian_factor is not None:
            try:
                assert len(self.jacobian_factor) == tot_jac.shape[0]
                tot_jac *= self.jacobian_factor
            except:
                raise ValueError(
                    'Size of the provided Jacobian factor ({0}) is not compatible with Jacobian size ({1}).' \
                        .format(len(self.jacobian_factor), tot_jac.shape[0]))

        return tot_loss, tot_jac

    def compute_batch_errors(self, b, e, f):
        e_true = b[DATA_TOTAL_ENERGY]
        f_true = b[DATA_FORCES]
        _, nat = np.unique(b[DATA_ATOMIC_STRUCTURE_MAP], return_counts=True)
        de = e_true - e
        de_pa = de / nat.reshape(-1, 1)
        df = f_true - f

        return de, de_pa, df, nat.reshape(-1, 1)

    def callback(self, x, *args, **kwargs):
        # call global callback
        self.metrics.record_time(self.eval_time)

        # get true metrics data that correspond to actual successfull x
        true_fit_metric_data = self.update_last_metrics(x)

        if self.fit_metric_callback is not None:
            # example of fit_metric_callback is:  print_extended_metrics
            self.fit_metric_callback(true_fit_metric_data)
        else:
            if self.iter_num % self.display_step == 0:
                print_extended_metrics(true_fit_metric_data)
            else:
                print_detailed_metrics(true_fit_metric_data)
        # test metric
        if self.iter_num % self.display_step == 0:
            self.process_test_metric()

        if self.iter_num % self.display_step == 0:
            self.tensorpot.potential.get_updated_config(prefix=self.iter_num)
            self.tensorpot.potential.save(prefix=self.iter_num)
            if self.opt is not None:
                save_optimizer_state(self.opt, './saves/step_{}/'.format(self.iter_num), 'opt_state')

        self.iter_num += 1
        # clean fit_metrics_data_dict for next iteration
        # self.fit_metrics_data_dict = {}

        if self.global_callback is not None:
            self.global_callback(x, *args, **kwargs)

    def process_test_metric(self, title="TEST_STATS"):
        if self.test_df is not None:
            curr_test_metrics_data = self.compute_test_metric()
            curr_test_metrics_data["eval_count"] = self.eval_count
            curr_test_metrics_data["iter_num"] = self.iter_num
            self.last_test_metric_data = curr_test_metrics_data
            if self.test_metric_callback is not None:
                self.test_metric_callback(curr_test_metrics_data)
            else:
                print_extended_metrics(curr_test_metrics_data, title=title)

    def update_last_metrics(self, x):
        x_hash = hash(x.tobytes())
        true_fit_metric_data = self.fit_metrics_data_dict[x_hash]
        true_fit_metric_data["iter_num"] = self.iter_num
        true_fit_metric_data["eval_count"] = len(self.loss_history)
        # update self.metrics with this true metrics data
        self.metrics.from_FitMetricsDict(true_fit_metric_data)
        self.last_fit_metric_data = true_fit_metric_data
        return true_fit_metric_data

    def predict(self, datadf=None):
        if datadf is None:
            batches = TPBatch(self.test_df, batch_size=self.batch_size, list_of_elements=self.potentials_elements,
                              list_of_bond_symbol_combinations=self.potentials_list_of_bonds,
                              bond_indexing=self.potentials_bond_indexing).batches
        else:
            batches = TPBatch(datadf, batch_size=self.batch_size, list_of_elements=self.potentials_elements,
                              list_of_bond_symbol_combinations=self.potentials_list_of_bonds,
                              bond_indexing=self.potentials_bond_indexing).batches

        pred_dict = {'energy_pred': [],
                     'forces_pred': [],
                     'stress_pred': [],
                     'de': [],
                     'de_pa': [],
                     'df': [],
                     'total_na': []}
        for b in batches:
            _, partition = np.unique(b[DATA_ATOMIC_STRUCTURE_MAP], return_counts=True)
            for i in range(1, len(partition)):
                partition[i] += partition[i - 1]
            loss_tup, data_tup = self.tensorpot.evaluate(b)
            if len(data_tup) == 3:
                e = data_tup[0].numpy()
                f = data_tup[1].numpy()
                s = data_tup[2].numpy()
            elif len(data_tup) == 2:
                e = data_tup[0].numpy()
                f = data_tup[1].numpy()
                s = np.zeros((3, 3))
            else:
                e = data_tup[0].numpy()
                f = np.zeros((b[DATA_NUM_OF_ATOMS], 3))
                s = np.zeros((3, 3))
            errors = self.compute_batch_errors(b, e, f)
            e_pred = e.reshape(-1, )
            f_pred = np.split(f.reshape(-1, 3), partition)[:-1]
            s_pred = s.reshape([-1, 3, 3])
            pred_dict['energy_pred'] += list(e_pred)
            pred_dict['forces_pred'] += list(f_pred)
            pred_dict['stress_pred'] += list(s_pred)
            pred_dict['de'] += list(errors[0].reshape(-1, ))
            pred_dict['de_pa'] += list(errors[1].reshape(-1, ))
            pred_dict['df'] += list(np.split(errors[2].reshape(-1, 3), partition)[:-1])
            pred_dict['total_na'].append(errors[3])
        if self.test_metrics is not None:
            self.test_metrics.regs = loss_tup[2].numpy().reshape(-1, 1).astype(np.float64)
        # return pd.DataFrame(pred_dict)
        return pred_dict

    def get_fitting_data(self):
        return self.data_df

    # def get_reg_components(self):
    #     regs = tf.convert_to_tensor(self.tensorpot.reg_components).numpy().reshape(-1, 1).astype(np.float64)
    #     # return [float(regs[i]) for i in range(regs.shape[0])]
    #     return regs

    def get_reg_weights(self):
        loss_specs = self.tensorpot.loss_specs
        l1 = np.array(loss_specs[SPEC_LOSS_L1_REG_FACTOR]).reshape(-1, 1)
        l2 = np.array(loss_specs[SPEC_LOSS_L2_REG_FACTOR]).reshape(-1, 1)
        aux = np.array(loss_specs[SPEC_AUX_LOSS_FACTORS]).reshape(-1, 1)

        return np.vstack([l1, l2, aux]) * self.n_batches

    def save_coefs(self):
        np.savetxt('Fit_coefficients_step_{0}'.format(self.eval_count), self.fit_coefs)

    def get_fitted_coefficients(self):
        if self.eval_count > 0:
            return self.fit_coefs
        else:
            raise AttributeError('Coefficients have not been fitted yet')

    def print_detailed_metrics(self, title='Iteration:'):
        print_detailed_metrics(self.last_fit_metric_data, title=title)

    def print_extended_metrics(self, title="FIT_STATS"):
        print_extended_metrics(self.last_fit_metric_data, title=title)

    def compute_test_metric(self):
        self.tensorpot.potential.get_updated_config(prefix=self.iter_num)
        if self.test_df is not None:
            # 'energy_pred', 'forces_pred', 'stress_pred',
            # 'de', 'de_pa', 'df', 'total_na'
            test_pred = self.predict(datadf=self.test_df)
            #self.test_metrics.regs = self.get_reg_components()
            self.test_metrics.reg_weights = self.get_reg_weights()
            self.test_metrics.compute_metrics(np.vstack(test_pred['de']),
                                              np.vstack(test_pred['de_pa']),
                                              np.vstack(test_pred['df']),
                                              np.vstack(test_pred['total_na']),
                                              dataframe=self.test_df)
            # self.test_metrics.loss = tot_loss
            # self.test_metrics.eval_time = self.eval_time
            curr_test_metrics_data = self.test_metrics.to_FitMetricsDict()
            curr_test_metrics_data["loss"] = curr_test_metrics_data["e_loss_contrib"] + \
                                             curr_test_metrics_data["f_loss_contrib"] + \
                                             curr_test_metrics_data["l1_reg_contrib"] + \
                                             curr_test_metrics_data["l2_reg_contrib"] + \
                                             sum(curr_test_metrics_data["extra_regularization_contrib"])
            curr_test_metrics_data["iter_num"] = self.iter_num
            return curr_test_metrics_data


class FitMetrics:
    def __init__(self, w_e, w_f, e_scale, f_scale, ncoefs, regs=None):
        self.w_e = w_e
        self.w_f = w_f
        self.e_scale = e_scale
        self.f_scale = f_scale
        self.regs = regs
        self.ncoefs = ncoefs
        self.nfuncs = None
        self.time_history = []

        self.loss = 0
        self.eval_time = 0

    def record_time(self, time):
        self.time_history.append(time)

    def to_FitMetricsDict(self):
        """
        Store all metric-relevant info into a dictionary
        :return: fit metrics dictionary
        """

        regularization_loss = [float(r_comp * r_weight) for r_comp, r_weight in zip(self.regs, self.reg_weights)]
        l1 = regularization_loss[0]
        l2 = regularization_loss[1]
        smoothness_reg_loss = regularization_loss[2:]
        res_dict = {
            # total loss
            "loss": self.loss,

            # loss contributions
            "e_loss_contrib": self.e_loss * self.e_scale,
            "f_loss_contrib": self.f_loss * self.f_scale,
            "l1_reg_contrib": l1,
            "l2_reg_contrib": l2,
            "extra_regularization_contrib": smoothness_reg_loss,

            # non-weighted e and f losses
            "e_loss": self.e_loss,
            "f_loss": self.f_loss,

            # e and f loss weights (scales)
            "e_scale": self.e_scale,
            "f_scale": self.f_scale,

            # RMSE metrics
            "rmse_epa": self.rmse_epa,
            "low_rmse_epa": self.low_rmse_epa,
            "rmse_f": self.rmse_f,
            "low_rmse_f": self.low_rmse_f,
            "rmse_f_comp": self.rmse_f_comp,
            "low_rmse_f_comp": self.low_rmse_f_comp,

            # MAE metrics
            "mae_epa": self.mae_epa,
            "low_mae_epa": self.low_mae_epa,
            "mae_f": self.mae_f,
            "low_mae_f": self.low_mae_f,
            "mae_f_comp": self.mae_f_comp,
            "low_mae_f_comp": self.low_mae_f_comp,

            # MAX metrics
            "max_abs_epa": self.max_abs_epa,
            "low_max_abs_epa": self.low_max_abs_epa,
            "max_abs_f": self.max_abs_f,
            "low_max_abs_f": self.low_max_abs_f,

            "eval_time": self.eval_time,
            "nat": self.nat,
            "ncoefs": self.ncoefs
        }

        if self.nfuncs is not None:
            res_dict["nfuncs"] = self.nfuncs

        return res_dict

    def from_FitMetricsDict(self, fit_metrics_dict):
        self.loss = fit_metrics_dict["loss"]

        self.e_loss = fit_metrics_dict["e_loss"]
        self.f_loss = fit_metrics_dict["f_loss"]

        self.e_scale = fit_metrics_dict["e_scale"]
        self.f_scale = fit_metrics_dict["f_scale"]

        # RMSE metrics
        self.rmse_epa = fit_metrics_dict["rmse_epa"]
        self.low_rmse_epa = fit_metrics_dict["low_rmse_epa"]
        self.rmse_f = fit_metrics_dict["rmse_f"]
        self.low_rmse_f = fit_metrics_dict["low_rmse_f"]

        self.rmse_f_comp = fit_metrics_dict["rmse_f_comp"]
        self.low_rmse_f_comp = fit_metrics_dict["low_rmse_f_comp"]

        # MAE metrics
        self.mae_epa = fit_metrics_dict["mae_epa"]
        self.low_mae_epa = fit_metrics_dict["low_mae_epa"]
        self.mae_f = fit_metrics_dict["mae_f"]
        self.low_mae_f = fit_metrics_dict["low_mae_f"]
        self.mae_f_comp = fit_metrics_dict["mae_f_comp"]
        self.low_mae_f_comp = fit_metrics_dict["low_mae_f_comp"]

        # MAX metrics
        self.max_abs_epa = fit_metrics_dict["max_abs_epa"]
        self.low_max_abs_epa = fit_metrics_dict["low_max_abs_epa"]
        self.max_abs_f = fit_metrics_dict["max_abs_f"]
        self.low_max_abs_f = fit_metrics_dict["low_max_abs_f"]

        self.eval_time = fit_metrics_dict["eval_time"]
        self.nat = fit_metrics_dict["nat"]
        self.ncoefs = fit_metrics_dict["ncoefs"]

        if "nfuncs" in fit_metrics_dict:
            self.nfuncs = fit_metrics_dict["nfuncs"]

    def compute_metrics(self, de, de_pa, df, nat, dataframe=None, de_low=None):
        if de_low is None:
            de_low = 1.
        self.nat = np.sum(nat)
        self.rmse_epa = np.sqrt(np.mean(de_pa ** 2))
        self.rmse_e = np.sqrt(np.mean(de ** 2))
        self.rmse_f = np.sqrt(np.mean(np.sum(df ** 2, axis=1)))
        self.rmse_f_comp = np.sqrt(np.mean(df ** 2))  # per component
        self.mae_epa = np.mean(np.abs(de_pa))
        self.mae_e = np.mean(np.abs(de))
        self.mae_f = np.mean(np.linalg.norm(df, axis=1))
        self.mae_f_comp = np.mean(np.abs(df).flatten())  # per component
        self.mae_f = np.mean(np.sum(np.abs(df), axis=1))
        # self.mae_f = np.mean(np.linalg.norm(df, axis=1))

        self.e_loss = float(np.sum(self.w_e * de_pa ** 2))
        self.f_loss = np.sum(self.w_f * df ** 2)
        self.max_abs_e = np.max(np.abs(de))
        self.max_abs_epa = np.max(np.abs(de_pa))
        self.max_abs_f = np.max(np.abs(df))

        self.low_rmse_epa = 0
        self.low_mae_epa = 0
        self.low_max_abs_epa = 0
        self.low_rmse_f = 0
        self.low_mae_f = 0
        self.low_max_abs_f = 0
        self.low_rmse_f_comp = 0
        self.low_mae_f_comp = 0

        if dataframe is not None:
            try:
                if "e_chull_dist_per_atom" in dataframe.columns:
                    nrgs = dataframe["e_chull_dist_per_atom"].to_numpy().reshape(-1, )
                    mask = nrgs <= de_low
                else:
                    nrgs = dataframe['energy_corrected'].to_numpy().reshape(-1, ) / nat.reshape(-1, )
                    emin = min(nrgs)
                    mask = (nrgs <= (emin + de_low))
                mask_f = np.repeat(mask, nat.reshape(-1, ))
                self.low_rmse_epa = np.sqrt(np.mean(de_pa[mask] ** 2))
                self.low_mae_epa = np.mean(np.abs(de_pa[mask]))
                self.low_max_abs_epa = np.max(np.abs(de_pa[mask]))
                self.low_rmse_f = np.sqrt(np.mean(np.sum(df[mask_f] ** 2, axis=1)))
                self.low_mae_f = np.mean(np.linalg.norm(df[mask_f], axis=1))
                self.low_max_abs_f = np.max(np.abs(df[mask_f]))
                self.low_rmse_f_comp = np.sqrt(np.mean(df[mask_f] ** 2))  # per component
                self.low_mae_f_comp = np.mean(np.abs(df[mask_f]).flatten())  # per component
            except:
                pass


class LMfunc:
    def __init__(self, tensorpot_fit, w_e):
        self.tensorpot_fit = tensorpot_fit
        self.resid_value = None
        self.jacob_value = None
        self.x_current = None
        self.w_e = w_e

    def run_fit(self, x, batches):
        tot_res = []
        tot_jac = []
        de, de_pa, df = ([] for _ in range(3))
        total_na = []
        t0 = time.perf_counter()
        for index, b in enumerate(batches):
            loss, jac, e, f = self.tensorpot_fit.tensorpot.external_vector_fit(x, b)
            nregs = len(self.tensorpot_fit.get_reg_components())
            if (index + 1) < len(batches):
                tot_jac.append(jac.numpy().reshape(-1, len(x))[:-nregs])
                tot_res.append(loss.numpy().reshape(-1, 1)[:-nregs])
            else:
                tot_jac.append(jac.numpy().reshape(-1, len(x)))
                tot_res.append(loss.numpy().reshape(-1, 1))

            errors = self.tensorpot_fit.compute_batch_errors(b, e.numpy(), f.numpy())
            de.append(errors[0])
            de_pa.append(errors[1])
            df.append(errors[2])
            total_na.append(errors[3])
        tot_res = np.vstack(tot_res).reshape(-1, )
        tot_jac = np.vstack(tot_jac)
        tot_res[-nregs:] *= len(batches)

        scalar_loss = np.sum(self.w_e.reshape(-1, 1) * (tot_res[:-nregs].reshape(-1, 1) / self.w_e.reshape(-1, 1)) ** 2) \
                      + np.sum(tot_res[-nregs:])
        self.tensorpot_fit.fit_coefs = x
        self.tensorpot_fit.eval_count += 1
        self.tensorpot_fit.eval_time = time.perf_counter() - t0
        self.tensorpot_fit.metrics.regs = self.tensorpot_fit.get_reg_components()
        self.tensorpot_fit.metrics.compute_metrics(np.vstack(de), np.vstack(de_pa), np.vstack(df),
                                                   np.vstack(total_na), dataframe=self.tensorpot_fit.data_df)

        if self.tensorpot_fit.iter_num == 0:
            log.info('{:<32}'.format('Initial state:') + '{:>10}'.format('Loss: ') + "{loss: >3.6f}" \
                     .format(loss=scalar_loss) +
                     '{str:>21}{rmse_epa:>.2f} ({low_rmse_e:>.2f}) meV/at' \
                     .format(str=" | RMSE Energy(low): ",
                             rmse_epa=1e3 * float(self.tensorpot_fit.metrics.rmse_epa),
                             low_rmse_e=1e3 * self.tensorpot_fit.metrics.low_rmse_epa))
            self.tensorpot_fit.metrics.print_extended_metrics(self.tensorpot_fit.iter_num, scalar_loss,
                                                              self.tensorpot_fit.get_reg_components(),
                                                              self.tensorpot_fit.get_reg_weights())
            self.tensorpot_fit.iter_num += 1

        self.tensorpot_fit.loss_history.append(scalar_loss)

        return tot_res, tot_jac

    def update(self, x, batches):
        # self.resid_value, self.jacob_value = self.run_fit(x, batches)
        resid_value, jacob_value = self.run_fit(x, batches)
        self.resid_value = resid_value.copy()
        self.jacob_value = jacob_value.copy()
        self.x_current = x.copy()

    def func(self, x, batches):
        self.update(x, batches)
        # self.x_current = x
        self.tensorpot_fit.callback(x)
        return self.resid_value

    def jac(self, x, batches):
        if (x != self.x_current).any():
            # print(x != self.x_current, 'JAC CALLED FOR UPDATE')
            self.update(x, batches)
        return self.jacob_value


class DAfunc:
    def __init__(self, tensorpot_fit):
        self.tensorpot_fit = tensorpot_fit
        self.loss_value = None
        self.jacob_value = None
        self.x_current = None
        self.batches = None

    def run_fit(self, x, batches):
        tot_loss = 0
        tot_jac = 0
        de, de_pa, df = ([] for _ in range(3))
        total_na = []
        t0 = time.perf_counter()
        for b in batches:
            loss, jac, e, f = self.tensorpot_fit.tensorpot.external_fit(x, b, eager=self.tensorpot_fit.eager)
            tot_jac += jac.numpy()
            tot_loss += loss.numpy()
            errors = self.tensorpot_fit.compute_batch_errors(b, e.numpy(), f.numpy())
            de.append(errors[0])
            de_pa.append(errors[1])
            df.append(errors[2])
            total_na.append(errors[3])

        self.tensorpot_fit.fit_coefs = x
        self.tensorpot_fit.eval_count += 1
        self.tensorpot_fit.eval_time = time.perf_counter() - t0
        self.tensorpot_fit.metrics.regs = self.tensorpot_fit.get_reg_components()
        self.tensorpot_fit.metrics.compute_metrics(np.vstack(de), np.vstack(de_pa), np.vstack(df),
                                                   np.vstack(total_na), dataframe=self.tensorpot_fit.data_df)
        if self.tensorpot_fit.iter_num == 0:
            log.info('{:<32}'.format('Initial state:') + '{:>10}'.format('Loss: ') + "{loss: >3.6f}" \
                     .format(loss=tot_loss) +
                     '{str:>21}{rmse_epa:>.2f} ({low_rmse_e:>.2f}) meV/at' \
                     .format(str=" | RMSE Energy(low): ",
                             rmse_epa=1e3 * float(self.tensorpot_fit.metrics.rmse_epa),
                             low_rmse_e=1e3 * self.tensorpot_fit.metrics.low_rmse_epa))
            self.tensorpot_fit.metrics.print_extended_metrics(self.tensorpot_fit.iter_num, tot_loss,
                                                              self.tensorpot_fit.get_reg_components(),
                                                              self.tensorpot_fit.get_reg_weights())
            self.tensorpot_fit.iter_num += 1

        self.tensorpot_fit.loss_history.append(tot_loss)
        self.last_loss = self.tensorpot_fit.loss_history[-1]

        if self.tensorpot_fit.jacobian_factor is not None:
            try:
                assert len(self.tensorpot_fit.jacobian_factor) == tot_jac.shape[0]
                tot_jac *= self.tensorpot_fit.tensorpot_fit.jacobian_factor
            except:
                raise ValueError(
                    'Size of the provided Jacobian factor ({0}) is not compatible with Jacobian size ({1}).' \
                        .format(len(self.tensorpot_fit.jacobian_factor), tot_jac.shape[0]))

        return tot_loss, tot_jac

    def update(self, x, batches):
        # self.resid_value, self.jacob_value = self.run_fit(x, batches)
        loss_value, jacob_value = self.run_fit(x, batches)
        self.loss_value = loss_value.copy()
        self.jacob_value = jacob_value.copy()
        self.x_current = x.copy()

    def func(self, x, batches):
        self.batches = batches
        self.update(x, batches)
        # self.x_current = x
        self.tensorpot_fit.callback(x)
        return self.loss_value

    # def jac(self, x, batches):
    def jac(self, x):
        if (x != self.x_current).any():
            # print(x != self.x_current, 'JAC CALLED FOR UPDATE')
            self.update(x, self.batches)
        return self.jacob_value


def print_extended_metrics(fit_metrics_dict, title="FIT_STATS"):
    # (self, iter_num, total_loss, reg_comps, reg_weights, title='FIT STATS', nfuncs=None):

    iter_num = fit_metrics_dict["iter_num"]
    total_loss = fit_metrics_dict["loss"]

    str0 = '\n' + '-' * 44 + title + '-' * 44 + '\n'
    str1 = '{prefix:<11} #{iter_num:<4}'.format(prefix='Iteration:', iter_num=iter_num)
    str1 += '{prefix:<8}'.format(prefix='Loss:')
    str1 += '{prefix:>8} {tot_loss:>1.4e} ({fr:3.0f}%) '.format(prefix='Total: ', tot_loss=total_loss, fr=100)
    str1 += '\n'

    fr = fit_metrics_dict["e_loss_contrib"] / total_loss * 100 if total_loss > 0 else 0
    str2 = '{prefix:>33} {e_loss:>1.4e} ({fr:3.0f}%) '.format(prefix='Energy: ',
                                                              e_loss=fit_metrics_dict["e_loss_contrib"],
                                                              fr=fr)
    str2 += '\n'

    fr = fit_metrics_dict["f_loss_contrib"] / total_loss * 100 if total_loss > 0 else 0
    str3 = '{prefix:>33} {f_loss:>1.4e} ({fr:3.0f}%) '.format(prefix='Force: ',
                                                              f_loss=fit_metrics_dict["f_loss_contrib"],
                                                              fr=fr)
    str3 += '\n'

    l1 = fit_metrics_dict["l1_reg_contrib"]
    l2 = fit_metrics_dict["l2_reg_contrib"]

    fr = l1 / total_loss * 100 if total_loss != 0 else 0
    str4 = '{prefix:>33} {l1:>1.4e} ({fr:3.0f}%) '.format(prefix='L1: ', l1=l1, fr=fr)
    str4 += '\n'
    fr = l2 / total_loss * 100 if total_loss != 0 else 0
    str4 += '{prefix:>33} {l2:>1.4e} ({fr:3.0f}%) '.format(prefix='L2: ', l2=l2, fr=fr)
    str4 += '\n'

    reg_comps = fit_metrics_dict["extra_regularization_contrib"]
    str5 = ''
    for i, comp in enumerate(reg_comps):
        str5 += '{prefix:>33} '.format(prefix='Smooth_w{}: '.format(i + 1))
        str5 += '{s1:>1.4e} '.format(s1=comp)
        str5 += '({fr:3.0f}%) '.format(fr=comp / total_loss * 100)
        str5 += '\n'

    nfuncs = fit_metrics_dict.get('nfuncs')
    ncoefs = fit_metrics_dict.get('ncoefs')

    if nfuncs is None:
        line = 'Number of params.: '
    else:
        line = 'Number of params./funcs: '
    str6 = '{prefix:>20}'.format(prefix=line) + '{ncoefs:>6d}'.format(ncoefs=ncoefs)
    if nfuncs is not None:
        str6 += '/{nfuncs:<6d}'.format(nfuncs=nfuncs)

    avg_t = fit_metrics_dict["eval_time"] / fit_metrics_dict["nat"]  # in sec/atom
    str6 += '{prefix:>42}'.format(prefix='Avg. time: ') + \
            '{avg_t:>10.2f} {un:<6}'.format(avg_t=avg_t * 1e6, un='mcs/at')

    str6 += '\n' + '-' * 97 + '\n'
    str_loss = str0 + str1 + str2 + str3 + str4 + str5 + str6
    ##############################
    er_str_h = '{:>9}'.format('') + \
               '{:^22}'.format('Energy/at, meV/at') + \
               '{:^22}'.format('Energy_low/at, meV/at') + \
               '{:^22}'.format('Force, meV/A') + \
               '{:^22}\n'.format('Force_low, meV/A')

    er_rmse = '{prefix:>9} '.format(prefix='RMSE: ')
    er_rmse += '{:>14.2f}'.format(fit_metrics_dict["rmse_epa"] * 1e3) + \
               '{:>21.2f}'.format(fit_metrics_dict["low_rmse_epa"] * 1e3) + \
               '{:>21.2f}'.format(fit_metrics_dict["rmse_f_comp"] * 1e3) + \
               '{:>24.2f}\n'.format(fit_metrics_dict["low_rmse_f_comp"] * 1e3)
    er_mae = '{prefix:>9} '.format(prefix='MAE: ')
    er_mae += '{:>14.2f}'.format(fit_metrics_dict["mae_epa"] * 1e3) + \
              '{:>21.2f}'.format(fit_metrics_dict["low_mae_epa"] * 1e3) + \
              '{:>21.2f}'.format(fit_metrics_dict["mae_f_comp"] * 1e3) + \
              '{:>24.2f}\n'.format(fit_metrics_dict["low_mae_f_comp"] * 1e3)
    er_max = '{prefix:>9} '.format(prefix='MAX_AE: ')
    er_max += '{:>14.2f}'.format(fit_metrics_dict["max_abs_epa"] * 1e3) + \
              '{:>21.2f}'.format(fit_metrics_dict["low_max_abs_epa"] * 1e3) + \
              '{:>21.2f}'.format(fit_metrics_dict["max_abs_f"] * 1e3) + \
              '{:>24.2f}\n'.format(fit_metrics_dict["low_max_abs_f"] * 1e3)
    er_str = er_str_h + er_rmse + er_mae + er_max + '-' * 97  # + '\n'
    log.info(str_loss + er_str)


def print_detailed_metrics(fit_metrics_dict, title='Iteration:'):
    # fit_metrics_dict
    iter_num = fit_metrics_dict["iter_num"]
    total_loss = fit_metrics_dict["loss"]
    avg_t = fit_metrics_dict["eval_time"] / fit_metrics_dict["nat"]  # in sec/atom
    log.info('{:<12}'.format(title) +
             "#{iter_num:<5}".format(iter_num=iter_num) +
             '{:<14}'.format('({numeval} evals):'.format(numeval=fit_metrics_dict["eval_count"])) +
             '{:>10}'.format('Loss: ') + "{loss: >3.6f}".format(loss=total_loss) +
             '{str1:>21}{rmse_epa:>.2f} ({low_rmse_e:>.2f}) meV/at' \
             .format(str1=" | RMSE Energy(low): ",
                     rmse_epa=1e3 * fit_metrics_dict["rmse_epa"],
                     low_rmse_e=1e3 * fit_metrics_dict["low_rmse_epa"]) +
             '{str3:>16}{rmse_f:>.2f} ({low_rmse_f:>.2f}) meV/A' \
             .format(str3=" | Forces(low): ",
                     rmse_f=1e3 * fit_metrics_dict["rmse_f_comp"],
                     low_rmse_f=1e3 * fit_metrics_dict["low_rmse_f_comp"]) +
             ' | Time/eval: {:>6.2f} mcs/at'.format(avg_t * 1e6))


def save_optimizer_state(optimizer, save_path, save_name):
    '''
    Save keras.optimizers object state.

    Arguments:
    optimizer --- Optimizer object.
    save_path --- Path to save location.
    save_name --- Name of the .npy file to be created.

    '''

    # Create folder if it does not exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # save weights
    np.save(os.path.join(save_path, save_name), optimizer.get_weights())

    return


def load_optimizer_state(optimizer, load_path, load_name, model_train_vars):
    '''
    Loads keras.optimizers object state.

    Arguments:
    optimizer --- Optimizer object to be loaded.
    load_path --- Path to save location.
    load_name --- Name of the .npy file to be read.
    model_train_vars --- List of model variables (obtained using Model.trainable_variables)

    '''

    # Load optimizer weights
    opt_weights = np.load(os.path.join(load_path, load_name) + '.npy', allow_pickle=True)

    # dummy zero gradients
    zero_grads = [tf.zeros_like(w) for w in model_train_vars]
    # save current state of variables
    saved_vars = [tf.identity(w) for w in model_train_vars]

    # Apply gradients which don't do nothing with Adam
    optimizer.apply_gradients(zip(zero_grads, model_train_vars))

    # Reload variables
    [x.assign(y) for x, y in zip(model_train_vars, saved_vars)]

    # Set the weights of the optimizer
    optimizer.set_weights(opt_weights)

    return
