#!/usr/bin/env python

import argparse
from ruamel.yaml import YAML
import pandas as pd
import numpy as np

from ace_plus_q.graphspecs import *
from ace_plus_q.data import TPAtomsDataContainer
from ace_plus_q import TensorPotential
from ace_plus_q.potentials import QACE, QACE_PBC
from pyace import *
import sys
import json
import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

def parse_inline_dict(arg_value, name):
    """Parse inline JSON-style dictionary; raise error if invalid."""
    try:
        parsed = json.loads(arg_value)
        if not isinstance(parsed, dict):
            raise ValueError(f"{name} must be a JSON dictionary, not {type(parsed)}.")
        log.info(f"Using provided {name} values: {parsed}")
        return parsed
    except json.JSONDecodeError:
        raise ValueError(
            f"Invalid JSON format for {name}. "
            f"Example: '{{\"Au\": 1.1, \"Mg\": 2.1}}'"
        )

def is_periodic_atoms(atoms):
    # atoms.get_pbc() returns a length-3 boolean array
    return bool(np.any(atoms.get_pbc()))

def main(args=None):

    parser = argparse.ArgumentParser(description="Running fit for the ACE+Q model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "input",
        default='input.yaml',
        help="Location of the .yaml configuration file"
    )

    parser.add_argument(
        "-p", '--potential',
        default=None,
        help="Path to the starting potential configuration"
    )

    parser.add_argument(
        "-o", "--output",
        default='ace_plus_q',
        help="Saved model location"
    )

    parser.add_argument(
        "-ge", "--g_ewald",
        default=0.5,
        help="Parameter that controls the width of the Gaussian charges in the Ewald sum. It"
             "is related to the parameter eta in Eq. (A7) by the relation: eta = 1 / sqrt(2) /g_ewald"
    )

    parser.add_argument(
        "-k", "--kpoints",
        default=1,
        help="Number of k points used for the Ewald sum"
    )

    parser.add_argument(
        "--chi0",
        required=True,
        help="Inline JSON dictionary for χ₀ values, e.g. '{\"Au\": 1.1, \"Mg\": 2.1}'"
    )

    parser.add_argument(
        "--J0",
        required=True,
        help="Inline JSON dictionary for J₀ values, e.g. '{\"Au\": 2.0, \"O\": 3.5}'"
    )

    parser.add_argument(
        "--sigma",
        required=True,
        help="Inline JSON dictionary for σ values, e.g. '{\"Au\": 0.5, \"Mg\": 0.6}'"
    )

    args = parser.parse_args()

    def do_tpat(row, cutoff):
        ase_atom = row["ase_atoms"]
        energy_corrected = row["energy_corrected"]
        forces = row["forces"]
        total_charge = row['total_charge']
        atomic_charges = row['atomic_charges']
        # total_dipole_moment = row['total_dipole']

        # return TPAtomsDataContainer(ase_atom, energy=energy_corrected, forces=forces, total_chrg=total_charge,
        #                             atomic_chrg=atomic_charges, total_dpl_mom=total_dipole_moment, cutoff=cutoff)
        return TPAtomsDataContainer(ase_atom, energy=energy_corrected, forces=forces, total_chrg=total_charge,
                                    atomic_chrg=atomic_charges, cutoff=cutoff)

    with open(args.input, "r") as file:
        # try:
        yaml = YAML(typ='safe')
        potential_config = yaml.load(file)
        # except yaml.YAMLError as exc:
        #     print(exc)

    chi0_dict = parse_inline_dict(args.chi0, "chi0")
    J0_dict = parse_inline_dict(args.J0, "J0")
    sigma_dict = parse_inline_dict(args.sigma, "sigma")

    fit_config = potential_config["fit"]

    data_config = potential_config['data']
    dff = pd.read_pickle(data_config['data_path'], compression='gzip')[:2]

    dff['tp_atoms'] = dff.apply(do_tpat, args=(data_config['cutoff'],), axis=1)
    max_at = np.max(dff['ase_atoms'].apply(len).to_numpy())

    pbc_flags = dff['ase_atoms'].apply(is_periodic_atoms)
    n_periodic = int(pbc_flags.sum())
    n_total = len(pbc_flags)

    log.info(f"Detected {n_periodic}/{n_total} periodic structures in the dataset.")

    if n_periodic == n_total:
        dataset_periodic = True
        log.info("All structures are periodic -> using QACE_PBC.")
    elif n_periodic == 0:
        dataset_periodic = False
        log.info("All structures are non-periodic -> using QACE.")
    else:
        # Mixed dataset: decide policy. Here we raise; you can switch to majority.
        raise ValueError(
            f"Mixed periodicity detected: {n_periodic}/{n_total} periodic. "
            "Please provide a dataset with uniform periodicity or split it."
        )

    # assert 1 > data_config['test_fraction'] >= 0, ('Test fraction must be between 0 and 1')
    # train_frac = 1 - data_config['test_fraction']
    # df = dff.sample(frac=train_frac, random_state=322)
    # if data_config['test_fraction'] == 0:
    #     dft = None
    # else:
    #     dft = dff.drop(df.index)

    kpoints = int(args.kpoints) if args.kpoints is not None else 1
    g_ewald = float(args.g_ewald) if args.g_ewald is not None else 0.5

    if args.potential is None:
        bbasisconf = create_multispecies_basis_config(
            potential_config,
            func_coefs_initializer="zero",
        )
    else:
        bbasisconf = BBasisConfiguration(args.potential)

    if dataset_periodic:
        qace = QACE_PBC(potconfig=bbasisconf, n_e_atomic_prop=1, max_at=max_at, save_path='saves',
                    chi0_dict=chi0_dict, J0_dict=J0_dict, sigma_dict=sigma_dict, invert_matrix=True,
                        nkopints=kpoints, g_ewald=g_ewald,
                        required_optional_data_entries=[DATA_ATOMIC_CHRG, DATA_TOTAL_CHRG, DATA_TOTAL_DIPOLE_MOM])
    else:
        qace = QACE(potconfig=bbasisconf, n_e_atomic_prop=1, max_at=max_at, save_path='saves',
                    chi0_dict=chi0_dict, J0_dict=J0_dict, sigma_dict=sigma_dict, invert_matrix=True,
                    required_optional_data_entries=[DATA_ATOMIC_CHRG, DATA_TOTAL_CHRG, DATA_TOTAL_DIPOLE_MOM])

    # TODO: add here loss factor for total charge, total dipole moment and partial charges
    tp = TensorPotential(qace, mode='scf_train', compute_forces=True, compute_stress=False, eager_evaluate=True,
                         loss_specs = {
                        SPEC_LOSS_ENERGY_NORM_TYPE: 'per-atom',
                        SPEC_LOSS_ENERGY_FACTOR: 1.0,  #100.
                        SPEC_LOSS_FORCE_FACTOR: 100.0,  #100.
                        SPEC_LOSS_SCF_FACTOR: 1.0,
                    }, jit_compile=False)

    b_size = fit_config['batch_size']

    tp.train(
        dff,
        # test_df=dft,
        niter=1000000,
        batch_size=b_size,
        display_step=50,
        optimizer='L-BFGS-B',
        optimizer_options={'maxcor':1000}
    )


if __name__ == "__main__":
    main(sys.argv[1:])