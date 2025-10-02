# ACE+Q

This repository contains TensorFlow implementations of the **ACE+Q** theory, as described in the paper [Charge-constrained atomic cluster expansion](https://journals.aps.org/prmaterials/abstract/10.1103/PhysRevMaterials.9.033802) by Rinaldi et al.


## Overview 

The ACE+Q framework allows to train Atomic Cluster Expansion (ACE) models with charge equilibration (Qeq). The repository contains the potential files designed for non periodic (`qace.py`) and periodic (`qace_pbc.py`) systems, with the periodic version employing **Ewald summation**. 

These implementations were developed as part of [pacemaker](https://github.com/ICAMS/python-ace), a package for training Atomic Cluster Expansion models.
