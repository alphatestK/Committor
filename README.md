# Committor

This repository contains all the input files and data related to the paper "Computing the Committor with the Committor: an Anatomy of the Transition State Ensemble" [https://arxiv.org/abs/2401.05279v2](https://arxiv.org/abs/2401.05279v2)

The following folders contains the simulation setup, iterative training data, and models for the different systems presented in the manuscript.
```
.
├── mueller
├── ala2
├── dasa
└── chignolin
```
And the plugin folder contains the cpp patch for the bias interface in plumed.

The definition and training of NN-based committor model is available through the [mlcolvar](https://github.com/luigibonati/mlcolvar/) library, a simple tutorial on Mueller-Brown potential could be found in [/docs/notebooks/tutorials/adv_committor.ipynb](https://github.com/luigibonati/mlcolvar/blob/main/docs/notebooks/tutorials/adv_committor.ipynb)
