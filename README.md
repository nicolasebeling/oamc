# OAMC

This project aims to facilitate the _optimal additive manufacturing of (continuous-fiber) composites_ (OAMC) by providing various path planning algorithms, which take loads and constraints into account.

The subpackage `oamc.lpp` is based on Garth Pearce's [load path plotter](https://github.com/GarthPearce/LoadPathMATLAB/) but faster and more flexible; it supports tetrahedral and quadratic element types, for example. The generated load paths may be used to visualize and understand the transfer of loads through the structure, directly for 3D printing, or to initialize a gradient-based optimization algorithm.

This project is part of my bachelor's thesis _FEA-Driven Fiber Path Optimization for Nonplanar FDM Printing with Sparse Continuous Fiber Reinforcement_ at TUM.

## Current Limitations

FEA and hence gradient-based optimization are only available for single-body parts. The load path plotter also works with multi-body parts if the stress values at the nodes are pre-computed and imported from Ansys, for example.

## Installation

See [INSTALL.md](INSTALL.md)

## Changes

See [CHANGELOG.md](CHANGELOG.md)

## Conventions

Node and element indices are converted from 1-based to 0-based indexing upon import.

Strains and stresses are stored in standard Voigt notation `[X, Y, Z, YZ, XZ, XY]` and engineering shear strain convention (twice the tensorial shear strains to keep the strain energy density consistent between vector and tensor notations). Utility functions may offer multiple conventions, but engineering is always the default.
