# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

This project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Additions

- In the `oamc.lpp` subpackage:
  - Implement higher-order RK schemes and adaptive step size control during load path generation. Increasing the step size as much as possible without loosing accuracy would benefit a subsequent gradient-based optimization procedure as the total number of points and hence the number of design variables would be lower.
  - Parallelize load path computation.
  - Automatically generate seed points.
  - Constrain load paths to surfaces provided by the user (with constant offset from the outer surface, for example) to facilitate the 3D printing process.
  - Implement fiber path optimization based on the topology optimization algorithm described in the paper "On interpreting load paths and identifying a load bearing topology from finite element analysis."

### Changes

-

### Deprecations

-

### Removals

-

### Fixes

-

### Security

-

## [0.2.0] - 2025-12-19

### Additions

See note on restructuring below. Everything related to logging, FEA, fiber path optimization, and unit tests is new.

### Changes

- Update Python from 3.10 to 3.13.
- Switch to NumPy-style docstrings.
- Restructure the project as follows:
  - `oamc.constants`, `oamc.enums`, `oamc.logging` are top-level modules containing constants, enums, and logging functionality shared across subpackages, respectively.
  - `oamc.fiber` is a top-level module containing the `Fiber` class, which is used across subpackages too.
  - `oamc.fem` is an independent subpackage for linear static FEA that serves as the basis of most other subpackages.
  - `oam.lpp` is a subpackage containing functionality for the computation of load paths. The visualization has been moved to `oamc.post`. It depends on `oamc.fem` for stress computation.
  - `oamc.post` is a subpackage for visualizing meshes from `oamc.fem` and fiber paths generated with `oamc.lpp` and `oamc.core`. It depends on `oamc.fem` for displacement and stress visualization.
  - `oamc.core` is a subpackage for optimizing fiber trajectories for nonplanar FDM printing with continuous fiber reinforcement. It is based on `oamc.fem`.
- The `tests/` directory mirrors the `src/oamc/` directory. Currently, it contains the following unit tests:
  - `tests/test_fem/test_model` tests FEA results against results from Ansys Mechanical.

## [0.1.0] - 2025-07-21

### Additions

- Implement load path plotter.
- Add support for linear and quadratic tetrahedral and hexahedral elements (SOLID185, SOLID186, SOLID187, and SOLID285 in Ansys). Note that quadratic elements yield more accurate FEA results but stress values are interpolated linearly because Ansys does not provide accurate stress results at the mid-side nodes.
- Add function to export paths in SpaceClaim or CSV format.
