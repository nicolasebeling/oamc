# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

This project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Additions

- In the `oamc.lpp` subpackage:
  - Implement higher-order RK schemes and adaptive step size control during load path generation. Increasing the step size as much as possible without loosing accuracy would benefit a subsequent gradient-based optimization procedure as the total number of points and hence the number of design variables would be lower.
  - Implement parallel load path computation.
  - Implement automatic seed point generation.
  - Constrain load paths to surfaces provided by the user (with constant offset from the outer surface, for example) to facilitate the 3D printing process.
  - Implement fiber path optimization based on the topology optimization algorithm described in [1].

### Changes

- Update Python version

### Deprecations

-

### Removals

-

### Fixes

-

### Security

-

## [0.2.0] - 2025-09-XX

### Additions

- Logging
- Linear static FEA
- General-purpose optimizer
- Fiber trajectory optimization
- Unit tests

### Changes

- Restructure the project as follows:
  - `oamc.constants`, `oamc.enums`, and `oamc.logging` are modules containing constants, enums, and logging functionality shared across subpackages, respectively.
  - `oamc.fem` is an independent subpackage for linear static FEA.
  - `oamc.optimization` is an independent subpackage for numerical optimization.
  - `oam.lpp` is a subpackage containing the load path plotting algorithm based on [1] though the plotting functionality has been moved to `oamc.visualization`. It uses `oamc.fem` for stress computation.
  - `oamc.visualization` is a subpackage for visualizing meshes from `oamc.fem` and paths from `oamc.lpp`.
  - `oamc.x` is a subpackage for optimizing fiber trajectories for nonplanar FDM printing with continuous fiber reinforcement. It is based on `oamc.fem` and `oamc.optimization`. A proper name has not yet been decided.

## [0.1.0] - 2025-07-21

### Additions

- Implement load path plotter based on [1].
- Add support for linear and quadratic tetrahedral and hexahedral elements (SOLID185, SOLID186, SOLID187, and SOLID285 in Ansys). Note that quadratic elements yield more accurate FEA results but stress values are interpolated linearly because Ansys does not provide accurate stress results at the mid-side nodes.
- Add function to export paths in SpaceClaim or CSV format.
