# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

This project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Implement higher-order RK schemes and adaptive step size control during load path generation. Increasing the step size as much as possible without loosing accuracy would benefit a subsequent gradient-based optimization procedure as the total number of points and hence the number of design variables would be lower.
- Implement parallel load path computation.
- Implement automatic seed point generation.
- Constrain load paths to surfaces provided by the user (with constant offset from the outer surface, for example) to facilitate the 3D printing process.
- Implement fiber path optimization based on the topology optimization algorithm described in [1].
- Implement gradient-based fiber path optimization.

### Changed

- Write log files instead of printing to the console.

### Deprecated

-

### Removed

-

### Fixed

-

### Security

-

## [0.1.0] - 2025-07-21

### Added

- Implement load path plotter based on [1].
- Add support for linear and quadratic tetrahedral and hexahedral elements (SOLID185, SOLID186, SOLID187, and SOLID285 in Ansys). Note that quadratic elements yield more accurate FEA results but stress values are interpolated linearly because Ansys does not provide accurate stress results at the mid-side nodes.
- Add function to export paths in SpaceClaim or CSV format.
