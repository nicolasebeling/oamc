# OAMC

This project aims to facilitate the _optimal additive manufacturing of (continuous-fiber) composites_ (OAMC) by providing various path planning algorithms. These algorithms utilize FEA results and therefore take loads and constraints into account. In addition, the user will be able to specify optimization objectives such as weight or compliance.

Part of the project (`oamc.lpp`) is based on Garth Pearce's [load path plotter](https://github.com/GarthPearce/LoadPathMATLAB/) but faster and more flexible (with support for tetrahedral and quadratic elements, for example). The generated load paths may be used directly for 3D printing or to initialize a gradient-based optimization algorithm (yet to be implemented).

This project is part of my BSc thesis at TUM.

## Installation

See [INSTALL.md](INSTALL.md)

## Changes

See [CHANGELOG.md](CHANGELOG.md)

## Notation

The term "pointing stress vector" is defined in [1].

Stresses are stored using the same Voigt-style notation as Ansys: `[X, Y, Z, XY, YZ, ZX]`

## References

[1] D. Kelly, C. Reidsema, A. Bassandeh, G. Pearce, and M. Lee, “On interpreting load paths and identifying a load bearing topology from finite element analysis,” Finite Elements in Analysis and Design, vol. 47, no. 8, pp. 867–876, Aug. 2011, doi: https://doi.org/10.1016/j.finel.2011.03.007.
