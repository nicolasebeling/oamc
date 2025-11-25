# OAMC

This project aims to facilitate the _optimal additive manufacturing of (continuous-fiber) composites_ (OAMC) by providing various path planning algorithms. These algorithms utilize FEA results and therefore take loads and constraints into account. In addition, the user will be able to specify optimization objectives such as weight or compliance.

One subpackage (`oamc.lpp`) is based on Garth Pearce's [load path plotter](https://github.com/GarthPearce/LoadPathMATLAB/) but faster and more flexible (with support for tetrahedral and quadratic elements, for example). The generated load paths may be used directly for 3D printing or to initialize a gradient-based optimization algorithm (yet to be implemented).

This project is part of my BSc thesis at TUM.

## Current Limitations

FEA and hence gradient-based optimization are only available for single-body parts. The load path plotter also works with multi-body parts if the stress values at the nodes are pre-computed and exported from Ansys, for example.

## Installation

See [INSTALL.md](INSTALL.md)

## Changes

See [CHANGELOG.md](CHANGELOG.md)

## Conventions

Node and element indices are converted from 1-based to 0-based indexing upon import.

Strains and stresses are stored in standard Voigt notation `[X, Y, Z, YZ, XZ, XY]` and engineering shear strain convention (twice the tensorial shear strains to keep the strain energy density consistent between vector and tensor notations). Utility functions may offer multiple conventions, but engineering is always the default.

Units mm, N, t, s, K are assumed.

## References

[1] D. Kelly, C. Reidsema, A. Bassandeh, G. Pearce, and M. Lee, “On interpreting load paths and identifying a load bearing topology from finite element analysis,” Finite Elements in Analysis and Design, vol. 47, no. 8, pp. 867–876, Aug. 2011, doi: https://doi.org/10.1016/j.finel.2011.03.007.

[2] S. Wang, J. Liu, Z. He, and D. Yang, “Concurrent optimisation of structural topology and fibre paths for 3D printing of continuous fibre composites based on chain primitive projection,” Composites Part A: Applied Science and Manufacturing, vol. 185, p. 108333, Jun. 2024, doi: https://doi.org/10.1016/j.compositesa.2024.108333.

[3] H. Ren, D. Wang, G. Liu, D. W. Rosen, and Y. Xiong, “Concurrent optimization of structural topology and toolpath for additive manufacturing of continuous fiber-reinforced polymer composites,” Computer Methods in Applied Mechanics and Engineering, vol. 430, p. 117227, Oct. 2024, doi: https://doi.org/10.1016/j.cma.2024.117227.

[4] K. Svanberg, “The method of moving asymptotes—a new method for structural optimization,” Numerical Meth Engineering, vol. 24, no. 2, pp. 359–373, Feb. 1987, doi: https://doi.org/10.1002/nme.1620240207.

[5] C. Zillober, “A globally convergent version of the method of moving asymptotes,” Structural Optimization, vol. 6, no. 3, pp. 166–174, Sep. 1993, doi: https://doi.org/10.1007/BF01743509.

[6] K.-U. Bletzinger, “Extended method of moving asymptotes based on second-order information,” Structural Optimization, vol. 5, no. 3, pp. 175–183, Sep. 1993, doi: https://doi.org/10.1007/BF01743354.
