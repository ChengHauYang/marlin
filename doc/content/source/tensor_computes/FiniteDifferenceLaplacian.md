# FiniteDifferenceLaplacian

!syntax description /TensorComputes/Solve/FiniteDifferenceLaplacian

## Overview

Computes `factor` times the Laplacian of the input buffer in real space, using
the second-order 3-point or the fourth-order 5-point central difference stencil
on each axis ([!param](/TensorComputes/Solve/FiniteDifferenceLaplacian/stencil_width)).
Ghost layers are exchanged before the stencil is applied.

The operator also exposes its stencil, which implicit solvers use to build
coarse multigrid levels. On a periodic grid the discrete Laplacian is
symmetric negative semidefinite with the constants as its nullspace, so with
`factor < 0` it is positive semidefinite.

## Example Input File Syntax

!listing test/tests/real_space/laplacian.i block=TensorComputes/Solve/lap_compute

!syntax parameters /TensorComputes/Solve/FiniteDifferenceLaplacian

!syntax inputs /TensorComputes/Solve/FiniteDifferenceLaplacian

!syntax children /TensorComputes/Solve/FiniteDifferenceLaplacian
