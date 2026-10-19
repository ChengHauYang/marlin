# TimeDerivativeOperator

!syntax description /TensorComputes/Solve/TimeDerivativeOperator

## Overview

The backward Euler time derivative term for implicit finite-difference solves.
Its action on the solved field $u$ is $u/\Delta t$, and it contributes
$u^n/\Delta t$ to the right-hand side, with $\Delta t$ the current substep size.
Listed together with the spatial operators of a field, an implicit solver
then solves

!equation
\frac{u - u^n}{\Delta t} + \sum_i L_i u = f.

The operator is positive definite, and its stencil is the single centre entry
$1/\Delta t$ at every grid spacing.

!syntax parameters /TensorComputes/Solve/TimeDerivativeOperator

!syntax inputs /TensorComputes/Solve/TimeDerivativeOperator

!syntax children /TensorComputes/Solve/TimeDerivativeOperator
