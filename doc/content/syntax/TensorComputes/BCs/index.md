# BCs System

This system is used to specify generic, non-LBM boundary conditions for the real-space
finite-difference solve path (as opposed to the LBM-specific lattice boundary conditions
under the [Boundary](syntax/TensorComputes/Boundary/index.md) system).

## Overview

Each object fills the ghost cells of one buffer on one face (`left`, `right`, `bottom`,
`top`, `back`, `front`) of a non-periodic axis, before any compute that reads the ghost
cells runs. All of them impose the cell-centred Robin condition

!equation
a u + b \frac{\partial u}{\partial n} = c

on the face, with $n$ the outward normal, $u$ approximated by the average of the
boundary-adjacent cell and its ghost, and $\partial u / \partial n$ by their difference
over the grid spacing. `TensorDirichletBC` ($a = 1$, $b = 0$) and `TensorNeumannBC`
($a = 0$, $b = 1$) are presets of `TensorRobinBC`. Every non-periodic face of a buffer
that is read with ghost cells needs a boundary condition.

## Example Input File Syntax

A Dirichlet condition on the left face and a Neumann condition on the right face of a
domain that is periodic in $y$:

!listing test/tests/real_space/neumann_linear.i block=Domain TensorComputes/BCs

!syntax list /TensorComputes/BCs objects=True actions=False subsystems=False

!syntax list /TensorComputes/BCs objects=False actions=False subsystems=True

!syntax list /TensorComputes/BCs objects=False actions=True subsystems=False
