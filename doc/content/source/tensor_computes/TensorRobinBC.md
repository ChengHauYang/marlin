# TensorRobinBC

!syntax description /TensorComputes/BCs/TensorRobinBC

## Overview

Imposes $a u + b \partial u / \partial n = c$ on one face of a non-periodic direction, with $n$
the outward normal. See the [BCs system](syntax/TensorComputes/BCs/index.md) for the
discretization. $a/2 + b/\Delta x$ must be nonzero, with $\Delta x$ the grid spacing normal to the
face. For $a < 0$ the ghost value amplifies the boundary-adjacent cell, and explicit time
integration diverges.

## Example Input File Syntax

!listing test/tests/real_space/robin_steady_state.i block=TensorComputes/BCs

!syntax parameters /TensorComputes/BCs/TensorRobinBC

!syntax inputs /TensorComputes/BCs/TensorRobinBC

!syntax children /TensorComputes/BCs/TensorRobinBC
