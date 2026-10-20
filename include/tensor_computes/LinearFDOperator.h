/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "MooseTypes.h"
#include <vector>

/**
 * One term of a finite-difference stencil: a grid-point offset (one entry
 * per spatial dimension) and the coefficient it multiplies. Coefficients
 * already include any 1/dx^2-style grid-spacing scaling and operator
 * prefactor, matching exactly what the operator's matrix-free action
 * applies.
 */
struct FDStencilEntry
{
  std::vector<int> offset;
  Real coefficient;
};

/**
 * Mixin interface for TensorComputes whose action is a linear, known
 * finite-difference stencil (e.g. FiniteDifferenceLaplacian). This is NOT a
 * TensorOperatorBase subclass; it is multiply-inherited alongside
 * TensorOperator<T> by operators that can expose their stencil for uses
 * beyond matrix-free evaluation (e.g. assembling a preconditioner matrix).
 */
class LinearFDOperator
{
public:
  virtual ~LinearFDOperator() = default;

  /**
   * Returns this operator's finite-difference stencil as grid-point offsets
   * and coefficients, matching the coefficients used by the operator's own
   * matrix-free (e.g. convolution-based) action exactly.
   */
  virtual std::vector<FDStencilEntry> stencilAt() const = 0;
};
