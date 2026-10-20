/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "TensorBoundaryCondition.h"

/**
 * General Robin boundary condition: a*u + b*du/dn = c on the given face,
 * where n is the outward normal. Both the explicit ghost-fill path
 * (fillGhostCells) and the implicit matrix-row/RHS path (stencilContribution)
 * are derived from the same (a, b, c) triple, so the two cannot drift apart.
 */
class TensorRobinBC : public TensorBoundaryCondition
{
public:
  static InputParameters validParams();

  TensorRobinBC(const InputParameters & parameters);

  virtual void fillGhostCells(torch::Tensor & buffer,
                              unsigned int ghost_layers,
                              int64_t halo,
                              int64_t owned_extent,
                              Real dx) const override;

  virtual std::pair<std::vector<FDStencilEntry>, Real>
  stencilContribution(unsigned int ghost_layers, Real dx) const override;

  /**
   * Solve a*(u0 + ughost)/2 + b*(ughost - u0)/dx = c for ughost = coefficient*u0 +
   * constant, with dx > 0 the unsigned spacing. The ghost cell lies on the outward
   * side of either face, so the same expression holds on the lower and upper face.
   * Both fillGhostCellsStatic() and stencilContributionStatic() build from these.
   */
  static Real ghostCoefficientOnU0(Real a, Real b, Real dx);
  static Real ghostConstantTerm(Real a, Real b, Real c, Real dx);

  /// Object-independent core of fillGhostCells(), used by the instance method and the unit test.
  static void fillGhostCellsStatic(torch::Tensor & buffer,
                                   bool is_upper_face,
                                   int64_t face_dim,
                                   unsigned int ghost_layers,
                                   int64_t halo,
                                   int64_t owned_extent,
                                   Real a,
                                   Real b,
                                   Real c,
                                   Real dx);

  /// Object-independent core of stencilContribution(), used by the instance method and the unit test.
  static std::pair<std::vector<FDStencilEntry>, Real>
  stencilContributionStatic(unsigned int dim, Real a, Real b, Real c, Real dx);

protected:
  /// Robin coefficients: a*u + b*du/dn = c
  const Real _a;
  const Real _b;
  const Real _c;
};
