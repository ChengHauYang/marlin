/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/
#pragma once

#include "TensorOperator.h"
#include "LinearFDOperator.h"

/**
 * Backward Euler time derivative term for implicit solves. The action is u/dt
 * and the right-hand side contribution is u^n/dt, so that together with the
 * spatial operators the solver solves (u - u^n)/dt + sum_i L_i u = f.
 */
class TimeDerivativeOperator : public TensorOperator<torch::Tensor>, public LinearFDOperator
{
public:
  static InputParameters validParams();

  TimeDerivativeOperator(const InputParameters & parameters);

  virtual void realSpaceComputeBuffer() override;

  virtual void computeBuffer() override;

  virtual std::vector<FDStencilEntry> stencilAt() const override;
  virtual std::vector<FDStencilEntry>
  stencilAt(const std::vector<Real> & grid_spacing) const override;
  virtual Definiteness definiteness() const override { return Definiteness::POSITIVE_DEFINITE; }
  virtual const std::string & inputBufferName() const override { return _input_name; }

  /// u^n/dt on the owned cells; call before the solve overwrites the input buffer
  virtual torch::Tensor rhsContribution() const override;

protected:
  /// input buffer name
  const TensorInputBufferName _input_name;
  /// input buffer
  const torch::Tensor & _u_in;
  /// input buffer wrapper (for the owned-cell view)
  TensorBufferBase & _u_in_base;
};
