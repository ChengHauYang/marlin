/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/
#pragma once

#include "TensorSolver.h"
#include "IterativeTensorSolverInterface.h"
#include "LinearFDOperator.h"

class TensorBufferBase;

/**
 * Implicit solver for fields whose linear terms are LinearFDOperator computes.
 * Each field u solves sum_i A_i u = sum_i r_i + f, where A_i is the action of
 * the i-th operator as written and r_i its right-hand side contribution (e.g.
 * u_old/dt for TimeDerivativeOperator). The fine-level operator action reuses
 * each operator's own computeBuffer(), so it is exactly the explicit kernel.
 */
class ImplicitFDSolver : public TensorSolver, public IterativeTensorSolverInterface
{
public:
  static InputParameters validParams();

  ImplicitFDSolver(const InputParameters & parameters);

protected:
  struct Field
  {
    /// name and buffer of the solved field (doubles as Krylov scratch during the solve)
    std::string _name;
    torch::Tensor & _u;
    TensorBufferBase & _u_base;
    /// the linear operators acting on u, and their output buffers
    std::vector<TensorOperatorBase *> _operators;
    std::vector<const LinearFDOperator *> _fd_operators;
    std::vector<TensorBufferBase *> _operator_outputs;
    /// explicit source term f (null if none)
    TensorBufferBase * _source;
    /// sum_i A_i has the constants as its nullspace
    bool _singular;
  };

  virtual void substep() override;

  /// y = sum_i A_i x on the owned cells (x is copied into u's owned cells first)
  torch::Tensor applyOperator(Field & field, const torch::Tensor & x);

  /// sum of the operators' centre stencil coefficients (the Jacobi diagonal)
  Real diagonal(const Field & field) const;

  /// solve sum_i A_i x = b (with ||b||_2 = b_norm) by torch preconditioned CG, warm started from x
  unsigned int
  solveTorchPCG(Field & field, const torch::Tensor & b, Real b_norm, torch::Tensor & x);

  enum class LinearSolver
  {
    PETSC,
    TORCH_PCG
  };

  /// selected linear solver backend
  LinearSolver _linear_solver;

  /// relative tolerance on the unpreconditioned residual 2-norm
  const Real _l_tol;

  /// maximum number of linear iterations
  const unsigned int _l_max_its;

  std::vector<Field> _fields;
};
