/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "ImplicitFDSolver.h"
#include "TensorProblem.h"
#include "TensorBufferBase.h"
#include "DomainAction.h"
#include "MarlinUtils.h"

#include "libmesh/int_range.h"

#include <algorithm>
#include <cmath>
#include <limits>

registerMooseObject("MarlinApp", ImplicitFDSolver);

InputParameters
ImplicitFDSolver::validParams()
{
  InputParameters params = TensorSolver::validParams();
  params.addClassDescription(
      "Implicit solver for fields whose linear terms are finite-difference operators, solving "
      "sum_i A_i u = sum_i r_i + f for each field with a Krylov method.");

  params.addRequiredParam<std::vector<TensorOutputBufferName>>("buffer", "The solved fields");
  params.addRequiredParam<std::vector<std::vector<TensorComputeName>>>(
      "operators",
      "For each field (separated by ';'), the LinearFDOperator computes whose sum is the system "
      "operator");
  params.addParam<std::vector<TensorInputBufferName>>(
      "source", {}, "Explicit source term f for each field (empty for no sources)");

  MooseEnum linear_solver("AUTO PETSC TORCH_PCG", "AUTO");
  params.addParam<MooseEnum>("linear_solver",
                             linear_solver,
                             "Linear solver backend. AUTO selects TORCH_PCG on MPS and PETSC on "
                             "all other devices.");
  params.addRangeCheckedParam<Real>(
      "l_tol", 1e-8, "l_tol > 0", "Relative tolerance on the unpreconditioned residual norm");
  params.addRangeCheckedParam<unsigned int>(
      "l_max_its", 10000, "l_max_its > 0", "Maximum number of linear iterations");

  return params;
}

ImplicitFDSolver::ImplicitFDSolver(const InputParameters & parameters)
  : TensorSolver(parameters),
    IterativeTensorSolverInterface(),
    _l_tol(getParam<Real>("l_tol")),
    _l_max_its(getParam<unsigned int>("l_max_its"))
{
  // backend selection (an explicit choice, never a fallback)
  const auto device = MooseTensor::floatTensorOptions().device();
  const auto & linear_solver = getParam<MooseEnum>("linear_solver");
  if (linear_solver == "AUTO")
  {
    _linear_solver = device.is_mps() ? LinearSolver::TORCH_PCG : LinearSolver::PETSC;
    mooseInfo("Using the ",
              _linear_solver == LinearSolver::PETSC ? "PETSC" : "TORCH_PCG",
              " linear solver on device '",
              device.str(),
              "'.");
  }
  else
    _linear_solver = linear_solver == "PETSC" ? LinearSolver::PETSC : LinearSolver::TORCH_PCG;

  if (_linear_solver == LinearSolver::PETSC)
    paramError("linear_solver", "The PETSC backend is not implemented yet.");

  // the operators act on ghosted real space buffers
  if (!_domain.isRealSpaceMode())
    mooseError("ImplicitFDSolver requires Domain/parallel_mode = REAL_SPACE.");
  const auto & periodic = _domain.getPeriodicDirections();
  for (const auto d : make_range(_dim))
    if (!periodic[d])
      mooseError("ImplicitFDSolver currently supports periodic directions only, but direction ",
                 d,
                 " is not periodic.");

  const auto & buffers = getParam<std::vector<TensorOutputBufferName>>("buffer");
  const auto & operators = getParam<std::vector<std::vector<TensorComputeName>>>("operators");
  const auto & sources = getParam<std::vector<TensorInputBufferName>>("source");
  if (operators.size() != buffers.size())
    paramError("operators", "Must have one ';'-separated list of operators per entry in 'buffer'.");
  if (!sources.empty() && sources.size() != buffers.size())
    paramError("source", "Must be empty or have one entry per entry in 'buffer'.");

  _fields.reserve(buffers.size());
  for (const auto i : make_range(buffers.size()))
  {
    Field field{buffers[i],
                getOutputBufferByName(buffers[i]),
                _tensor_problem.getBufferBase(buffers[i]),
                {},
                {},
                {},
                sources.empty() ? nullptr : &getBufferBase(sources[i]),
                false};

    if (operators[i].empty())
      paramError("operators", "No operators given for field '", buffers[i], "'.");

    // sum_i A_i is SPD if one term is SPD and the others are SPD or SPSD; it is
    // SPSD with the constants as nullspace if all terms are SPSD
    bool definite = false;
    for (const auto & op_name : operators[i])
    {
      auto & op = _tensor_problem.getCompute(op_name);
      const auto * fd_op = dynamic_cast<const LinearFDOperator *>(&op);
      if (!fd_op)
        paramError("operators", "Compute '", op_name, "' is not a LinearFDOperator.");
      if (fd_op->inputBufferName() != buffers[i])
        paramError("operators",
                   "Operator '",
                   op_name,
                   "' acts on '",
                   fd_op->inputBufferName(),
                   "' rather than on the solved field '",
                   buffers[i],
                   "'.");

      switch (fd_op->definiteness())
      {
        case LinearFDOperator::Definiteness::POSITIVE_DEFINITE:
          definite = true;
          break;
        case LinearFDOperator::Definiteness::POSITIVE_SEMIDEFINITE:
          break;
        case LinearFDOperator::Definiteness::OTHER:
          paramError("operators",
                     "Operator '",
                     op_name,
                     "' is not symmetric positive (semi)definite, which the conjugate gradient "
                     "solve requires. For -D*Laplacian(u) use FiniteDifferenceLaplacian with "
                     "factor = -D.");
      }

      field._operators.push_back(&op);
      field._fd_operators.push_back(fd_op);
      field._operator_outputs.push_back(
          &_tensor_problem.getBufferBase(op.getParam<TensorOutputBufferName>("buffer")));
    }
    field._singular = !definite;

    _fields.push_back(std::move(field));
  }
}

void
ImplicitFDSolver::substep()
{
  // update the explicit terms (sources)
  _compute->realSpaceComputeBuffer();
  forwardBuffers();

  _iterations = 0;
  for (auto & field : _fields)
  {
    // u is used as Krylov scratch below and is written in place. Its storage may
    // be shared with its old states (advanceState() stores a shallow copy), so
    // give it its own storage first.
    field._u = field._u.clone();

    // b = sum_i r_i + f, evaluated while u still holds u_old
    torch::Tensor b = torch::zeros_like(field._u_base.ownedView()).contiguous();
    for (const auto * fd_op : field._fd_operators)
      if (const auto r = fd_op->rhsContribution(); r.defined())
        b += r;
    if (field._source)
      b += field._source->ownedView();

    if (field._singular)
    {
      // b must be in the range of the operator (zero mean), and only a round-off
      // mean is removed. Pairwise summation of N terms has the error bound
      // ~log2(N) eps sum|b| (Higham, Accuracy and Stability of Numerical
      // Algorithms, ch. 4); the factor 16 leaves headroom for the rounding in
      // evaluating b itself and for the cross-rank reduction.
      const auto n_cells = static_cast<Real>(_domain.getNumberOfCells());
      const Real eps = b.scalar_type() == torch::kFloat ? std::numeric_limits<float>::epsilon()
                                                        : std::numeric_limits<double>::epsilon();
      const Real b_sum = _domain.allreduceSum(b.sum()).cpu().item<double>();
      const Real b_abs_sum = _domain.allreduceSum(b.abs().sum()).cpu().item<double>();
      if (std::abs(b_sum) > 16.0 * std::log2(n_cells) * eps * b_abs_sum)
        mooseError("The operator of field '",
                   field._name,
                   "' is singular (its nullspace is the constants), so the right-hand side must "
                   "have zero mean, but its mean is ",
                   b_sum / n_cells,
                   ".");
      b -= b_sum / n_cells;
    }

    // warm start from u_old
    torch::Tensor x = field._u_base.ownedView().contiguous().clone();

    unsigned int iterations = 0;
    const Real b_norm = std::sqrt(_domain.allreduceSum((b * b).sum()).cpu().item<double>());
    if (b_norm == 0.0)
      // the zero-mean solution of A x = 0 is x = 0
      x.zero_();
    else
      iterations = solveTorchPCG(field, b, b_norm, x);

    if (field._singular)
      x -= _domain.allreduceSum(x.sum()) / static_cast<Real>(_domain.getNumberOfCells());

    // write the solution into u and update its ghost layers
    field._u_base.ownedView().copy_(x);
    _tensor_problem.exchangeGhostLayers(field._name, _tensor_problem.getMaxGhostLayer());

    _iterations += iterations;
  }
  _is_converged = true;
}

torch::Tensor
ImplicitFDSolver::applyOperator(Field & field, const torch::Tensor & x)
{
  field._u_base.ownedView().copy_(x);

  torch::Tensor y;
  for (const auto i : index_range(field._operators))
  {
    // ghost exchange of u followed by the operator's own kernel
    _tensor_problem.runComputeWithGhosts(*field._operators[i]);
    const auto output = field._operator_outputs[i]->ownedView();
    y = y.defined() ? y + output : output.clone();
  }
  return y;
}

Real
ImplicitFDSolver::diagonal(const Field & field) const
{
  Real diag = 0.0;
  for (const auto * fd_op : field._fd_operators)
    for (const auto & entry : fd_op->stencilAt())
      if (std::all_of(entry.offset.begin(), entry.offset.end(), [](int o) { return o == 0; }))
        diag += entry.coefficient;
  return diag;
}

unsigned int
ImplicitFDSolver::solveTorchPCG(Field & field,
                                const torch::Tensor & b,
                                Real b_norm,
                                torch::Tensor & x)
{
  // Jacobi preconditioner; with constant coefficients the diagonal is a scalar
  const Real inv_diag = 1.0 / diagonal(field);
  const auto A = [&](const torch::Tensor & v) { return applyOperator(field, v); };
  const auto M = [inv_diag](const torch::Tensor & r) { return r * inv_diag; };
  const auto reduce_sum = [this](const torch::Tensor & t) { return _domain.allreduceSum(t); };

  const auto [x_new, iterations, res_norm] =
      MooseTensor::conjugateGradientSolve(A, b, x, _l_tol, _l_max_its, M, reduce_sum);

  if (res_norm > _l_tol * b_norm)
    mooseError("The TORCH_PCG solve of field '",
               field._name,
               "' did not converge in ",
               _l_max_its,
               " iterations (relative residual ",
               res_norm / b_norm,
               ", l_tol = ",
               _l_tol,
               ").");

  x = x_new;
  return iterations;
}
