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
#include "ComputeGroup.h"
#include "TransientBase.h"
#include "TensorSolveIterationAdaptiveDT.h"

#include "PetscSupport.h"

#include "libmesh/int_range.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <sstream>

// the PETSc backend on CUDA needs PETSc and torch both built with CUDA
#if defined(__has_include)
#if __has_include(<ATen/cuda/CUDAConfig.h>)
#define MARLIN_TORCH_HAS_CUDA 1
#endif
#endif
#if PetscDefined(HAVE_CUDA) && defined(MARLIN_TORCH_HAS_CUDA)
#define MARLIN_PETSC_CUDA 1
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAEvent.h>
#include <petscdevice_cuda.h>
#else
#define MARLIN_PETSC_CUDA 0
#endif

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

  // PETSc options of the PETSC backend, applied with the prefix <buffer>_
  params += Moose::PetscSupport::getPetscValidParams();
  params.suppressParameter<MooseEnum>("solve_type");
  params.suppressParameter<MooseEnum>("mffd_type");

  return params;
}

ImplicitFDSolver::ImplicitFDSolver(const InputParameters & parameters)
  : TensorSolver(parameters),
    IterativeTensorSolverInterface(),
    _l_tol(getParam<Real>("l_tol")),
    _l_max_its(getParam<unsigned int>("l_max_its"))
{
  // backend selection
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
  {
    if (device.is_mps())
      paramError("linear_solver",
                 "The PETSC linear solver does not run on MPS. Use linear_solver = TORCH_PCG.");
    if (torch::typeMetaToScalarType(MooseTensor::floatTensorOptions().dtype()) != torch::kDouble)
      paramError("linear_solver",
                 "The PETSC linear solver requires double precision. Use "
                 "linear_solver = TORCH_PCG for single precision.");
    if (device.is_cuda() && !MARLIN_PETSC_CUDA)
      paramError("linear_solver",
                 "The PETSC linear solver on CUDA requires PETSc and libtorch built with CUDA. Use "
                 "linear_solver = TORCH_PCG.");
  }
  else
    for (const auto & name : {"petsc_options", "petsc_options_iname", "petsc_options_value"})
      if (isParamSetByUser(name))
        paramError(name, "PETSc options are not used by linear_solver = TORCH_PCG.");

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

  // PETSc matches option names case-insensitively, so such fields would share a prefix
  for (const auto i : index_range(buffers))
    for (const auto j : make_range(i))
      if (MooseUtils::toLower(buffers[i]) == MooseUtils::toLower(buffers[j]))
        paramError("buffer",
                   "Fields '",
                   buffers[j],
                   "' and '",
                   buffers[i],
                   "' differ only in case, so their PETSc options prefixes would collide.");

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
                false,
                nullptr};

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
ImplicitFDSolver::findExplicitComputes()
{
  // the root compute updates the explicit terms; the field operators are applied by the
  // solve itself, so they are left out (an automatic root compute contains every compute)
  std::set<const TensorOperatorBase *> operators;
  for (const auto & field : _fields)
    operators.insert(field._operators.begin(), field._operators.end());
  const auto * group = dynamic_cast<const ComputeGroup *>(_compute.get());
  if (group)
    for (const auto & cmp : group->getComputes())
      if (!operators.count(cmp.get()))
        _explicit_computes.push_back(cmp.get());

  // a group without field operators (or a single explicit compute) runs as a whole, keeping
  // its JIT and debug checks
  if (group ? _explicit_computes.size() == group->getComputes().size()
            : !operators.count(_compute.get()))
    _explicit_computes = {_compute.get()};
  _explicit_computes_found = true;
}

void
ImplicitFDSolver::substep()
{
  // update the explicit terms (sources)
  if (!_explicit_computes_found)
    findExplicitComputes();
  for (auto * cmp : _explicit_computes)
    cmp->realSpaceComputeBuffer();
  forwardBuffers();

  _iterations = 0;
  _is_converged = true;
  std::vector<torch::Tensor> u_old;
  for (auto & field : _fields)
  {
    // u is Krylov scratch below; give it its own storage (advanceState() keeps a
    // shallow copy in its old states)
    u_old.push_back(field._u);
    field._u = field._u.clone();

    // b = sum_i r_i + f, evaluated while u still holds u_old
    torch::Tensor b = torch::zeros_like(field._u_base.ownedView(), torch::MemoryFormat::Contiguous);
    for (const auto * fd_op : field._fd_operators)
      if (const auto r = fd_op->rhsContribution(); r.defined())
        b += r;
    if (field._source)
      b += field._source->ownedView();

    if (field._singular)
    {
      // remove only a round-off mean: pairwise summation errs by ~log2(N) eps sum|b|
      // (Higham, ch. 4); 16 covers rounding in b and the cross-rank reduction
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
    torch::Tensor x = field._u_base.ownedView().clone(torch::MemoryFormat::Contiguous);

    SolveResult result;
    const Real b_norm = std::sqrt(_domain.allreduceSum((b * b).sum()).cpu().item<double>());
    if (b_norm == 0.0)
      // the zero-mean solution of A x = 0 is x = 0
      x.zero_();
    else if (_linear_solver == LinearSolver::PETSC)
      result = solvePetsc(field, b, x);
    else
      result = solveTorchPCG(field, b, b_norm, x);
    _iterations += result.iterations;

    if (!result.failure.empty())
    {
      // only a TensorSolveIterationAdaptiveDT time stepper cuts dt on a failed solve, and
      // the old states are already advanced past a failed later substep
      const auto * transient = dynamic_cast<const TransientBase *>(_app.getExecutioner());
      if (_substeps > 1 || !transient ||
          !dynamic_cast<const TensorSolveIterationAdaptiveDT *>(transient->getTimeStepper()))
        mooseError(result.failure,
                   " Use the TensorSolveIterationAdaptiveDT time stepper (with substeps = 1) to "
                   "retry the step with a smaller dt.");

      _console << result.failure << std::endl;
      // retry from the previous state of every field solved so far
      for (const auto i : index_range(u_old))
        _fields[i]._u = u_old[i];
      _is_converged = false;
      return;
    }

    if (field._singular)
      x -= _domain.allreduceSum(x.sum()) / static_cast<Real>(_domain.getNumberOfCells());

    // write x into u and apply the operators to it once more, so that their output buffers
    // hold A_i u rather than A_i of the last Krylov vector
    applyOperator(field, x);
    _tensor_problem.exchangeGhostLayers(field._name, _tensor_problem.getMaxGhostLayer());
  }
}

torch::Tensor
ImplicitFDSolver::applyOperator(Field & field, const torch::Tensor & x)
{
  field._u_base.ownedView().copy_(x);

  torch::Tensor y;
  for (const auto i : index_range(field._operators))
  {
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

ImplicitFDSolver::SolveResult
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

  // written so that a NaN residual counts as not converged
  if (!(res_norm <= _l_tol * b_norm))
  {
    std::ostringstream failure;
    failure << "The TORCH_PCG solve of field '" << field._name << "' did not converge in "
            << _l_max_its << " iterations (relative residual " << res_norm / b_norm
            << ", l_tol = " << _l_tol << ").";
    return {iterations, failure.str()};
  }

  x = x_new;
  return {iterations, ""};
}

ImplicitFDSolver::PetscSolve::~PetscSolve()
{
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&b));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&x));
  PetscCallAbort(PETSC_COMM_WORLD, KSPDestroy(&ksp));
  PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&fine));
  for (auto & B : coarse)
    PetscCallAbort(PETSC_COMM_WORLD, MatDestroy(&B));
  PetscCallAbort(PETSC_COMM_WORLD, MatNullSpaceDestroy(&nullspace));
  hierarchy.reset();
  if (comm != MPI_COMM_NULL)
    MPI_Comm_free(&comm);
}

namespace
{
/// the stencil entries of all operators at the given spacing (empty: the domain's)
std::vector<FDStencilEntry>
fieldStencil(const std::vector<const LinearFDOperator *> & fd_operators,
             const std::vector<Real> & grid_spacing = {})
{
  std::vector<FDStencilEntry> stencil;
  for (const auto * fd_op : fd_operators)
  {
    const auto entries = grid_spacing.empty() ? fd_op->stencilAt() : fd_op->stencilAt(grid_spacing);
    stencil.insert(stencil.end(), entries.begin(), entries.end());
  }
  return stencil;
}

bool
sameStencil(const std::vector<FDStencilEntry> & a, const std::vector<FDStencilEntry> & b)
{
  if (a.size() != b.size())
    return false;
  for (const auto i : index_range(a))
    if (a[i].offset != b[i].offset || a[i].coefficient != b[i].coefficient)
      return false;
  return true;
}

#if MARLIN_PETSC_CUDA
/// makes torch enqueue on PETSc's current CUDA stream while a shell callback runs
class PetscStreamGuard
{
public:
  PetscStreamGuard()
  {
    cudaStream_t petsc_stream;
    PetscCallAbort(PETSC_COMM_SELF, PetscGetCurrentCUDAStream(&petsc_stream));
    _guard = std::make_unique<c10::cuda::CUDAStreamGuard>(
        c10::cuda::getStreamFromExternal(petsc_stream, c10::cuda::current_device()));
  }

private:
  std::unique_ptr<c10::cuda::CUDAStreamGuard> _guard;
};
#endif

/// options prefix of a field's KSP
std::string
petscPrefix(const std::string & name)
{
  return name + "_";
}

bool
onCuda()
{
  return MooseTensor::floatTensorOptions().device().is_cuda();
}
}

void
ImplicitFDSolver::insertPetscOptions(const Field & field)
{
  const auto prefix = petscPrefix(field._name);

  Moose::PetscSupport::PetscOptions po;
  Moose::PetscSupport::addPetscFlagsToPetscOptions(
      getParam<MultiMooseEnum>("petsc_options"), prefix, *this, po);
  Moose::PetscSupport::addPetscPairsToPetscOptions(
      getParam<MooseEnumItem, std::string>("petsc_options_iname", "petsc_options_value"),
      _dim,
      prefix,
      *this,
      po);

  // global database, so the PCMG level KSPs (prefix <buffer>_mg_levels_*) see them too
  for (const auto & flag : po.flags)
    Moose::PetscSupport::setSinglePetscOption(flag.rawName());
  for (const auto & [name, value] : po.pairs)
    Moose::PetscSupport::setSinglePetscOption(name, value);
  // command line options win
  Moose::PetscSupport::addPetscOptionsFromCommandline();
}

void
ImplicitFDSolver::setupPetsc(Field & field)
{
  field._petsc = std::make_unique<PetscSolve>();
  auto & ps = *field._petsc;
  ps.solver = this;
  ps.field = &field;

  const auto partition = PetscFD::gatherPartition(_domain);
  ps.comm = PetscFD::dmdaComm(partition);
  const auto comm = ps.comm;

  const auto stencil = fieldStencil(field._fd_operators);
  const auto width = PetscFD::stencilWidth(stencil);
  const auto levels = PetscFD::countLevels(partition, width);
  ps.hierarchy = std::make_unique<PetscFD::Hierarchy>();
  PetscFD::buildHierarchy(
      partition, comm, levels, width, PetscFD::needsBoxStencil(stencil), *ps.hierarchy);
  auto & hierarchy = *ps.hierarchy;

  const bool cuda = onCuda();
#if MARLIN_PETSC_CUDA
  if (cuda)
  {
    for (auto & da : hierarchy.dms)
    {
      PetscCallAbort(comm, DMSetVecType(da, VECCUDA));
      PetscCallAbort(comm, DMSetMatType(da, MATAIJCUSPARSE));
    }
    for (auto & P : hierarchy.interpolations)
      PetscCallAbort(comm, MatConvert(P, MATAIJCUSPARSE, MAT_INPLACE_MATRIX, &P));
  }
#endif

  // fine level: the operators' own kernels behind a shell
  const PetscInt local = field._u_base.ownedView().numel();
  const PetscInt global = _domain.getNumberOfCells();
  PetscCallAbort(comm, MatCreateShell(comm, local, local, global, global, &ps, &ps.fine));
  PetscCallAbort(comm, MatShellSetOperation(ps.fine, MATOP_MULT, (void (*)(void))shellMult));
  PetscCallAbort(comm,
                 MatShellSetOperation(ps.fine, MATOP_GET_DIAGONAL, (void (*)(void))shellDiagonal));
#if MARLIN_PETSC_CUDA
  if (cuda)
    PetscCallAbort(comm, MatShellSetVecType(ps.fine, VECCUDA));
#endif

  if (field._singular)
    PetscCallAbort(comm, MatNullSpaceCreate(comm, PETSC_TRUE, 0, nullptr, &ps.nullspace));

  // coarse levels: rediscretised and assembled
  ps.coarse.resize(levels - 1);
  for (const auto l : index_range(ps.coarse))
    PetscCallAbort(comm, DMCreateMatrix(hierarchy.dms[l + 1], &ps.coarse[l]));

  if (field._singular)
  {
    std::vector<Mat> all_levels{ps.fine};
    all_levels.insert(all_levels.end(), ps.coarse.begin(), ps.coarse.end());
    for (auto & A : all_levels)
    {
      PetscCallAbort(comm, MatSetOption(A, MAT_SYMMETRIC, PETSC_TRUE));
      PetscCallAbort(comm, MatSetOption(A, MAT_SYMMETRY_ETERNAL, PETSC_TRUE));
      PetscCallAbort(comm, MatSetNullSpace(A, ps.nullspace));
    }
  }
  else
  {
    // MAT_SPD of the coarse levels is set after every assembly (assembleCoarseLevels)
    PetscCallAbort(comm, MatSetOption(ps.fine, MAT_SPD, PETSC_TRUE));
    PetscCallAbort(comm, MatSetOption(ps.fine, MAT_SPD_ETERNAL, PETSC_TRUE));
  }

  // KSPCG + PCMG defaults, set before KSPSetFromOptions so that user options win
  PetscCallAbort(comm, KSPCreate(comm, &ps.ksp));
  PetscCallAbort(comm, KSPSetOptionsPrefix(ps.ksp, petscPrefix(field._name).c_str()));
  PetscCallAbort(comm, KSPSetOperators(ps.ksp, ps.fine, ps.fine));
  PetscCallAbort(comm, KSPSetType(ps.ksp, KSPCG));
  PetscCallAbort(comm, KSPSetNormType(ps.ksp, KSP_NORM_UNPRECONDITIONED));
  PetscCallAbort(comm, KSPSetTolerances(ps.ksp, _l_tol, PETSC_CURRENT, PETSC_CURRENT, _l_max_its));
  PetscCallAbort(comm, KSPSetInitialGuessNonzero(ps.ksp, PETSC_TRUE));

  PC pc;
  PetscCallAbort(comm, KSPGetPC(ps.ksp, &pc));
  PetscCallAbort(comm, PCSetType(pc, PCMG));
  PetscCallAbort(comm, PCMGSetLevels(pc, PetscInt(levels), nullptr));
  // PCMG numbers the levels coarsest first: PCMG level i is hierarchy level levels - 1 - i
  for (const auto i : make_range(1u, levels))
  {
    const auto h = levels - 1 - i;
    PetscCallAbort(comm, PCMGSetInterpolation(pc, PetscInt(i), hierarchy.interpolations[h]));
  }
  for (const auto i : make_range(levels - 1))
  {
    KSP smoother;
    PetscCallAbort(comm, PCMGGetSmoother(pc, PetscInt(i), &smoother));
    const auto B = ps.coarse[levels - 2 - i];
    PetscCallAbort(comm, KSPSetOperators(smoother, B, B));
  }
  // PCMG's default smoother is Chebyshev + SOR. The fine shell has no SOR, so it uses
  // Jacobi from its diagonal. On CUDA the coarse levels use Jacobi too, since PETSc's
  // SOR runs on the host. PCMG level 0 is the coarse LU when levels > 1.
  for (const auto i : make_range(levels))
    if (i == levels - 1 || (cuda && i > 0))
    {
      KSP smoother;
      PC smoother_pc;
      PetscCallAbort(comm, PCMGGetSmoother(pc, PetscInt(i), &smoother));
      PetscCallAbort(comm, KSPGetPC(smoother, &smoother_pc));
      PetscCallAbort(comm, PCSetType(smoother_pc, PCJACOBI));
    }
  if (field._singular && levels > 1)
  {
    // the coarse LU meets a zero pivot from the constant nullspace; shifting it changes
    // only the preconditioner
    KSP coarse_ksp;
    PC coarse_pc;
    PetscCallAbort(comm, PCMGGetCoarseSolve(pc, &coarse_ksp));
    PetscCallAbort(comm, KSPGetPC(coarse_ksp, &coarse_pc));
    PetscCallAbort(comm, PCFactorSetShiftType(coarse_pc, MAT_SHIFT_NONZERO));
  }

  assembleCoarseLevels(field);

  insertPetscOptions(field);
  PetscCallAbort(comm, KSPSetFromOptions(ps.ksp));

  // b and x get the solver's tensors placed before every solve
  if (cuda)
  {
#if MARLIN_PETSC_CUDA
    PetscCallAbort(comm, VecCreateMPICUDAWithArray(comm, 1, local, global, nullptr, &ps.b));
    PetscCallAbort(comm, VecCreateMPICUDAWithArray(comm, 1, local, global, nullptr, &ps.x));
#endif
  }
  else
  {
    PetscCallAbort(comm, VecCreateMPIWithArray(comm, 1, local, global, nullptr, &ps.b));
    PetscCallAbort(comm, VecCreateMPIWithArray(comm, 1, local, global, nullptr, &ps.x));
  }
}

void
ImplicitFDSolver::assembleCoarseLevels(Field & field)
{
  auto & ps = *field._petsc;
  const auto stencil = fieldStencil(field._fd_operators);
  if (sameStencil(stencil, ps.assembled_stencil))
    return;

  const auto comm = ps.comm;
  const auto & grid_spacing = _domain.getGridSpacing();
  for (const auto l : index_range(ps.coarse))
  {
    // level k = l + 1 has spacing 2^k h. PCMG restricts with P^T, whose column sums
    // are 2^dim per level, so the rediscretised operator is scaled by 2^(dim k) to
    // match P^T A P on smooth fields. This scales only the preconditioner.
    const auto k = l + 1;
    std::vector<Real> spacing(_dim);
    for (const auto d : make_range(_dim))
      spacing[d] = grid_spacing(d) * Real(1 << k);
    const PetscReal scale = std::pow(2.0, Real(_dim * k));
    auto & B = ps.coarse[l];
    PetscFD::assembleStencilOperator(
        ps.hierarchy->dms[k], fieldStencil(field._fd_operators, spacing), scale, B);
    // assembly clears MAT_SPD
    if (!field._singular)
    {
      PetscCallAbort(comm, MatSetOption(B, MAT_SPD, PETSC_TRUE));
      PetscCallAbort(comm, MatSetOption(B, MAT_SPD_ETERNAL, PETSC_TRUE));
    }
  }

  // the shell's action changed too; bump its state and reset the KSP operators so
  // that PCMG sets up every level again (Chebyshev bounds, coarse factorisation)
  PetscCallAbort(comm, PetscObjectStateIncrease((PetscObject)ps.fine));
  PetscCallAbort(comm, KSPSetOperators(ps.ksp, ps.fine, ps.fine));
  ps.assembled_stencil = stencil;
}

ImplicitFDSolver::SolveResult
ImplicitFDSolver::solvePetsc(Field & field, torch::Tensor & b, torch::Tensor & x)
{
  if (!field._petsc)
    setupPetsc(field);
  else
    assembleCoarseLevels(field);
  auto & ps = *field._petsc;
  const auto comm = ps.comm;

  mooseAssert(b.is_contiguous() && x.is_contiguous(), "b and x must be contiguous");
  if (onCuda())
  {
#if MARLIN_PETSC_CUDA
    // b and x were written on torch's stream; KSPSolve runs on PETSc's
    cudaStream_t petsc_stream;
    PetscCallAbort(comm, PetscGetCurrentCUDAStream(&petsc_stream));
    const auto petsc = c10::cuda::getStreamFromExternal(petsc_stream, c10::cuda::current_device());
    at::cuda::CUDAEvent written;
    written.record(c10::cuda::getCurrentCUDAStream());
    written.block(petsc);
    PetscCallAbort(comm, VecCUDAPlaceArray(ps.b, b.data_ptr<PetscScalar>()));
    PetscCallAbort(comm, VecCUDAPlaceArray(ps.x, x.data_ptr<PetscScalar>()));
#endif
  }
  else
  {
    PetscCallAbort(comm, VecPlaceArray(ps.b, b.data_ptr<PetscScalar>()));
    PetscCallAbort(comm, VecPlaceArray(ps.x, x.data_ptr<PetscScalar>()));
  }

  ps.exception = nullptr;
  const auto ierr = KSPSolve(ps.ksp, ps.b, ps.x);

  if (onCuda())
  {
#if MARLIN_PETSC_CUDA
    PetscCallAbort(comm, VecCUDAResetArray(ps.b));
    PetscCallAbort(comm, VecCUDAResetArray(ps.x));
    // and torch reads x after PETSc has written it
    cudaStream_t petsc_stream;
    PetscCallAbort(comm, PetscGetCurrentCUDAStream(&petsc_stream));
    at::cuda::CUDAEvent solved;
    solved.record(c10::cuda::getStreamFromExternal(petsc_stream, c10::cuda::current_device()));
    solved.block(c10::cuda::getCurrentCUDAStream());
#endif
  }
  else
  {
    PetscCallAbort(comm, VecResetArray(ps.b));
    PetscCallAbort(comm, VecResetArray(ps.x));
  }

  // an exception from the operator kernels (caught in shellMult) is rethrown here, outside
  // PETSc's frames
  if (ps.exception)
    std::rethrow_exception(ps.exception);
  PetscCallAbort(comm, ierr);

  KSPConvergedReason reason;
  PetscInt iterations;
  PetscCallAbort(comm, KSPGetConvergedReason(ps.ksp, &reason));
  PetscCallAbort(comm, KSPGetIterationNumber(ps.ksp, &iterations));
  if (reason < 0)
    return {static_cast<unsigned int>(iterations),
            "The PETSC solve of field '" + field._name + "' failed after " +
                std::to_string(iterations) + " iterations: " + KSPConvergedReasons[reason] + "."};
  return {static_cast<unsigned int>(iterations), ""};
}

PetscErrorCode
ImplicitFDSolver::shellMult(Mat A, Vec x, Vec y)
{
  PetscSolve * ps;
  const PetscScalar * x_array;
  PetscScalar * y_array;
  PetscMemType x_memtype, y_memtype;
  PetscFunctionBeginUser;
  PetscCall(MatShellGetContext(A, &ps));
  PetscCall(VecGetArrayReadAndMemType(x, &x_array, &x_memtype));
  PetscCall(VecGetArrayWriteAndMemType(y, &y_array, &y_memtype));
  PetscCheck(PetscMemTypeDevice(x_memtype) == PetscMemTypeDevice(y_memtype),
             PETSC_COMM_SELF,
             PETSC_ERR_ARG_INCOMP,
             "x and y must live on the same device");

  auto options = MooseTensor::floatTensorOptions().device(torch::kCPU);
#if MARLIN_PETSC_CUDA
  std::unique_ptr<PetscStreamGuard> guard;
  if (PetscMemTypeDevice(x_memtype))
  {
    options = options.device(torch::kCUDA, c10::cuda::current_device());
    guard = std::make_unique<PetscStreamGuard>();
  }
#endif
  // C++ exceptions must not unwind through PETSc's C frames: keep it for solvePetsc
  PetscErrorCode ierr = PETSC_SUCCESS;
  try
  {
    const auto shape = ps->field->_u_base.ownedView().sizes().vec();
    const auto in = torch::from_blob(const_cast<PetscScalar *>(x_array), shape, options);
    torch::from_blob(y_array, shape, options).copy_(ps->solver->applyOperator(*ps->field, in));
  }
  catch (...)
  {
    ps->exception = std::current_exception();
    ierr = PETSC_ERR_LIB;
  }

  PetscCall(VecRestoreArrayReadAndMemType(x, &x_array));
  PetscCall(VecRestoreArrayWriteAndMemType(y, &y_array));
  PetscCheck(
      ierr == PETSC_SUCCESS, PETSC_COMM_SELF, ierr, "The operator action threw an exception");
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode
ImplicitFDSolver::shellDiagonal(Mat A, Vec d)
{
  PetscSolve * ps;
  PetscFunctionBeginUser;
  PetscCall(MatShellGetContext(A, &ps));
  PetscCall(VecSet(d, ps->solver->diagonal(*ps->field)));
  PetscFunctionReturn(PETSC_SUCCESS);
}
