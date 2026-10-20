/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "TensorRobinBC.h"
#include "DomainAction.h"

registerMooseObject("MarlinApp", TensorRobinBC);

InputParameters
TensorRobinBC::validParams()
{
  InputParameters params = TensorBoundaryCondition::validParams();
  params.addParam<Real>("a", 1, "Robin coefficient on u (a*u + b*du/dn = c).");
  params.addParam<Real>("b", 0, "Robin coefficient on du/dn (a*u + b*du/dn = c).");
  params.addParam<Real>("c", 0, "Robin right-hand-side value (a*u + b*du/dn = c).");
  params.addClassDescription(
      "General Robin boundary condition a*u + b*du/dn = c on a non-periodic domain face.");
  return params;
}

TensorRobinBC::TensorRobinBC(const InputParameters & parameters)
  : TensorBoundaryCondition(parameters),
    _a(getParam<Real>("a")),
    _b(getParam<Real>("b")),
    _c(getParam<Real>("c"))
{
  // the ghost value divides by a/2 + b/dx (see ghostCoefficientOnU0); this also rejects a = b = 0
  const auto dx = _domain.getGridSpacing()(faceDim());
  if (std::abs(_a / 2.0 + _b / dx) <= 1e-12 * (std::abs(_a) / 2.0 + std::abs(_b) / dx))
    paramError(
        "a", "a/2 + b/dx must be nonzero (dx = ", dx, " is the grid spacing normal to the face).");
}

Real
TensorRobinBC::ghostCoefficientOnU0(Real a, Real b, Real dx)
{
  // ughost = [c - u0*(a/2 - b/dx)] / (a/2 + b/dx)  =>  coefficient on u0
  return -(a / 2.0 - b / dx) / (a / 2.0 + b / dx);
}

Real
TensorRobinBC::ghostConstantTerm(Real a, Real b, Real c, Real dx)
{
  return c / (a / 2.0 + b / dx);
}

void
TensorRobinBC::fillGhostCellsStatic(torch::Tensor & buffer,
                                    bool is_upper_face,
                                    int64_t face_dim,
                                    unsigned int ghost_layers,
                                    int64_t halo,
                                    int64_t owned_extent,
                                    Real a,
                                    Real b,
                                    Real c,
                                    Real dx)
{
  if (ghost_layers > 2)
    ::mooseError(
        "TensorRobinBC fills at most two ghost layers, but ", ghost_layers, " were requested.");
  // the second ghost layer divides by a/2 + b/(3dx) (the constructor checks a/2 + b/dx)
  if (ghost_layers == 2 &&
      std::abs(a / 2.0 + b / (3.0 * dx)) <= 1e-12 * (std::abs(a) / 2.0 + std::abs(b) / (3.0 * dx)))
    ::mooseError("TensorRobinBC with two ghost layers requires a/2 + b/(3*dx) to be nonzero (dx = ",
                 dx,
                 " is the grid spacing normal to the face).");

  // Marlin's real-space grid is cell-centered (axis built from
  // min+dx/2 ... max-dx/2), so for either face the boundary sits at the
  // midpoint between the first interior cell (u0) and the first ghost cell
  // (ughost), and the outward normal derivative at that midpoint is well
  // approximated by a central difference between u0 and ughost. Both the
  // lower and upper face reduce to the SAME formula (ghostCoefficientOnU0 /
  // ghostConstantTerm) because "ughost" always means "the cell on the
  // outward side of u0" for whichever face is being filled.
  const auto u0 = is_upper_face ? buffer.narrow(face_dim, halo + owned_extent - 1, 1)
                                : buffer.narrow(face_dim, halo, 1);

  const auto coeff = ghostCoefficientOnU0(a, b, dx);
  const auto rhs = ghostConstantTerm(a, b, c, dx);

  // first ghost row, directly from the Robin relation at the face
  auto ghost1 = is_upper_face ? buffer.narrow(face_dim, halo + owned_extent, 1)
                              : buffer.narrow(face_dim, halo - 1, 1);
  ghost1.copy_(coeff * u0 + rhs);

  if (ghost_layers >= 2)
  {
    // second ghost row: the same Robin relation on the pair (u1, ghost2), which is also
    // centred on the face but 3*dx apart. Exact for linear fields; the even mirror
    // ghost2 = u1 for zero flux and the odd mirror ghost2 = 2c - u1 for Dirichlet.
    const auto u1 = is_upper_face ? buffer.narrow(face_dim, halo + owned_extent - 2, 1)
                                  : buffer.narrow(face_dim, halo + 1, 1);
    auto ghost2 = is_upper_face ? buffer.narrow(face_dim, halo + owned_extent + 1, 1)
                                : buffer.narrow(face_dim, halo - 2, 1);
    ghost2.copy_(ghostCoefficientOnU0(a, b, 3.0 * dx) * u1 + ghostConstantTerm(a, b, c, 3.0 * dx));
  }
}

std::pair<std::vector<FDStencilEntry>, Real>
TensorRobinBC::stencilContributionStatic(unsigned int dim, Real a, Real b, Real c, Real dx)
{
  // Substitute the SAME ghost-value formula used by fillGhostCells(Static)
  // (ughost = coeff*u0 + rhs) into the interior stencil's u0 row: any
  // reference to the ghost cell is replaced by `coeff` times the
  // boundary-adjacent interior unknown u0 plus a constant `rhs`. The caller
  // (a future implicit assembler) is responsible for folding this
  // coefficient into its own stencil row at the u0 offset; here we report
  // it as a single-entry stencil (offset 0 along faceDim(), i.e. "the
  // ghost's dependence on u0") plus the resulting constant contribution.
  const auto coeff = ghostCoefficientOnU0(a, b, dx);
  const auto rhs = ghostConstantTerm(a, b, c, dx);

  std::vector<int> offset(dim, 0);
  return {std::vector<FDStencilEntry>{FDStencilEntry{offset, coeff}}, rhs};
}

void
TensorRobinBC::fillGhostCells(torch::Tensor & buffer,
                              unsigned int ghost_layers,
                              int64_t halo,
                              int64_t owned_extent,
                              Real dx) const
{
  fillGhostCellsStatic(buffer,
                       isUpperFace(),
                       static_cast<int64_t>(faceDim()),
                       ghost_layers,
                       halo,
                       owned_extent,
                       _a,
                       _b,
                       _c,
                       dx);
}

std::pair<std::vector<FDStencilEntry>, Real>
TensorRobinBC::stencilContribution(unsigned int ghost_layers, Real dx) const
{
  // only the first ghost relation is reported; a second ghost layer is not represented
  if (ghost_layers != 1)
    mooseError("TensorRobinBC::stencilContribution supports one ghost layer only.");

  return stencilContributionStatic(_dim, _a, _b, _c, dx);
}
