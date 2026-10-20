/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "TensorRobinBC.h"
#include "MarlinUtils.h"
#include "gtest/gtest.h"

// fillGhostCellsStatic() (the explicit/matrix-free path) and
// stencilContributionStatic() (the implicit path) must, for the same (a, b, c, dx),
// imply the same ghost value, and that ghost value must satisfy the Robin condition
// itself. The second check is independent of the shared ghostCoefficientOnU0() /
// ghostConstantTerm() helpers, so a sign error in them is caught.
namespace
{
// Checks fillGhostCellsStatic() and stencilContributionStatic() agree, and satisfy
// the BC, for one Robin triple (a, b, c), on both the lower and upper face of a small
// 1D interior field.
void
checkDualPath(Real a, Real b, Real c, Real dx)
{
  const int64_t halo = 2;
  const int64_t owned_extent = 4;
  const int64_t total = owned_extent + 2 * halo;
  const Real u0_lower = 1.7;  // boundary-adjacent interior value, lower face
  const Real u0_upper = -0.4; // boundary-adjacent interior value, upper face

  for (const bool is_upper_face : {false, true})
  {
    auto buffer = torch::zeros({total}, MooseTensor::floatTensorOptions());
    const auto u0 = is_upper_face ? u0_upper : u0_lower;
    buffer[is_upper_face ? halo + owned_extent - 1 : halo] = u0;

    TensorRobinBC::fillGhostCellsStatic(
        buffer, is_upper_face, /*face_dim=*/0, /*ghost_layers=*/1, halo, owned_extent, a, b, c, dx);

    const auto ghost_value = buffer[is_upper_face ? halo + owned_extent : halo - 1].item<double>();

    const auto [stencil, rhs] = TensorRobinBC::stencilContributionStatic(1, a, b, c, dx);
    ASSERT_EQ(stencil.size(), 1u);
    const auto predicted_ghost_value = stencil[0].coefficient * u0 + rhs;

    // ghost_value comes from a tensor op (fillGhostCellsStatic), which may run in
    // single precision depending on the active compute device/dtype setting;
    // predicted_ghost_value is plain double arithmetic, so the tolerance must
    // allow for that precision gap, not just floating-point roundoff.
    const Real tolerance =
        MooseTensor::floatTensorOptions().dtype() == torch::kFloat64 ? 1e-12 : 1e-6;
    EXPECT_NEAR(ghost_value, predicted_ghost_value, tolerance)
        << "a=" << a << " b=" << b << " c=" << c << " dx=" << dx
        << " is_upper_face=" << is_upper_face;

    // the ghost satisfies a*u + b*du/dn = c at the face: u = (u0 + ughost)/2 and,
    // since the ghost lies on the outward side of either face, du/dn = (ughost - u0)/dx
    const Real bc_residual = a * (u0 + ghost_value) / 2.0 + b * (ghost_value - u0) / dx - c;
    EXPECT_NEAR(bc_residual, 0.0, tolerance * (std::abs(a) / 2.0 + std::abs(b) / dx))
        << "a=" << a << " b=" << b << " c=" << c << " dx=" << dx
        << " is_upper_face=" << is_upper_face;
  }
}
}

TEST(TensorBoundaryConditionTest, DirichletDualPath)
{
  // a=1, b=0 (TensorDirichletBC preset): u = value on the face.
  checkDualPath(/*a=*/1.0, /*b=*/0.0, /*c=*/3.5, /*dx=*/0.1);
}

TEST(TensorBoundaryConditionTest, NeumannDualPath)
{
  // a=0, b=1 (TensorNeumannBC preset): du/dn = flux on the face.
  checkDualPath(/*a=*/0.0, /*b=*/1.0, /*c=*/-2.0, /*dx=*/0.1);
}

TEST(TensorBoundaryConditionTest, GeneralRobinDualPath)
{
  checkDualPath(/*a=*/2.0, /*b=*/3.0, /*c=*/1.25, /*dx=*/0.05);
}
