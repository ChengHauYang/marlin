/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "FiniteDifferenceBiharmonic.h"
#include "FiniteDifferenceLaplacian.h"
#include "DomainAction.h"

#include <map>

registerMooseObject("MarlinApp", FiniteDifferenceBiharmonic);

InputParameters
FiniteDifferenceBiharmonic::validParams()
{
  InputParameters params = TensorOperator<torch::Tensor>::validParams();
  params.addRequiredParam<TensorInputBufferName>("input", "The input buffer");
  params.addParam<Real>("factor", 1, "Pre-factor for the biharmonic");
  MooseEnum stencil("3=3 5=5", "3");
  params.addParam<MooseEnum>("stencil_width",
                             stencil,
                             "Stencil width of the Laplacian that is applied twice (3 => 3-point, "
                             "5 => 5-point)");
  params.addClassDescription(
      "Computes the biharmonic Laplacian(Laplacian(u)) using finite differences.");
  return params;
}

FiniteDifferenceBiharmonic::FiniteDifferenceBiharmonic(const InputParameters & parameters)
  : TensorOperator<torch::Tensor>(parameters),
    _laplacian_radius(getParam<MooseEnum>("stencil_width") == 5 ? 2u : 1u),
    _input_name(getParam<TensorInputBufferName>("input")),
    _u_in(getInputBufferByName(_input_name, 2 * _laplacian_radius)),
    _factor(getParam<Real>("factor"))
{
}

void
FiniteDifferenceBiharmonic::realSpaceComputeBuffer()
{
  _tensor_problem.runComputeWithGhosts(*this);
}

void
FiniteDifferenceBiharmonic::computeBuffer()
{
  // convolution kernel from the stencil (offset o at kernel index o + radius)
  const int radius = 2 * _laplacian_radius;
  const int64_t ksize = 2 * radius + 1;
  std::vector<int64_t> kernel_shape(_dim, ksize);
  auto kernel = torch::zeros(kernel_shape, _u_in.options());
  for (const auto & entry : stencilAt())
  {
    auto element = kernel;
    for (const auto d : make_range(_dim))
      element = element[entry.offset[d] + radius];
    element.fill_(entry.coefficient);
  }

  std::vector<int64_t> kernel_view{1, 1};
  kernel_view.insert(kernel_view.end(), kernel_shape.begin(), kernel_shape.end());
  kernel = kernel.view(kernel_view);
  std::vector<int64_t> input_view{1, 1};
  input_view.insert(input_view.end(), _u_in.sizes().begin(), _u_in.sizes().end());
  const auto input = _u_in.view(input_view);

  torch::Tensor result;
  if (_dim == 1)
    result = torch::nn::functional::conv1d(
        input, kernel, torch::nn::functional::Conv1dFuncOptions().padding(radius));
  else if (_dim == 2)
    result = torch::nn::functional::conv2d(
        input, kernel, torch::nn::functional::Conv2dFuncOptions().padding(radius));
  else
    result = torch::nn::functional::conv3d(
        input, kernel, torch::nn::functional::Conv3dFuncOptions().padding(radius));
  _u.copy_(result.view(_u.sizes()));
}

std::vector<FDStencilEntry>
FiniteDifferenceBiharmonic::stencilAt() const
{
  const auto & grid_spacing = _domain.getGridSpacing();
  std::vector<Real> dx(_dim);
  for (const auto d : make_range(_dim))
    dx[d] = grid_spacing(d);
  return stencilAt(dx);
}

std::vector<FDStencilEntry>
FiniteDifferenceBiharmonic::stencilAt(const std::vector<Real> & grid_spacing) const
{
  return buildStencil(_dim, _laplacian_radius == 2, grid_spacing, _factor);
}

LinearFDOperator::Definiteness
FiniteDifferenceBiharmonic::definiteness() const
{
  // the square of the symmetric periodic Laplacian is positive semidefinite with
  // the constants as its nullspace
  return _factor > 0.0 ? Definiteness::POSITIVE_SEMIDEFINITE : Definiteness::OTHER;
}

std::vector<FDStencilEntry>
FiniteDifferenceBiharmonic::buildStencil(unsigned int dim,
                                         bool wide,
                                         const std::vector<Real> & grid_spacing,
                                         Real factor)
{
  const auto laplacian = FiniteDifferenceLaplacian::buildStencil(dim, wide, grid_spacing, 1.0);

  std::map<std::vector<int>, Real> composed;
  for (const auto & a : laplacian)
    for (const auto & b : laplacian)
    {
      std::vector<int> offset(dim);
      for (const auto d : make_range(dim))
        offset[d] = a.offset[d] + b.offset[d];
      composed[offset] += a.coefficient * b.coefficient;
    }

  std::vector<FDStencilEntry> stencil;
  for (const auto & [offset, coefficient] : composed)
    stencil.push_back({offset, factor * coefficient});
  return stencil;
}
