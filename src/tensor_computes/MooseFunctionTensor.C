/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "MooseFunctionTensor.h"
#include "Function.h"
#include "MarlinUtils.h"
#include "TensorProblem.h"

registerMooseObject("MarlinApp", MooseFunctionTensor);

InputParameters
MooseFunctionTensor::validParams()
{
  InputParameters params = TensorOperator::validParams();
  params.addClassDescription("Map a MooseFunction to a tensor.");
  params.addRequiredParam<FunctionName>("function", "Function to map.");
  // params.addParam<bool>("reciprocal", false, "Construct a reciprocal buffer");
  return params;
}

MooseFunctionTensor::MooseFunctionTensor(const InputParameters & parameters)
  : TensorOperator(parameters), FunctionInterface(this), _func(getFunction("function"))
{
}

void
MooseFunctionTensor::computeBuffer()
{
  // this rank's owned cells: local size and global index offset
  std::array<int64_t, 3> begin{{0, 0, 0}}, end{{0, 0, 0}};
  _domain.getLocalBounds(_domain.comm().rank(), begin, end);
  const auto & n = _domain.getLocalGridSize();
  const auto & dx = _domain.getGridSpacing();
  const auto & x0 = _domain.getDomainMin();
  auto buffer = torch::zeros(std::vector<int64_t>(n.begin(), n.begin() + _dim), torch::kDouble);

  switch (_domain.getDim())
  {
    {
      case 1:
        auto b = buffer.accessor<double, 1>();
        for (const auto i : make_range(n[0]))
          b[i] = _func.value(_time, Point(x0(0) + (begin[0] + i) * dx(0) + dx(0) / 2.0, 0.0, 0.0));
        break;
    }
    case 2:
    {
      auto b = buffer.accessor<double, 2>();
      for (const auto j : make_range(n[1]))
        for (const auto i : make_range(n[0]))
          b[i][j] = _func.value(_time,
                                Point(x0(0) + (begin[0] + i) * dx(0) + dx(0) / 2.0,
                                      x0(1) + (begin[1] + j) * dx(1) + dx(1) / 2.0,
                                      0.0));
      break;
    }
    case 3:
    {
      auto b = buffer.accessor<double, 3>();
      for (const auto k : make_range(n[2]))
        for (const auto j : make_range(n[1]))
          for (const auto i : make_range(n[0]))
            b[i][j][k] = _func.value(_time,
                                     Point(x0(0) + (begin[0] + i) * dx(0) + dx(0) / 2.0,
                                           x0(1) + (begin[1] + j) * dx(1) + dx(1) / 2.0,
                                           x0(2) + (begin[2] + k) * dx(2) + dx(2) / 2.0));
      break;
    }
    default:
      mooseError("Unsupported dimension");
  }

  // write into the owned cells of a tensor with this rank's local (ghosted) shape; the
  // ghost cells are filled by TensorProblem::exchangeGhostLayers() before any ghosted read
  const auto ghost = _tensor_problem.getMaxGhostLayer();
  _u = torch::zeros(_tensor_problem.getLocalTensorShape({}), MooseTensor::floatTensorOptions());
  auto owned = _u;
  for (const auto d : make_range(_dim))
    owned = owned.narrow(d, ghost, n[d]);
  owned.copy_(buffer);
}

void
MooseFunctionTensor::realSpaceComputeBuffer()
{
  computeBuffer();
}
