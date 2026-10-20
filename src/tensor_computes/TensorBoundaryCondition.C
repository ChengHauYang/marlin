/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "TensorBoundaryCondition.h"
#include "DomainAction.h"

namespace
{
const std::string face_names = "left right bottom top front back";
}

std::string
TensorBoundaryCondition::faceName(unsigned int face)
{
  return MooseUtils::toLower(MooseEnum(face_names).getNames()[face]);
}

InputParameters
TensorBoundaryCondition::validParams()
{
  InputParameters params = TensorOperatorBase::validParams();
  params.addRequiredParam<TensorInputBufferName>("variable",
                                                 "The field this boundary condition applies to.");
  MooseEnum boundary(face_names);
  params.addRequiredParam<MooseEnum>(
      "boundary", boundary, "Which face of the domain this boundary condition applies to.");
  params.addClassDescription("Base class for boundary conditions on a non-periodic real-space "
                             "domain edge.");
  return params;
}

TensorBoundaryCondition::TensorBoundaryCondition(const InputParameters & parameters)
  : TensorOperatorBase(parameters),
    _face(getParam<MooseEnum>("boundary").getEnum<Face>()),
    _variable_name(getParam<TensorInputBufferName>("variable"))
{
  if (faceDim() >= _domain.getDim())
    paramError("boundary", "This face does not exist in a ", _domain.getDim(), "D domain.");
  if (_domain.getPeriodicDirections()[faceDim()])
    paramError("boundary", "This face lies on a periodic direction.");
}
