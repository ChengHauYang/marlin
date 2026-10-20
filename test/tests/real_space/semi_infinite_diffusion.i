# Verification against Crank, "The Mathematics of Diffusion" (2nd ed.), Sec. 3.3(i),
# eq. (3.13): a semi-infinite medium (x > 0) with uniform initial concentration C0 = 0
# and constant surface concentration C1 maintained at x = 0 has the exact solution
#   C(x, t) = C1 * erfc(x / (2*sqrt(D*t)))
# (special case of (3.13) with C0 = 0, using erfc = 1 - erf).
#
# Pseudo-1D (ny = 1, Y periodic), D = 1. The domain extends 10 diffusion lengths at
# the comparison time t = 1 (erfc(10) ~ 2e-45), so the far Dirichlet u = 0 does not
# measurably affect the solution near x = 0.
[Domain]
  dim = 2
  nx = 100
  ny = 1
  parallel_mode = REAL_SPACE
  periodic_directions = 'Y'
  xmin = 0
  xmax = 10
  ymin = -0.5
  ymax = 0.5
[]

[TensorComputes]
  # no [Initialize]: buffers start at zero, the C0 = 0 initial condition
  [BCs]
    [left]
      type = TensorDirichletBC
      variable = u
      boundary = left
      value = 1
    []
    [right]
      # Far boundary, approximating the semi-infinite medium's x -> infinity
      # condition; diffusion has not reached here by the comparison time.
      type = TensorDirichletBC
      variable = u
      boundary = right
      value = 0
    []
  []
  [Solve]
    [dt]
      type = FiniteDifferenceLaplacian
      input = u
      buffer = dudt
      factor = 1
    []
    [u_exact]
      type = ParsedCompute
      buffer = u_exact
      expression = 'erfc(x / (2 * sqrt(t))) + 0*y'
      extra_symbols = true
    []
    [diff]
      type = ParsedCompute
      buffer = diff
      expression = 'abs(u - u_exact)'
      inputs = 'u u_exact'
    []
  []
[]

[TensorSolver]
  type = RealSpaceForwardEuler
  buffer = u
  time_derivative = dudt
  substeps = 250
[]

[Postprocessors]
  [max_error]
    type = TensorExtremeValuePostprocessor
    buffer = diff
    value_type = MAX
    execute_on = 'TIMESTEP_END'
  []
[]

[Executioner]
  type = Transient
  dt = 1
  num_steps = 1
[]

[Outputs]
  [out]
    type = CSV
  []
[]
