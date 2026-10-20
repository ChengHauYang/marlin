# Nonzero-flux Neumann boundary condition: steady 1D diffusion with u(0) = 1
# (Dirichlet, left) and du/dn = 2 (Neumann, right) has the exact solution
# u = 1 + 2*x. Y is periodic, and the IC is not the steady profile.
[Domain]
  dim = 2
  nx = 20
  ny = 8
  parallel_mode = REAL_SPACE
  periodic_directions = 'Y'
  xmin = 0
  xmax = 1
  ymin = -1
  ymax = 1
[]

[Functions]
  [u_ic_func]
    type = ParsedFunction
    expression = '0.5'
  []
  [u_exact_func]
    type = ParsedFunction
    expression = '1 + 2 * x'
  []
[]

[TensorComputes]
  [Initialize]
    [u_ic]
      type = MooseFunctionTensor
      buffer = u
      function = u_ic_func
    []
  []
  [BCs]
    [left]
      type = TensorDirichletBC
      variable = u
      boundary = left
      value = 1
    []
    [right]
      type = TensorNeumannBC
      variable = u
      boundary = right
      flux = 2
    []
  []
  [Solve]
    [dt]
      type = FiniteDifferenceLaplacian
      input = u
      buffer = dudt
    []
    [u_exact]
      type = MooseFunctionTensor
      buffer = u_exact
      function = u_exact_func
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
  substeps = 1000
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
  # same grid and CFL margin as dirichlet_linear.i; the Neumann face makes the slowest
  # mode decay more slowly, so max_error needs 15 steps to plateau
  dt = 1
  num_steps = 15
[]

[Outputs]
  [out]
    type = CSV
  []
[]
