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
    expression = '1 + 2 * x + 0.001 * y'
  []
[]

[TensorComputes]
  [Initialize]
    [u_ic]
      # Not the steady solution and not matching the BCs pointwise (it depends on y);
      # diffusion must drive u to the exact steady profile u = 1 + 2*x.
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
      type = TensorDirichletBC
      variable = u
      boundary = right
      value = 3
    []
  []
  [Solve]
    [dt]
      type = FiniteDifferenceLaplacian
      input = u
      buffer = dudt
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
  [max_u]
    type = TensorExtremeValuePostprocessor
    buffer = u
    value_type = MAX
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [min_u]
    type = TensorExtremeValuePostprocessor
    buffer = u
    value_type = MIN
    execute_on = 'INITIAL TIMESTEP_END'
  []
[]

[Executioner]
  type = Transient
  # dx = 0.05, dy = 0.25: explicit CFL limit 1/(2*(1/dx^2 + 1/dy^2)) ~= 1.2e-3,
  # above the substep dt/substeps = 1e-3.
  dt = 1
  num_steps = 4
[]

[Outputs]
  [out]
    type = CSV
  []
[]
