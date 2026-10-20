# Zero-flux Neumann boundary condition: pure diffusion with zero flux on the X faces
# and periodic Y, from a non-uniform initial condition. The total integral of u must
# be conserved to near machine precision while u relaxes to its mean.
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

[TensorComputes]
  [Initialize]
    [u_ic]
      type = ParsedCompute
      buffer = u
      # non-uniform in x and y, so diffusion moves mass internally
      expression = '1 + sin(2 * pi * x) + 0.3 * cos(pi * y)'
      extra_symbols = true
    []
  []
  [BCs]
    [left]
      type = TensorNeumannBC
      variable = u
      boundary = left
      flux = 0
    []
    [right]
      type = TensorNeumannBC
      variable = u
      boundary = right
      flux = 0
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
  [mass]
    type = TensorIntegralPostprocessor
    buffer = u
    execute_on = 'INITIAL TIMESTEP_END'
  []
[]

[Executioner]
  type = Transient
  # same grid and CFL margin as dirichlet_linear.i
  dt = 1
  num_steps = 4
[]

[Outputs]
  [out]
    type = CSV
  []
[]
