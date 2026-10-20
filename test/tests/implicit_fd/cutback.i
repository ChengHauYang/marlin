# Backward Euler diffusion of a Gaussian with a linear iteration limit that the first dt
# exceeds. The torch PCG solve needs about 44 iterations at dt = 0.008, 34 at 0.004 and 23
# at 0.002 (32 x 32, l_tol = 1e-8), so with l_max_its = 30 the step is cut twice and then
# converges.

D = 0.5
n = 32

[Domain]
  dim = 2
  nx = ${n}
  ny = ${n}
  xmax = 1
  ymax = 1
  parallel_mode = REAL_SPACE
  periodic_directions = 'X Y'
[]

[TensorBuffers]
  [u]
  []
  [dudt_u]
  []
  [lap_u]
  []
[]

[TensorComputes]
  [Initialize]
    [u_ic]
      type = ParsedCompute
      buffer = u
      expression = 'exp(-((x - 0.5)^2 + (y - 0.5)^2) / 0.005)'
      extra_symbols = true
    []
  []
  [Solve]
    [dudt]
      type = TimeDerivativeOperator
      buffer = dudt_u
      input = u
    []
    [lap]
      type = FiniteDifferenceLaplacian
      buffer = lap_u
      input = u
      factor = '${fparse -D}'
    []
  []
[]

[TensorSolver]
  type = ImplicitFDSolver
  buffer = u
  operators = 'dudt lap'
  linear_solver = TORCH_PCG
  l_max_its = 30
[]

[Postprocessors]
  [max_u]
    type = TensorExtremeValuePostprocessor
    buffer = u
    value_type = MAX
  []
[]

[Problem]
  type = TensorProblem
[]

[Executioner]
  type = Transient
  num_steps = 2
  [TimeStepper]
    type = TensorSolveIterationAdaptiveDT
    dt = 0.008
    # no growth or shrinking from the iteration count, only the cut on failure
    min_iterations = 0
    max_iterations = 1000
  []
[]

[Outputs]
  csv = true
[]
