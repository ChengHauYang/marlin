# Steady periodic Poisson problem -D Laplacian(u) = f with the manufactured
# solution u = sin(2 pi x) cos(4 pi y) + cos(6 pi x) / 2 (zero mean). The
# operator is singular (constants), and the solver returns the zero-mean
# solution, so the error against u converges at the stencil order (2 for the
# 3-point, 4 for the 5-point stencil).

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
  [lap_u]
  []
  [f]
  []
  [exact]
  []
  [diff]
  []
[]

[TensorComputes]
  [Initialize]
    [u_ic]
      type = ParsedCompute
      buffer = u
      # initial guess (any field; the solve returns the zero-mean solution)
      expression = 'sin(2*pi*x) * sin(2*pi*y)'
      extra_symbols = true
    []
    [f_ic]
      type = ParsedCompute
      buffer = f
      # -D Laplacian(u): the two modes have |k|^2 = 20 pi^2 and 36 pi^2
      expression = 'D * (20 * pi^2 * sin(2*pi*x) * cos(4*pi*y) + 18 * pi^2 * cos(6*pi*x))'
      constant_names = 'D'
      constant_expressions = '${D}'
      extra_symbols = true
    []
  []
  [Solve]
    [lap]
      type = FiniteDifferenceLaplacian
      buffer = lap_u
      input = u
      factor = '${fparse -D}'
    []
  []
  [Postprocess]
    [exact_compute]
      type = ParsedCompute
      buffer = exact
      expression = 'sin(2*pi*x) * cos(4*pi*y) + cos(6*pi*x) / 2'
      extra_symbols = true
    []
    [diff_compute]
      type = ParsedCompute
      buffer = diff
      expression = 'abs(u - exact)'
      inputs = 'u exact'
    []
  []
[]

[TensorSolver]
  type = ImplicitFDSolver
  buffer = u
  operators = 'lap'
  source = f
[]

[Postprocessors]
  [max_error]
    type = TensorExtremeValuePostprocessor
    buffer = diff
    value_type = MAX
  []
  [mean_u]
    type = TensorAveragePostprocessor
    buffer = u
  []
[]

[Problem]
  type = TensorProblem
[]

[Executioner]
  type = Transient
  num_steps = 1
[]

[Outputs]
  csv = true
[]
