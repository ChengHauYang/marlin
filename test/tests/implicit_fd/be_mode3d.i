# Backward Euler diffusion u_t = D Laplacian(u) of a single periodic Fourier
# mode. On the periodic cell-centred grid the mode is an eigenvector of the
# discrete Laplacian with eigenvalue -lambda_h, so every implicit step must
# multiply it by exactly g = 1 / (1 + dt D lambda_h), up to the linear solver
# tolerance.

D = 0.5
# 5 steps of dt, then 5 steps of dt2 (dt2 != dt exercises the operator rebuild)
dt = 0.01
dt2 = 0.01
n = 16
stencil = 3
dim = 3

# Symbol of -Laplacian per axis for the wavenumber 2 pi, h = 1/n:
#   3-point: (2 - 2 cos(2 pi h)) / h^2
#   5-point: (5/2 - 8/3 cos(2 pi h) + 1/6 cos(4 pi h)) / h^2
lambda3 = '${fparse n^2 * (2 - 2 * cos(2 * pi / n))}'
lambda5 = '${fparse n^2 * (5 / 2 - 8 / 3 * cos(2 * pi / n) + 1 / 6 * cos(4 * pi / n))}'
lambda = '${fparse dim * if(stencil = 3, lambda3, lambda5)}'

[Domain]
  dim = 3
  nx = ${n}
  ny = ${n}
  nz = ${n}
  zmax = 1
  xmax = 1
  ymax = 1
  parallel_mode = REAL_SPACE
  periodic_directions = 'X Y Z'
[]

[TensorBuffers]
  [u]
  []
  [dudt_u]
  []
  [lap_u]
  []
  [gold]
  []
  [diff]
  []
[]

[TensorComputes]
  [Initialize]
    [u_ic]
      type = ParsedCompute
      buffer = u
      expression = 'sin(2*pi*x)*sin(2*pi*y)*sin(2*pi*z)'
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
      stencil_width = ${stencil}
    []
  []
  [Postprocess]
    [gold_compute]
      type = ParsedCompute
      buffer = gold
      # g(dt)^(t/dt) up to t1 = 5 dt, then g(dt)^5 g(dt2)^((t - t1)/dt2)
      expression = 'sin(2*pi*x)*sin(2*pi*y)*sin(2*pi*z) * if(t < t1 + dt / 2, g1^(t / dt), g1^5 * g2^((t - t1) / dt2))'
      constant_names = 'g1 g2 dt dt2 t1'
      constant_expressions = '${fparse 1 / (1 + dt * D * lambda)} ${fparse 1 / (1 + dt2 * D * lambda)} ${dt} ${dt2} ${fparse 5 * dt}'
      extra_symbols = true
    []
    [diff_compute]
      type = ParsedCompute
      buffer = diff
      expression = 'abs(u - gold)'
      inputs = 'u gold'
    []
  []
[]

[TensorSolver]
  type = ImplicitFDSolver
  buffer = u
  operators = 'dudt lap'
  root_compute = dudt
[]

[Postprocessors]
  [max_error]
    type = TensorExtremeValuePostprocessor
    buffer = diff
    value_type = MAX
  []
  [max_gold]
    type = TensorExtremeValuePostprocessor
    buffer = gold
    value_type = MAX
  []
[]

[Problem]
  type = TensorProblem
[]

[Executioner]
  type = Transient
  num_steps = 10
  [TimeStepper]
    type = TimeSequenceStepper
    time_sequence = '${fparse 1 * dt} ${fparse 2 * dt} ${fparse 3 * dt} ${fparse 4 * dt} ${fparse 5 * dt}
                     ${fparse 5 * dt + dt2} ${fparse 5 * dt + 2 * dt2} ${fparse 5 * dt + 3 * dt2}
                     ${fparse 5 * dt + 4 * dt2} ${fparse 5 * dt + 5 * dt2}'
  []
[]

[Outputs]
  csv = true
[]
