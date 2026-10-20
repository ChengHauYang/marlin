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
  [u_func]
    type = ParsedFunction
    expression = '1 + 2 * x'
  []
[]

[TensorComputes]
  [Initialize]
    [u_ic]
      # u = 1 + 2*x satisfies the Dirichlet values below exactly, so the discrete
      # Laplacian is zero to roundoff everywhere, including the boundary-adjacent cells.
      type = MooseFunctionTensor
      buffer = u
      function = u_func
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
    [lap_compute]
      type = FiniteDifferenceLaplacian
      input = u
      buffer = lap
    []
  []
[]

[Postprocessors]
  [max_lap]
    type = TensorExtremeValuePostprocessor
    buffer = lap
    value_type = MAX
  []
  [min_lap]
    type = TensorExtremeValuePostprocessor
    buffer = lap
    value_type = MIN
  []
[]

[Executioner]
  type = Transient
  num_steps = 1
[]

[Outputs]
  [out]
    type = CSV
  []
[]
