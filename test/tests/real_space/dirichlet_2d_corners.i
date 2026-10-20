# Non-periodic on both axes (Dirichlet left/right, zero-flux Neumann bottom/top),
# so the ghost fill reaches the four domain corners. exchangeGhostLayers() fills one
# axis at a time, so a corner ghost depends on the fill order.
#
# u = 1 + 2*x is uniform on every face, so it satisfies all four BCs exactly, and the
# discrete Laplacian must be zero everywhere, including the corner-adjacent cells.
[Domain]
  dim = 2
  nx = 20
  ny = 10
  parallel_mode = REAL_SPACE
  xmin = 0
  xmax = 1
  ymin = 0
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
      # the steady profile implied by the four BCs below
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
    [bottom]
      type = TensorNeumannBC
      variable = u
      boundary = bottom
      flux = 0
    []
    [top]
      type = TensorNeumannBC
      variable = u
      boundary = top
      flux = 0
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
