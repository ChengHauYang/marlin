# Verification against an independent analytic solution: the steady 1D elliptic
# Robin-BC benchmark from OpenGeoSys's documentation,
# https://www.opengeosys.org/6.5.7/docs/benchmarks/elliptic/elliptic-robin/
# ("second benchmark": Dirichlet at x=0, Robin at x=1). The governing equation is
# k*Laplacian(h) = 0 on [0,1] with k=1, boundary conditions
#   h(0) = g_D,                 dh/dn|_{x=1} = alpha*(h0 - h(1)),
# and exact linear solution
#   h(x) = [alpha*(h0 - g_D)/(1 + alpha)] * x + g_D.
# TensorRobinBC's convention is a*u + b*du/dn = c; the OGS Robin condition
# du/dn = alpha*(h0 - u) rearranges to alpha*u + du/dn = alpha*h0, i.e.
# a = alpha, b = 1, c = alpha*h0.
#
# Uses alpha = +2 rather than OGS's alpha = -2: the ghost coefficient on u0,
# -(a/2 - b/dx)/(a/2 + b/dx), has magnitude > 1 for a < 0, so explicit relaxation to
# the steady state diverges (OGS solves the steady system directly). With alpha = 2,
# h0 = 1.5 and g_D = 1 the exact solution is h(x) = x/3 + 1.
#
# Y is periodic, and the IC matches neither the steady profile nor the BCs.
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
    expression = '1 + x / 3'
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
      # alpha*u + du/dn = alpha*h0, with alpha = 2, h0 = 1.5.
      type = TensorRobinBC
      variable = u
      boundary = right
      a = 2
      b = 1
      c = 3
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
  # same grid and CFL margin as dirichlet_linear.i
  dt = 1
  num_steps = 4
[]

[Outputs]
  [out]
    type = CSV
  []
[]
