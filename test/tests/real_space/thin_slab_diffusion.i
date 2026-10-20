# Verification against Crank, "The Mathematics of Diffusion" (2nd ed.), Sec. 2.5,
# eq. (2.15): an instantaneous planar source of concentration C0 confined initially
# to a thin slab -h < x < h, diffusing into an infinite medium, has the exact solution
#   C(x, t) = (C0/2) * { erf[(h-x)/(2*sqrt(D*t))] + erf[(h+x)/(2*sqrt(D*t))] }
#
# Pseudo-1D (ny = 1, Y periodic), D = 1, h = 1. The X faces are 10 diffusion lengths
# beyond the slab edge at the comparison time t = 1, so the far-field Dirichlet u = 0
# does not measurably affect the solution near the slab.
[Domain]
  dim = 2
  # dx = 0.1 puts the slab edges x = +/-1 on cell faces, so the box IC holds exactly
  # the slab mass 2*h*C0 (nx = 200 leaves a ~1% deficit that does not refine away)
  nx = 220
  ny = 1
  parallel_mode = REAL_SPACE
  periodic_directions = 'Y'
  xmin = -11
  xmax = 11
  ymin = -0.5
  ymax = 0.5
[]

[TensorComputes]
  [Initialize]
    [u_ic]
      type = ParsedCompute
      buffer = u
      # C0 = 1 inside the slab |x| < h = 1, zero outside
      expression = 'if(abs(x) < 1, 1, 0) + 0*y'
      extra_symbols = true
    []
  []
  [BCs]
    [left]
      type = TensorDirichletBC
      variable = u
      boundary = left
      value = 0
    []
    [right]
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
      expression = '0.5 * (erf((1 - x) / (2 * sqrt(t))) + erf((1 + x) / (2 * sqrt(t)))) + 0*y'
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
  # dx = 0.1; explicit-diffusion CFL limit dt_cfl = dx^2/(2*D) = 5e-3,
  # requiring >= dt/dt_cfl = 200 substeps for dt = 1; 250 gives a safety margin.
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
