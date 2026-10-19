# Implicit FD Cahn-Hilliard size scaling

Single-GPU timing of the implicit finite-difference solve of
`../01_spinodal_decomposition/1a_implicit_fd.i` (backward Euler, biharmonic
implicit, double-well flux divergence explicit) for the `PETSC` (CG +
geometric multigrid) and `TORCH_PCG` (Jacobi CG) linear solvers, on grids from
512^2 to 8192^2 cells.

The domain stays [0,200]^2 and only the grid is refined, so the runs are not
PFHub 1a results; they are for timing only. Each time step is 1000 linear
solves (`substeps`), and the `wall_time` postprocessor gives the run time.

```
./run_scaling.sh
SIZES="1024 2048" STEPS=1 BACKENDS=PETSC ./run_scaling.sh
```

Results go to `results/` (`summary.csv`, one log and postprocessor CSV per
run, `provenance.txt`).
