#!/usr/bin/env bash
#
# Single-GPU size scaling of the implicit finite-difference Cahn-Hilliard solve
# (benchmarks/01_spinodal_decomposition/1a_implicit_fd.i) for the PETSC and
# TORCH_PCG linear solvers.
#
# The domain stays [0,200]^2 and the grid is refined (h = 200/n), so each size
# is a different discrete problem from PFHub 1a; the runs are for timing only.
# The biharmonic grows like 1/h^4, so the systems become less diagonally
# dominated with n and Jacobi PCG needs more iterations per solve.
#
# Usage (from this directory, after building marlin-opt):
#   ./run_scaling.sh
#   SIZES="1024 2048" STEPS=2 BACKENDS=PETSC ./run_scaling.sh
#
# Results: results/<backend>_<n>.{log,csv} and results/summary.csv

set -euo pipefail

# ---- parameters -------------------------------------------------------------
SIZES=${SIZES:-"512 1024 2048 4096 8192"}
BACKENDS=${BACKENDS:-"PETSC TORCH_PCG"}
# time steps per run (each is `substeps` = 1000 linear solves, as in the input)
STEPS=${STEPS:-2}
DEVICE=${DEVICE:-cuda}
# per-run limit (s); a run that hits it is reported as TIMEOUT
TIME_LIMIT=${TIME_LIMIT:-1800}
MARLIN=${MARLIN:-../../marlin-opt}
INPUT=../01_spinodal_decomposition/1a_implicit_fd.i
OUT=results
# -----------------------------------------------------------------------------

mkdir -p "$OUT"
summary="$OUT/summary.csv"
echo "backend,n,dofs,steps,status,wall_time_s,wall_time_per_step_s" > "$summary"

# provenance
{
  echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host: $(hostname)"
  echo "git: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "device: $DEVICE"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || true
} > "$OUT/provenance.txt"

for n in $SIZES; do
  for backend in $BACKENDS; do
    base="$OUT/${backend}_${n}"
    echo "== $backend n=$n ($((n * n)) dofs), $STEPS steps"

    status=OK
    timeout "$TIME_LIMIT" "$MARLIN" -i "$INPUT" \
      Domain/nx=$n Domain/ny=$n \
      Executioner/num_steps=$STEPS \
      TensorSolver/linear_solver=$backend \
      "TensorOutputs/active=''" \
      Outputs/file_base="$base" \
      --compute-device="$DEVICE" > "$base.log" 2>&1 || status=$?
    [ "$status" = 124 ] && status=TIMEOUT
    [ "$status" != OK ] && [ "$status" != TIMEOUT ] && status="FAILED($status)"

    # wall_time is the last column of the postprocessor CSV; take the last row
    wall=""
    rows=0
    if [ -f "$base.csv" ]; then
      wall=$(tail -n 1 "$base.csv" | awk -F, '{print $NF}')
      rows=$(($(wc -l < "$base.csv") - 1)) # minus the header (output is on TIMESTEP_END only)
    fi
    per_step=""
    [ -n "$wall" ] && [ "$rows" -gt 0 ] && per_step=$(awk -v w="$wall" -v r="$rows" 'BEGIN{print w / r}')
    echo "$backend,$n,$((n * n)),$rows,$status,$wall,$per_step" >> "$summary"
    echo "   $status, wall_time=${wall:-n/a} s over $rows steps"
  done
done

echo
column -s, -t "$summary"
