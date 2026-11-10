# Handover: verify Marlin NEML2 v3 migration on CUDA (PR #198)

Branch: `patch_neml2_cuda` (fork `ChengHauYang/marlin`), currently identical to `patch_neml2`
at commit `2b41d17` "Fix NEML2 v3 eager model construction and precision handling".
MOOSE submodule: `66e88fb` (`next`). Do not change or commit the submodule pointer.

Goal: confirm on a CUDA machine what the macOS run could not: gcc `-Werror` build, CUDA
runtime in opt and dbg, CUDA DOUBLE and SINGLE precision, threaded runs, and unit tests.
Fix anything CUDA-specific on `patch_neml2_cuda`, then port the fix to `patch_neml2`.

---

## 1. What changed and why (read before debugging)

All code changes are in `src/tensor_computes/NEML2TensorCompute.C`.

| Change | Reason |
| :--- | :--- |
| `neml2::eager::Model(path, model_name, device)`: removed the 5th `dtype` argument | The NEML2 v3 constructor is `(path, name, optional<Device>, vector<string> load)`. This was the only CI failure: every job stopped at compile, so no test ever ran on CI. |
| Load under `DefaultDtypeGuard(Marlin dtype)`; reload once if a **parameter-free** model's dtype mismatches | An eager model's dtype = its first float parameter (NEML2 always builds these as float64), otherwise the **torch default dtype** at load time. The first eager model in a process imports torch into the embedded Python interpreter, and that import resets the default to float32 mid-load. Without the reload, `copy_vector` runs in float32 in DOUBLE mode and `typed_tensors.gradient` diffs by ~7e-7. |
| On MPS, model stays on CPU | MPS has no float64. Not relevant to CUDA. |
| `computeBuffer()`: inputs `.to(device).to(dtype)`; outputs `.to(dtype).to(device)` | Avoids float64 conversions on a float64-less device. On CUDA this is just two cheap steps. |

`forward()` strictly checks that every input matches the model's device **and** dtype
(`neml2/types/_boundary.py::check_tensor`), so any mismatch shows up as a `TypeError`
from `EagerModel`.

Expected precision behaviour on CUDA:
- DOUBLE: every model runs in float64 on the GPU.
- SINGLE: parameter-free models (`copy_vector`, `symmetric`) run in float32. Models with
  parameters (`multiply`) **run in float64 on the GPU**: inputs are upcast and outputs
  downcast to float32. This is a NEML2 v3 API limitation (no dtype control), not a bug.

### Already verified on macOS (arm64, clang 21, libtorch 2.13, NEML2 3.1.0)
- Build OK (with NEML2 + libtorch). clang-format clean.
- Full suite CPU: 316 passed, 22 skipped, 0 failed. `--n-threads 2` and `-p 2` pass.
- Unit tests: 43/43 pass.
- MPS: `neml2.scalar`, `neml2.scalar_single`, `neml2.symmetric` pass.
- SINGLE on CPU: `scalar` and `symmetric` CSVs match gold.
- Reload instrumentation: the first parameter-free model reloads once (Float to Double);
  later models load correctly the first time. Models with parameters never reload.

### NOT verified yet (this handover)
CUDA opt and dbg runtime, gcc 14 `-Werror` build, Linux, `--n-threads 2` on CUDA, unit
tests on Linux, Apptainer container job, Documentation build.

---

## 2. Environment: replicate CI (recommended)

CI ("Test cuda", "Test cuda debug") ran in:
```
oras://mooseharbor.hpc.inl.gov/moose-dev/moose-dev-ubuntu24-cuda-gcc-openmpi-x86_64_v3:2026.11.09
```
That image has gcc 14.2, CUDA 13.3, Open MPI 5.0.11, Python 3.14.9 at `/opt/python`, with
torch and NEML2 installed in `/opt/python/lib/python3.14/site-packages`. The embedded
NEML2 interpreter therefore finds torch without any `PYTHONPATH` setup. On macOS,
`PYTHONPATH` *was* required.

```bash
apptainer shell --nv oras://mooseharbor.hpc.inl.gov/moose-dev/moose-dev-ubuntu24-cuda-gcc-openmpi-x86_64_v3:2026.11.09
nvidia-smi                                   # GPU visible?
python3 -c 'import torch, neml2; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), neml2.__version__)'
# expect: CUDA build of torch, is_available True, neml2 3.x
```
CI pins one GPU with `CUDA_VISIBLE_DEVICES=0` (dbg job) or `1` (opt job). Do the same:
`export CUDA_VISIBLE_DEVICES=0`.

If you are not using the container (conda on a Linux CUDA box): follow
`compile_marlin.md` section 2, but you need a **CUDA** torch build, the torch python
version must match the python MOOSE embeds, and if torch lives in a different env you
must `export PYTHONPATH=<that env>/lib/python3.X/site-packages` before running tests.

---

## 3. Get the code (CI tests the PR *merged into* `idaholab/marlin:devel`)

```bash
git clone https://github.com/ChengHauYang/marlin.git && cd marlin
git checkout patch_neml2_cuda
git log -1 --oneline                         # 2b41d17 (or later, if fixes were added)
# Optional, CI-identical: merge onto upstream devel in a scratch branch
git checkout -b cuda_ci_merge
git pull --no-ff https://github.com/idaholab/marlin.git devel
git submodule update --init --checkout moose
git submodule status moose                   # 66e88fb...
```

---

## 4. Build (mirrors CI steps)

```bash
N=16                                          # parallel jobs
cd moose && ./configure --with-neml2 --with-libtorch && cd ..
grep -E '^(ENABLE_NEML2|ENABLE_LIBTORCH)' moose/conf_vars.mk     # both true
(cd moose/framework && make -j $N)

# Opt, as in "Test cuda" (warnings are errors)
ADDITIONAL_CPPFLAGS=-Werror make -j $N 2>&1 | tee build_opt.log
./marlin-opt --show-capabilities | grep -A3 -E '"(neml2|libtorch)"'   # neml2 true

# Unit tests, as in "Build unit"
(cd unit && make -j $N 2>&1 | tee ../build_unit.log)
```
Debug, as in "Test cuda debug" (run later, it is slow):
```bash
METHOD=dbg make -j $N 2>&1 | tee build_dbg.log
```

---

## 5. Test plan (run in order, stop and triage on first unexpected result)

| # | Command | Expect |
| :--- | :--- | :--- |
| T1 | build above | `make` exit 0, no `-Werror` failure in `NEML2TensorCompute.C` |
| T2 | `./marlin-opt --show-capabilities` | neml2 true, libtorch true |
| T3 | `./run_tests --re='neml2\|typed_tensors' --compute-device=cuda -j 4` | `neml2.scalar`, `neml2.symmetric`, `typed_tensors.gradient`, `typed_tensors.gradient_vector` OK; `neml2.scalar_single` SKIP (spec is mps-only) |
| T4 | same as T3 with `--n-threads 2` | same as T3 |
| T5 | manual CUDA SINGLE (below) | exit 0; CSVs match gold within ~1e-6 |
| T6 | manual CUDA DOUBLE dtype check (below) | log says `Running on 'cuda' with double precision` |
| T7 | `./run_tests -j $N --compute-device=cuda` | 0 failed (record pass/skip counts) |
| T8 | `./run_tests -j $N --compute-device=cuda --n-threads 2` | 0 failed |
| T9 | `./run_tests -j $N` (CPU, gcc) | 0 failed (mac: 316 passed, 22 skipped) |
| T10 | `(cd unit && ./run_tests -j $N)` | all pass (mac: 43) |
| T11 | `METHOD=dbg ./run_tests --re='neml2\|typed_tensors' --compute-device=cuda -j 2` | same as T3; watch memory (specs note ~360 MB/slot in cuda debug) |
| T12 | optional: `METHOD=dbg ./run_tests -j $N --compute-device=cuda` | 0 failed |

Use `-v` on any failing test to get the full log.

### T5: CUDA SINGLE (not covered by the test specs)
```bash
cd test/tests/neml2
for t in scalar symmetric; do
  ../../../marlin-opt -i $t.i Domain/floating_precision=SINGLE --compute-device=cuda \
    Outputs/file_base=/tmp/${t}_cuda_single 2>&1 | tee /tmp/${t}_cuda_single.log | grep -E "Running on|ERROR"
  python3 - "$t" <<'EOF'
import csv, sys
t = sys.argv[1]
a = list(csv.reader(open(f"gold/{t}_out.csv"))); b = list(csv.reader(open(f"/tmp/{t}_cuda_single.csv")))
assert a[0] == b[0] and len(a) == len(b), ("shape/header", a, b)
err = max(abs(float(x) - float(y)) for ra, rb in zip(a[1:], b[1:]) for x, y in zip(ra, rb))
print(t, "max abs err", err, "OK" if err < 1e-6 else "CHECK")
EOF
done
cd ../typed_tensors
../../../marlin-opt -i gradient.i Domain/floating_precision=SINGLE --compute-device=cuda \
  TensorOutputs/xdmf/file_base=/tmp/grad_cuda_single 2>&1 | grep -E "Running on|ERROR"   # must run, no gold compare
cd ../../..
```

### T6 (optional but recommended): confirm the reload path on CUDA
Temporarily add two debug lines in `NEML2TensorCompute.C` constructor, rebuild, run, then
`git checkout src/tensor_computes/NEML2TensorCompute.C`:
```cpp
  _model = load_model();
  std::cerr << "DBG " << name() << " first-load dtype=" << c10::toString(_model->dtype())
            << " device=" << _model->device() << std::endl;            // add
  ...
  if (_model->dtype() != dtype && _model->parameter_base_shapes().empty())
    _model = load_model();
  std::cerr << "DBG " << name() << " final dtype=" << c10::toString(_model->dtype()) << std::endl;  // add
```
Run `typed_tensors/gradient.i --compute-device=cuda` (DOUBLE). Expected:
`first-load dtype=Float device=cuda`, then `final dtype=Double`.
With SINGLE: `Float`, then `Float`. `neml2/scalar.i`: `Double`, then `Double` in both modes.

---

## 6. Triage guide

| Symptom | Likely cause | Action |
| :--- | :--- | :--- |
| `-Werror` failure in `NEML2TensorCompute.C` (gcc only) | gcc warning that clang does not emit | Fix minimally. Note: `#pragma once` at the top of the `.C` is pre-existing on `devel` (since #35) and harmless in unity builds. Only touch it if gcc flags it. |
| `ModuleNotFoundError: No module named 'torch'` / `'neml2'` | embedded python is not the one that has torch | In the container this should not happen. Outside it, set `PYTHONPATH` (section 2). |
| `EagerModel ... dtype=torch.float32 (expected torch.float64)` or the reverse | model dtype differs from the cast in `computeBuffer()` | Should be impossible (inputs are cast to `_model->dtype()`); capture the full message. |
| `EagerModel ... device=cuda:0 (expected cuda:1)` or similar | multi-GPU index mismatch | Re-run with a single `CUDA_VISIBLE_DEVICES`; report Marlin's `Running on` line and the error. |
| `typed_tensors.gradient` HDF5 DIFF ~1e-7 in DOUBLE on CUDA | parameter-free model ended up float32 (reload path not taken) | Run T6 instrumentation and report the DBG lines. |
| `neml2.*` CSVDIFF on CUDA DOUBLE at ~1e-15 | GPU vs CPU roundoff | Report the values. Do not loosen tolerances without discussing. |
| CUDA init errors / hangs on first `eager::Model` | torch CUDA context created from embedded python plus C++ | Capture the backtrace; try `-j 1`; report. |
| OOM / killed in dbg | per-slot memory (specs use `min_slots = 2`) | Re-run with lower `-j`. |

---

## 7. Report back (paste this filled in)

```
Machine / GPU / driver:             (nvidia-smi header)
Container or env:                   (image tag or conda env)
torch / CUDA / neml2 versions:      (python one-liner from section 2)
Commit tested (and merged onto devel?):
T1 opt -Werror build:               PASS/FAIL (first error)
T2 capabilities:
T3 neml2|typed_tensors cuda:        x passed, y skipped, z failed
T4 same, --n-threads 2:
T5 cuda SINGLE scalar / symmetric:  max abs err ... / ...
T6 DBG lines (if run):
T7 full cuda:                       x/y/z
T8 full cuda --n-threads 2:
T9 full cpu (gcc):
T10 unit:
T11/T12 dbg:
Any fix committed to patch_neml2_cuda (hash + one line):
```
