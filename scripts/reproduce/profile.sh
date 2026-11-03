#!/usr/bin/env bash
# Collect AMD profiler data for the six points: rocprof-compute counters and
# rocprofv3 host_trap PC samples. Run on an otherwise idle GPU.
#
#   srun -p <gpu-partition> --gres=gpu:1 -N1 -n1 -t 00:30:00 bash scripts/reproduce/profile.sh <weights.bin> <throughput-output-dir> <output-dir>
#
# <throughput-output-dir> is the output of run_throughput.sh (its archives/ are reused).
# env: BIN_DIR (default <repo>/build/reproduce/bin)
#      ROCPROF_COMPUTE          rocprof-compute executable (default: rocprof-compute)
#      ROCPROF_COMPUTE_PYTHON   optional Python used to start it (needed if it lives in a venv)
# Then: srun -p <cpu-partition> -N1 -n1 -t 00:10:00 python3 scripts/reproduce/analyze_cycles.py --profile-dir <output-dir> --bin-dir $BIN_DIR \
#         --throughput-json <throughput-output-dir>/summary.json --output cycles.json
set -uo pipefail
if [[ -z "${SLURM_JOB_ID:-}" || "$(hostname -s)" == login* || "$(hostname -s)" == hacchead* ]]; then
  echo "ERROR: submit profiling with srun on a GPU compute node." >&2
  exit 2
fi
src="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
weights="${1:?usage: profile.sh <weights.bin> <throughput-output-dir> <output-dir>}"
archives="${2:?usage: profile.sh <weights.bin> <throughput-output-dir> <output-dir>}/archives"
out="${3:?usage: profile.sh <weights.bin> <throughput-output-dir> <output-dir>}"
bins="${BIN_DIR:-${src}/build/reproduce/bin}"
compute=("${ROCPROF_COMPUTE_PYTHON:+${ROCPROF_COMPUTE_PYTHON}}" "${ROCPROF_COMPUTE:-$(command -v rocprof-compute)}")
[[ -z "${compute[0]}" ]] && compute=("${compute[@]:1}")
[[ ! -e "${out}" ]] || { echo "ERROR: ${out} exists" >&2; exit 2; }
mkdir -p "${out}"

#       dir     binary           block input  PC-sampling launches
points=("r0|barrier1_b8192|8192|global|3000"
        "o1|barrier0_b8192|8192|global|3000"
        "o2|barrier0_b8192|8192|lds|3000"
        "o3_4k|barrier0_b4096|4096|lds|5000"
        "o3_2k|barrier0_b2048|2048|lds|5000"
        "o3_1k|barrier0_b1024|1024|lds|5000")

for entry in "${points[@]}"; do
  IFS='|' read -r name binary block input pc_iters <<<"${entry}"
  bin="${bins}/hipans_decode_bench_${binary}"
  d="${out}/${name}"; mkdir -p "${d}"
  common=(--artifact "${archives}/block${block}.ans" --reference "${weights}" --rows 4096 --cols 16384
          --mode decode --input "${input}" --replicas 16 --threads 128 --grid-mult 4 --streams 1 --skip-verify)

  # 1. counters (one profiled launch after two warm-ups)
  timeout 2400 "${compute[@]}" profile --output-directory "${d}/counters" --overwrite \
    --block sol memchart spi wavefront cu_pipe lds vl1d l2 \
    -- "${bin}" "${common[@]}" --warmup 2 --iters 1 >"${d}/profile.log" 2>&1
  echo "${name}: counters profile rc=$?"
  pdir="$(dirname "$(find "${d}/counters" -name 'results_pmc_perf_*.csv' | head -1)")"
  dispatch="$(python3 -I - "${pdir}" <<'PY'
import csv, re, sys
from pathlib import Path
ids = set()
for p in Path(sys.argv[1]).glob("results_pmc_perf_*.csv"):
    for r in csv.DictReader(p.open()):
        if re.search(r"hipansLDS<\s*128,\s*false,\s*false", r["Kernel_Name"]):
            ids.add(int(r["Dispatch_ID"]))
if not ids:
    raise SystemExit("no hipANS decode dispatch")
print(max(ids))
PY
)"
  timeout 900 "${compute[@]}" analyze -p "${pdir}" --dispatch "${dispatch}" \
    --output-format stdout >"${d}/analysis.txt" 2>"${d}/analysis.err"
  echo "${name}: analyze rc=$? dispatch=${dispatch}"

  # 2. host_trap PC sampling, one sample every 10 us
  mkdir -p "${d}/pc"
  timeout 1800 rocprofv3 --pc-sampling-beta-enabled true --pc-sampling-unit time \
    --pc-sampling-method host_trap --pc-sampling-interval 10000 \
    --output-directory "${d}/pc" --output-file "${name}" --output-format json \
    -- "${bin}" "${common[@]}" --warmup 10 --iters "${pc_iters}" >"${d}/pc/rocprofv3.log" 2>&1
  echo "${name}: pc sampling rc=$?"
done
echo "profile data in ${out}"
