#!/usr/bin/env bash
# Measure decode throughput for the six points (run on an otherwise idle GPU).
#
#   srun -p <gpu-partition> --gres=gpu:1 -N1 -n1 -t 00:30:00 bash scripts/reproduce/run_throughput.sh <weights.bin> <output-dir>
#
# env: BIN_DIR (default <repo>/build/reproduce/bin, from build_points.sh)
# Protocol: archive from the library encoder; one byte-exact verification run
# per point; then 3 independent processes per point, each 10 warm-up + 50 timed
# launches (median of 50), with R = smallest power of two >= 8 such that
# R x archive size >= 768 MiB (so the compressed data cannot sit in the
# Infinity Cache).
set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" || "$(hostname -s)" == login* || "$(hostname -s)" == hacchead* ]]; then
  echo "ERROR: submit this benchmark with srun on a GPU compute node." >&2
  exit 2
fi
src="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
weights="${1:?usage: run_throughput.sh <weights.bin> <output-dir>}"
out="${2:?usage: run_throughput.sh <weights.bin> <output-dir>}"
bins="${BIN_DIR:-${src}/build/reproduce/bin}"
rows=4096; cols=16384
[[ "$(stat -c %s "${weights}")" == $((rows * cols)) ]] || { echo "ERROR: ${weights} must be ${rows}x${cols} bytes" >&2; exit 2; }
[[ ! -e "${out}" ]] || { echo "ERROR: ${out} exists" >&2; exit 2; }
mkdir -p "${out}/archives" "${out}/correctness" "${out}/timing"

#        name                 binary             block  input
points=("R0|barrier1_b8192|8192|global"
        "O1|barrier0_b8192|8192|global"
        "O1+O2|barrier0_b8192|8192|lds"
        "O1+O2+O3 @ 4 KiB|barrier0_b4096|4096|lds"
        "O1+O2+O3 @ 2 KiB|barrier0_b2048|2048|lds"
        "O1+O2+O3 @ 1 KiB|barrier0_b1024|1024|lds")

# Archives: one per block size, encoded by the library on the GPU. The 8 KiB
# archive is also re-encoded by the O1 binary and must be byte-identical.
prepare() {  # binary artifact
  "${bins}/hipans_decode_bench_$1" --prepare --artifact "$2" --input global --reference "${weights}" \
    --rows ${rows} --cols ${cols} --warmup 1 --iters 1 >"$2.json"
}
prepare barrier1_b8192 "${out}/archives/block8192.ans"
prepare barrier0_b8192 "${out}/archives/block8192.check.ans"
cmp "${out}/archives/block8192.ans" "${out}/archives/block8192.check.ans" || { echo "ERROR: O1 binary encodes a different archive" >&2; exit 1; }
for block in 4096 2048 1024; do prepare "barrier0_b${block}" "${out}/archives/block${block}.ans"; done
sha256sum "${out}"/archives/block*.ans | sed "s#${out}/##" >"${out}/archives/sha256.txt"

i=0
for entry in "${points[@]}"; do
  IFS='|' read -r name binary block input <<<"${entry}"
  archive="${out}/archives/block${block}.ans"
  common=(--artifact "${archive}" --reference "${weights}" --rows ${rows} --cols ${cols}
          --mode decode --input "${input}" --threads 128 --grid-mult 4 --streams 1)
  # byte-exact verification (decoded LDS tile copied out and compared with the input)
  "${bins}/hipans_decode_bench_${binary}" "${common[@]}" --replicas 1 --warmup 0 --iters 1 \
    >"${out}/correctness/point-${i}.json"
  size="$(stat -c %s "${archive}")"
  replicas=8
  while (( replicas * size < 805306368 && replicas < 256 )); do replicas=$((replicas * 2)); done
  for run in 1 2 3; do
    "${bins}/hipans_decode_bench_${binary}" "${common[@]}" --replicas "${replicas}" \
      --warmup 10 --iters 50 --skip-verify >"${out}/timing/point-${i}-run-${run}.json"
  done
  echo "[${i}] ${name}: replicas=${replicas}, 3 runs done"
  i=$((i + 1))
done
python3 -I "${src}/scripts/reproduce/summarize_throughput.py" "${out}" "${src}/scripts/reproduce/published_mi350x.json"
