#!/usr/bin/env bash
set -euo pipefail

# Submit the smallest ANS GPU round-trip test through Slurm.
#
# Optional:
#   GPU_PARTITION  HACC GPU partition name (default: mi210_u280_u55c)
#   BUILD_DIR      Existing build directory (default: <repo>/build)
#   TEST_FILTER    GoogleTest filter (default: ANSTest.ZeroSized)
#   LOG_DIR        Directory for terminal output copies (default: <repo>/logs)

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-${repo_dir}/build}"
test_filter="${TEST_FILTER:-ANSTest.ZeroSized}"
log_dir="${LOG_DIR:-${repo_dir}/logs}"
gpu_partition="${GPU_PARTITION:-mi210_u280_u55c}"

test_bin="${build_dir}/bin/ans_test"
if [[ ! -x "${test_bin}" ]]; then
  echo "Missing executable: ${test_bin}" >&2
  echo "Build the project in a CPU Slurm allocation before running this script." >&2
  exit 2
fi

mkdir -p "${log_dir}"

if [[ "${1:-}" != "--inside-allocation" ]]; then
  exec srun \
    --job-name=hipans-ans-smoke \
    -p "${gpu_partition}" \
    --cpus-per-task=1 \
    --time=00:05:00 \
    "${BASH_SOURCE[0]}" --inside-allocation
fi

if [[ -z "${SLURM_JOB_ID:-}" && -z "${SLURM_JOB_NAME:-}" ]]; then
  echo "Refusing to run the GPU test outside a Slurm allocation." >&2
  exit 2
fi

node_name="$(hostname)"
if [[ "${node_name}" == "hacchead"* ]]; then
  echo "Refusing to run the GPU test on the hacchead login node." >&2
  exit 2
fi

stdout_log="${log_dir}/ans-smoke-${SLURM_JOB_ID}.out"
stderr_log="${log_dir}/ans-smoke-${SLURM_JOB_ID}.err"

echo "Running ${test_filter} on ${node_name} in Slurm job ${SLURM_JOB_ID}."
echo "stdout log: ${stdout_log}"
echo "stderr log: ${stderr_log}"

"${test_bin}" "--gtest_filter=${test_filter}" \
  > >(tee "${stdout_log}") \
  2> >(tee "${stderr_log}" >&2)
