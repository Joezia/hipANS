#!/usr/bin/env bash
set -euo pipefail

# Submit the smallest ANS GPU round-trip test through Slurm.
#
# Optional:
#   GPU_PARTITION  HACC GPU partition name (default: mi210_u280_u55c)
#   BUILD_DIR      Existing build directory (default: <repo>/build)
#   TEST_FILTER    GoogleTest filter (default: ANSTest.ZeroSized)
#   LOG_DIR        Directory for the Slurm output (default: <repo>/logs)

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

exec srun \
  --job-name=hipans-ans-smoke \
  -p "${gpu_partition}" \
  --cpus-per-task=1 \
  --time=00:05:00 \
  --output="${log_dir}/ans-smoke-%j.out" \
  --error="${log_dir}/ans-smoke-%j.err" \
  "${test_bin}" "--gtest_filter=${test_filter}"
