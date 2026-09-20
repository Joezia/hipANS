#!/usr/bin/env bash
set -euo pipefail

# Configure hipANS and build only build/bin/ans_test through Slurm.
#
# Optional overrides:
#   BUILD_PARTITION  ROCm-enabled partition (default: mi210_u280_u55c)
#   BUILD_DIR        Build directory (default: <repo>/build)
#   BUILD_CPUS       CPU cores used by the build (default: 4)
#   BUILD_TIME       Slurm time limit (default: 00:30:00)
#   GPU_ARCH         HIP target architecture (default: gfx90a for MI210)
#   CMAKE_BIN        Absolute path to cmake, if it is not on PATH
#   HIPCC_BIN        Absolute path to hipcc, if it is not on PATH
#   HIP_COMPILER_BIN Absolute path to ROCm clang++ for CMake's HIP language

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-${repo_dir}/build}"
log_dir="${LOG_DIR:-${repo_dir}/logs}"
build_partition="${BUILD_PARTITION:-mi210_u280_u55c}"
build_cpus="${BUILD_CPUS:-4}"
build_time="${BUILD_TIME:-00:30:00}"
gpu_arch="${GPU_ARCH:-gfx90a}"

if [[ "${1:-}" != "--inside-allocation" ]]; then
  mkdir -p "${log_dir}"

  exec srun \
    --job-name=hipans-build-ans-test \
    -p "${build_partition}" \
    --cpus-per-task="${build_cpus}" \
    --time="${build_time}" \
    --output="${log_dir}/build-ans-test-%j.out" \
    --error="${log_dir}/build-ans-test-%j.err" \
    "${BASH_SOURCE[0]}" --inside-allocation
fi

if [[ -z "${SLURM_JOB_ID:-}" && -z "${SLURM_JOB_NAME:-}" ]]; then
  echo "Refusing to build outside a Slurm allocation." >&2
  exit 2
fi

node_name="$(hostname)"
if [[ "${node_name}" == "hacchead"* ]]; then
  echo "Refusing to build on the hacchead login node." >&2
  exit 2
fi

cmake_bin="${CMAKE_BIN:-}"
if [[ -z "${cmake_bin}" ]]; then
  cmake_bin="$(command -v cmake || true)"
fi
if [[ -z "${cmake_bin}" || ! -x "${cmake_bin}" ]]; then
  echo "cmake is unavailable on ${node_name}." >&2
  echo "Activate a user environment containing cmake, or set CMAKE_BIN to its absolute path." >&2
  echo "Do not use sudo on HACC." >&2
  exit 2
fi

hipcc_bin="${HIPCC_BIN:-}"
if [[ -z "${hipcc_bin}" ]]; then
  hipcc_bin="$(command -v hipcc || true)"
fi
if [[ -z "${hipcc_bin}" && -x /opt/rocm/bin/hipcc ]]; then
  hipcc_bin=/opt/rocm/bin/hipcc
fi
if [[ -z "${hipcc_bin}" || ! -x "${hipcc_bin}" ]]; then
  echo "hipcc is unavailable on ${node_name}." >&2
  echo "The selected compile node does not expose the HACC ROCm/HIP environment." >&2
  echo "Do not install ROCm yourself; select an approved ROCm-enabled partition or ask the administrator." >&2
  exit 2
fi

hip_compiler_bin="${HIP_COMPILER_BIN:-}"
if [[ -z "${hip_compiler_bin}" && -x /opt/rocm/llvm/bin/clang++ ]]; then
  hip_compiler_bin=/opt/rocm/llvm/bin/clang++
fi
if [[ -z "${hip_compiler_bin}" ]]; then
  hip_compiler_bin="$(command -v clang++ || true)"
fi
if [[ -z "${hip_compiler_bin}" || ! -x "${hip_compiler_bin}" ]]; then
  echo "ROCm clang++ is unavailable on ${node_name}." >&2
  echo "Set HIP_COMPILER_BIN to the ROCm clang++ executable provided by HACC." >&2
  exit 2
fi

echo "Building ans_test on ${node_name} in Slurm job ${SLURM_JOB_ID:-unknown}."
echo "CMake: ${cmake_bin}"
echo "HIP compiler: ${hipcc_bin}"
echo "CMake HIP compiler: ${hip_compiler_bin}"
echo "HIP architecture: ${gpu_arch}"

"${cmake_bin}" \
  --fresh \
  -S "${repo_dir}" \
  -B "${build_dir}" \
  -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="${hipcc_bin}" \
  -DCMAKE_HIP_COMPILER="${hip_compiler_bin}" \
  -DCMAKE_HIP_ARCHITECTURES="${gpu_arch}" \
  -DBUILD_SHARED_LIBS=OFF \
  -DHIPANS_LIBRARY_TYPE=STATIC \
  -DHIPANS_BUILD_FULL_LIBRARY=OFF

"${cmake_bin}" --build "${build_dir}" --target ans_test --parallel "${build_cpus}"

test_bin="${build_dir}/bin/ans_test"
if [[ ! -x "${test_bin}" ]]; then
  echo "Build command completed but ${test_bin} was not created." >&2
  exit 1
fi

echo "Built ${test_bin}"
