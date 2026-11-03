#!/usr/bin/env bash
# Build the five benchmark binaries behind the six measured points.
#
#   binary                O1 (barrier)   O3 (block size)
#   barrier1_b8192        kept           8 KiB    -> R0
#   barrier0_b8192        removed        8 KiB    -> O1, O1+O2 (O2 is the run-time flag --input lds)
#   barrier0_b4096/2048/1024  removed    4/2/1 KiB -> O1+O2+O3
#
# Usage: scripts/reproduce/build_points.sh   (env: GPU_ARCH=gfx950, JOBS=8, BUILD_ROOT=<repo>/build/reproduce)
set -euo pipefail
src="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
arch="${GPU_ARCH:-gfx950}"
jobs="${JOBS:-8}"
root="${BUILD_ROOT:-${src}/build/reproduce}"
hipcc_bin="$(command -v hipcc)"
rocm_root="$(cd "$(dirname "$(readlink -f "${hipcc_bin}")")/.." && pwd)"
hip_compiler="${rocm_root}/lib/llvm/bin/clang++"
[[ -x "${hip_compiler}" ]] || hip_compiler="${rocm_root}/llvm/bin/clang++"
[[ -x "${hip_compiler}" ]] || { echo "ERROR: HIP clang++ not found under ${rocm_root}" >&2; exit 2; }
mkdir -p "${root}/bin"

for variant in barrier1_b8192:1:8192 barrier0_b8192:0:8192 barrier0_b4096:0:4096 barrier0_b2048:0:2048 barrier0_b1024:0:1024; do
  IFS=: read -r name barrier block <<<"${variant}"
  dir="${root}/${name}"
  cmake -S "${src}" -B "${dir}" -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER="${hipcc_bin}" -DCMAKE_HIP_COMPILER="${hip_compiler}" \
    -DCMAKE_HIP_ARCHITECTURES="${arch}" "-DCMAKE_CXX_FLAGS=--offload-arch=${arch}" \
    "-DCMAKE_HIP_FLAGS=--offload-arch=${arch}" -DBUILD_SHARED_LIBS=OFF \
    -DHIPANS_LIBRARY_TYPE=STATIC -DHIPANS_BUILD_FULL_LIBRARY=OFF \
    -DHIPANS_BLOCK_SIZE="${block}" -DHIPANS_DECODE_WARP_BARRIER="${barrier}" >"${root}/${name}.configure.log" 2>&1
  cmake --build "${dir}" --parallel "${jobs}" --target hipans_decode_bench >"${root}/${name}.build.log" 2>&1 \
    || { tail -30 "${root}/${name}.build.log" >&2; exit 1; }
  cp "${dir}/bin/hipans_decode_bench" "${root}/bin/hipans_decode_bench_${name}"
  echo "built ${root}/bin/hipans_decode_bench_${name}  (barrier=${barrier}, block=${block})"
done
