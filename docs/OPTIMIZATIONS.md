# hipANS decode optimizations (MI350X)

This branch adds three optimizations to the hipANS rANS decoder and a benchmark that
measures each of them in isolation. All three are build- or run-time switches; the
defaults reproduce the upstream behaviour.

| | Optimization | Switch | Where |
|---|---|---|---|
| O1 | Remove the per-iteration workgroup barrier in `decodeOneWarp` | `-DHIPANS_DECODE_WARP_BARRIER=0` (CMake: `-DHIPANS_DECODE_WARP_BARRIER=0`) | `hipans/ans/GpuANSDecode.h` |
| O2 | Stage the compressed block in LDS before decoding | run-time flag `--input lds` of the benchmark | `hipans/ans/GpuANSStageLds.h`, `hipans/bench/hipans_decode_bench.cpp` |
| O3 | Smaller decode block (8 KiB -> 4/2/1 KiB) | `-DHIPANS_BLOCK_SIZE=4096` (or 2048, 1024) | `hipans/ans/GpuANSUtils.h` |

What each one does, in terms of the decode loop (one wave decodes one block, 64 output
bytes per iteration):

* **O1** - `decodeOneWarp` ended its rANS state update with `__syncthreads()` before the
  wave-level `__ballot`. Every wave of a workgroup decodes a different block and shares only
  the read-only lookup table (initialised before the loop), so no cross-wave ordering is
  needed there. Removing it removes one `s_barrier` per iteration.
* **O2** - the 16-bit renormalization loads read the compressed stream from global memory
  inside the loop. `stageCompressedBlockToLds` first copies the block's compressed words to
  a wave-private LDS region (one barrier per block), and the loop then reads LDS. This raises
  LDS use per workgroup (20,480 B -> 34,624 B for 2 waves at 8 KiB blocks) and lowers
  resident workgroups per CU (8 -> 4).
* **O3** - a block is `HIPANS_BLOCK_SIZE` bytes (`kDefaultBlockSize`). Smaller blocks need
  less LDS per wave, so more waves fit on a CU at once (with O2 staging: 1,024 resident
  workgroups on the GPU at 8 KiB, 2,048 at 4 KiB, 3,328 at 2 KiB, 4,096 at 1 KiB). The cost is
  more per-block metadata (one rANS state per lane per block), so the archive grows
  (58.97 MB at 8 KiB, 60.66 MB at 4 KiB, 64.08 MB at 2 KiB, 70.58 MB at 1 KiB for the test
  matrix). The encoder and decoder must use the same value.

## The benchmark

`hipans_decode_bench` decodes a native hipANS archive with the library's own decoder
(`ANSDecodeWarpFullBlock`) and writes the decoded bytes to LDS (not back to global memory),
which is what a fused decode+GEMM kernel consumes. It can also create the archive with the
library encoder (`--prepare`) and verify the decoded tile byte for byte against the input.

## Reproducing the MI350X results

Requirements: ROCm with `hipcc`, CMake, Python 3 with `requests`, one idle MI350X (gfx950).
Profiling additionally needs `rocprof-compute` and `rocprofv3`.

```bash
# 1. build the five binaries behind the six points (about 5-10 min)
scripts/reproduce/build_points.sh

# 2. download the 64 MiB test matrix (one expert of Qwen3.8-2.4T-A95B, layer 46); SHA-256 is checked
python3 scripts/reproduce/fetch_weights.py --output inputs/qwen38-expert0

# 3. throughput of the six points: archives, byte-exact check, 3 processes per point
scripts/reproduce/run_throughput.sh inputs/qwen38-expert0/weights.bin out/throughput

# 4. profiler data and per-iteration cycle accounting
scripts/reproduce/profile.sh inputs/qwen38-expert0/weights.bin out/throughput out/profile
python3 scripts/reproduce/analyze_cycles.py --profile-dir out/profile \
  --bin-dir build/reproduce/bin --throughput-json out/throughput/summary.json --output out/cycles.json
```

Step 3 prints your numbers next to the published ones (`scripts/reproduce/published_mi350x.json`).

## Notes on the profiler data

* Cycles per iteration (from `SQ_WAVE_CYCLES`) and IPC were stable to within 0.5% across runs.
* The core clock, average resident waves and VALU utilization divide by `GRBM_GUI_ACTIVE`.
  rocprof-compute collects counters in several passes, and in one of three profiling runs this
  counter was mis-recorded for the 1 KiB point in one of four profiling runs (resident waves 12,498 against a hardware limit of
  8,192). `analyze_cycles.py` prints a warning when that happens; rerun `profile.sh`.
* VALU utilization is `SQ_ACTIVE_INST_VALU / (GRBM_GUI_ACTIVE x CUs)` and is not capped at 100%.

## Published results (MI350X, gfx950)

Test matrix: Qwen3.8-2.4T-A95B, layer 46, expert 0, `gate_up_proj`, bf16, 4,096 x 16,384 bytes
(64 MiB). 16 replicas of the compressed matrix (0.94-1.13 GB, above the 256 MB Infinity
Cache), 128 threads per workgroup, 10 warm-up + 50 timed launches per process, median of the
50 launches, median of 3 independent processes. Throughput = decoded bytes / kernel time of one
launch (HIP events). One iteration = one wave decoding 64 output bytes, so a dispatch has
16 x 67,108,864 / 64 = 16,777,216 iterations; cycles per iteration = `Wave Cycles` (rocprof-compute
metric 7.2.3, summed over all waves) / 16,777,216.

| Point | Block | Compressed input | Archive | Cycles / iteration | Throughput |
|---|---|---|---|---|---|
| R0 (upstream) | 8 KiB | global memory | 58.97 MB | 1,092.2 | 476.7 GB/s |
| O1 | 8 KiB | global memory | 58.97 MB | 1,039.3 | 500.6 GB/s |
| O1 + O2 | 8 KiB | staged in LDS | 58.97 MB | 375.7 | 692.4 GB/s |
| O1 + O2 + O3 | 4 KiB | staged in LDS | 60.66 MB | 447.0 | 1,094.0 GB/s |
| O1 + O2 + O3 | 2 KiB | staged in LDS | 64.08 MB | 517.3 | 1,381.1 GB/s |
| O1 + O2 + O3 | 1 KiB | staged in LDS | 70.58 MB | 628.0 | 1,480.2 GB/s |

Three further runs of the whole pipeline on the same machine (two of them from a fresh clone of
this branch, including the data download) reproduced the throughput within 1.2% and the cycles
per iteration within 0.4% of these values; the archives were byte-identical in every run. The
largest throughput spread is at 1 KiB, where the three processes of one run differ by up to 3.3%.
