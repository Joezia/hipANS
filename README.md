# hipANS: Portable Lossless Compression of Numerical Data for GPU Architectures

Developer: Jinwu Yang, Yida Gu, Dingwen Tao @ Institute of Computing Technology, Chinese Academy of Sciences

hipANS is a GPU library implemented in the HIP programming language for fast, specialized lossless data compression, designed for ML and HPC applications. It also includes the **first** publicly available generalized [asymmetric numeral system (ANS)](https://en.wikipedia.org/wiki/Asymmetric_numeral_systems) compressor and decompressor, compatible with various GPU architectures, including AMD GPUs. It is a GPU analogue to Yann Collet's [FSE (Finite State Entropy)](https://github.com/Cyan4973/FiniteStateEntropy) ANS library. It has been primarily ported and optimized from Facebook's [dietGPU](https://github.com/facebookresearch/dietgpu) library.

It currently consists of two parts:

- **ANS entropy codec**: a generalized byte-oriented range-based ANS (rANS) entropy encoder and decoder, that operates at throughputs around 50-150 GB/s for reasonable data sizes on an MI100/MI210 GPU.
- **Floating point codec**: an extension to the above to handle fast lossless compression and decompression of unstructured floating point data, for use in ML and HPC applications, especially in communicating over local interconnects (PCIe / NVLink) and remote interconnects (Ethernet / InfiniBand). This operates at around 50-300 GB/s for reasonable data sizes on an MI100/MI210 GPU.

Both APIs are available in both C++ (raw device pointers) and Python/PyTorch (PyTorch tensor) API forms.

## Decode optimizations for AMD Instinct MI350X

This fork includes three switchable optimizations and a standalone reproduction workflow:

| Point | Per-symbol workgroup barrier | Compressed input | Decode block |
|---|---|---|---|
| R0 | Kept | Global memory | 8 KiB |
| O1 | Removed | Global memory | 8 KiB |
| O1 + O2 | Removed | Staged in LDS | 8 KiB |
| O1 + O2 + O3 | Removed | Staged in LDS | 4 / 2 / 1 KiB |

O1 uses `HIPANS_DECODE_WARP_BARRIER=0`, O3 uses `HIPANS_BLOCK_SIZE`, and O2
uses the benchmark's `--input lds` option. Defaults retain the original barrier
and 8 KiB block size. O2 is implemented in the benchmark, not enabled by the
standard `ansDecodeBatchPointer` API. Only this repository is needed; input
weights are downloaded separately by the supplied script.

### Environment

The reference results use **one AMD Instinct MI350X, gfx950, wave64, 256 CUs**
on Linux. The latest local reproduction used CMake 3.28.3 and ROCm's Clang
23.0.0, as recorded by its CMake compiler configuration. Load a ROCm toolchain
supporting gfx950, with `hipcc` and `cmake` on `PATH`; data download requires
Python 3.9+ and `requests`. Throughput measurement does not require a profiler.
Optional cycle analysis requires `rocprof-compute`, `rocprofv3` with host-trap
PC sampling, and ROCm LLVM tools. Other GPUs/toolchains can give different results.

### Reproduce all six points

```bash
git clone --recursive https://github.com/Joezia/hipANS.git
cd hipANS

# 1. Compile five binaries covering all six points.
GPU_ARCH=gfx950 JOBS=8 bash scripts/reproduce/build_points.sh

# 2. Download a fixed 64 MiB BF16 weight slice; verifies SHA-256.
python3 scripts/reproduce/fetch_weights.py --output inputs/qwen38-expert0

# 3. Create archives, verify every point byte for byte, then measure all points.
bash scripts/reproduce/run_throughput.sh \
  inputs/qwen38-expert0/weights.bin out/throughput
```

Run these commands in order, proceeding only if each succeeds. Steps 1 and 2
need no GPU; step 3 needs an allocated, otherwise idle GPU. On a shared Slurm
cluster, submit them as non-interactive jobs from the login node. For example,
on the cluster used for these measurements:

```bash
srun -p devel -N1 -n1 -c8 -t 00:20:00 \
  env GPU_ARCH=gfx950 JOBS=8 bash scripts/reproduce/build_points.sh
srun -p devel -N1 -n1 -c2 -t 00:10:00 \
  python3 scripts/reproduce/fetch_weights.py --output inputs/qwen38-expert0
srun -p mi3501x --gres=gpu:1 -N1 -n1 -c2 -t 00:30:00 \
  bash scripts/reproduce/run_throughput.sh \
  inputs/qwen38-expert0/weights.bin out/throughput
```

Partition names and GPU resource flags are cluster-specific. Use your site's
approved equivalents. The reproduction scripts do not submit Slurm jobs
themselves. Source and tool paths are discovered at runtime; there is no
dependency on the author's home/work directory. `BUILD_ROOT`, `BIN_DIR` and
`LLVM_BIN` can override build output, benchmark binaries and analysis tools.
If overriding `BUILD_ROOT`, set `BIN_DIR` to its `bin` subdirectory when running.

Input and output directories must not already exist. On subsequent runs, reuse
the input and binaries and choose a fresh result directory, e.g. `out/throughput-2`.
Generated inputs, build files, logs and results are ignored by Git.

### Check the results

The script prints three process throughputs per point, their median, and the
percentage difference from the reference in
[`published_mi350x.json`](scripts/reproduce/published_mi350x.json).
Each process uses 10 warm-up launches and the median of 50 timed launches;
the reported point is the median of the three process throughputs.

- `out/throughput/summary.json`: six results and archive SHA comparisons.
- `out/throughput/correctness/point-0.json` through `point-5.json`: all must
  report `verified_actual_LDS: true`. A mismatch exits the workflow with an error.
- `out/throughput/timing/point-*-run-*.json`: per-process times, throughput,
  GPU architecture, build commit and resource usage. These separate timing
  processes use `--skip-verify`; correctness is checked by the preceding runs.
- `archive_identical_to_published` in the summary: all four values should be
  `true` for the fixed input and reference archive format.

Reference throughput is 476.7, 500.6, 692.4, 1094.0, 1381.1 and 1480.2 GB/s
for R0 through the 1 KiB point. One local reproduction (job 125781) differed
by -0.30% to +0.49%; hardware, clocks, contention and compiler changes affect
the numbers. Consult [the optimization notes](docs/OPTIMIZATIONS.md) for full
configuration, reference results and optional profiling commands.

### What this throughput measures

`effective_LDS_GBps` is **uncompressed bytes decoded into LDS / median GPU
kernel seconds / 1e9**. The fixed matrix is replicated at independent GPU
addresses (16 replicas for the reference workload). All six points, including
R0, use the same LDS-output benchmark. Input staging for O2 and workgroup LUT
loading are inside the timed kernel. Global decode-table construction,
compression, transfers, correctness checks and profiling are outside timing.

The original README's 50-150 GB/s ANS and 50-300 GB/s float-codec figures refer
to the original MI100/MI210 codec workloads. This new experiment uses MI350X,
a fixed bytewise BF16 input, and **LDS output rather than decoded output written
back to global memory**. It is a kernel measurement, not full
`ansDecodeBatchPointer` API latency or measured HBM bandwidth. The historical
figures do not establish an equivalent timing protocol, so a direct speedup
against them would be misleading. Compare R0 and the optimization points
within this new benchmark instead.

The reason for decoding into LDS is **future kernel fusion**: a consumer such
as GEMM could use decoded tiles immediately in the same workgroup, avoiding
an intermediate decompressed tensor written to HBM and read back by the next
kernel. This benchmark measures that decode stage only; it does not execute
GEMM or demonstrate an end-to-end fused-kernel speedup.

## Building

Clone this repo using

```shell
git clone --recursive https://github.com/Joezia/hipANS.git
```

Do the standard CMake thing:

```shell
cd hipANS; mkdir build; cd build;
cmake .. -G Ninja
cmake --build . --target all
```

If you get complaints about `/hipANS/third_party/glog... does not contain a CMakeLists.txt file.` then you didn't pull the submodules; run

```shell
git submodule sync
git submodule update --init --recursive --jobs 0
```
and try again.

## Library rationale

As on-device global memory / HBM bandwidth continues to improve at a faster rate than CPU/GPU interconnect or server-to-server networking bandwidth, spending GPU compute and gmem bandwidth to save on data sent over interconnects is becoming more advantageous. HipANS aims to target this gap.

One can imagine a Pareto-optimal tradeoff curve between realizable compression ratios versus speed. On one end of the curve exists algorithms for supporting arbitrary data using dictionary/LZ type compression like some of the techniques in [Nvidia's nvCOMP](https://github.com/NVIDIA/nvcomp) at potentially high compression rates. At another end of the curve, one can imagine use completely on-device as something like a 1990s-style virtual RAM extender, where achievable compression is only 0.6x-0.9x or so, but compression can operate at around 1/4x to 1/2x the peak global memory bandwidth of the GPU. We emphasize the latter, where speed rather than compression ratio is important, where we can compress data that is even sent between GPUs in a single server over NVLink or PCIe. The savings may be low, but the effective network speed can be increased by 10-30%. For large-scale neural network training on hundreds of GPUs, this could translate into an additional 5-10% end-to-end performance increase.

The initial focus of this library will be in HPC/ML distributed collective communications libraries, for primitives such as all-to-all, all-gather, reduce-scatter and all-reduce. Right now no off the shelf integration is provided (in progress), but the basics of the C++ API are available for use, as are Python-level PyTorch tensor-based APIs.

## ANS codec

The rANS codec operates on 8 bit bytes. It can compress arbitrary data, but using statistics gathered on a bytewise basis, so data highly structured or redundant at a level above byte level will typically not compress well. This codec however is meant to be applicable for any number of lossless compression applications, including usage as an entropy coder for LZ or RLE type matches for a fully-formed compression system. Symbol probability precisions supported are 9, 10 and 11 bits (i.e., symbol occurances are quantized to the nearest 1/512, 1/1024 or 1/2048).

## Float codec

The floating point compressor at the moment uses the rANS codec to handle compression of floating point exponents, as typically in ML/HPC data a very limited exponent dynamic range is used and is highly compressible. Floating point sign and significand values tend to be less compressible / fairly high entropy in practice, though sparse data or presence of functions like ReLU in neural networks can result in a lot of outright zero values which are very compressible. A future extension to the library will allow for specialized compression of sparse or semi-sparse data, specializing compression of zeros. At the moment only float16 (IEEE 754 binary16) and bfloat16 (fields of the most significant 16 bits of a IEEE 754 binary32 word) are supported, with float32 (IEEE 754 binary32) support coming shortly.

## API design

The design fundamentals, originally established for the CUDA-based DietGPU library targeting CC 3.5+ (Kepler class) GPUs or later, have been effectively adapted for compatibility with the ROCm platform, specifically focusing on MI100/MI210 GPUs. This strategic porting, now named hipANS, ensures that the library leverages the advanced capabilities of AMD GPUs while retaining the efficiency and performance optimized for V100/A100 GPUs.

The library's APIs have been expertly ported to maintain their batch-oriented approach, available in both C++ and Python/PyTorch forms. This ensures that operations with raw pointers in C++ and tensor operations in PyTorch are both supported, facilitating a seamless transition for developers across platforms. The batch processing capability allows for the compression and decompression of multiple independent data arrays, with the stipulation that when using the floating-point compressor, all arrays in a batch must share the same data type. The ANS compression symbol probabilities are calculated on a per-array basis within the batch, ensuring that each compressed tensor produced is independently decompressible, with ANS statistics customized for each array. 

Despite the batch-oriented design, the library is also optimized for scenarios where a single large batch size is provided, offering commendable performance. It is worth noting that for substantial data sizes, a batch size (bs) of 1 may currently outperform larger batch sizes due to work imbalance issues. The library handles arrays of varying sizes, treating all input data as unstructured one-dimensional arrays, thereby abstracting away the need for PyTorch API to focus on data dimensionality.

The compression process operates on 4 KiB segments of the input data, allocated to individual warps. It is generally recommended to have at least 512 KiB of data to use hipANS effectively, given the overheads associated with compression. For optimal performance, the total data size should be significant enough to match the number of concurrently running warps capable of fully utilizing the GPU's streaming multiprocessors (SMs).

All computations within the library are device-side, with a design emphasis on minimizing memory allocations and deallocations, as well as reducing unnecessary device-to-host and host-to-device interactions and synchronizations. With properly sized inputs and outputs and sufficient pre-allocated temporary memory, compression and decompression can proceed asynchronously on the GPU without the need for CPU intervention.

However, during the compression process, only the GPU is aware of the actual final compressed size. Therefore, it is essential for applications to transfer the output size buffer, detailing the final compressed sizes for each job in the batch, back to the host. This facilitates the management of compressed data, whether for relocation in local memory or transmission over a network. Since the final output size is unpredictable, the library includes a function to estimate the maximum possible compressed output size, aiding in the allocation of appropriately sized memory regions for output. For applications seeking to realize compression savings beyond networking, additional memory allocation and data transfer to a precisely sized new buffer will be necessary.

This transition of the DietGPU library from the CUDA platform to the ROCm platform, resulting in hipANS, marks a significant advancement in providing a versatile and high-performance compression solution for a broader range of HPC applications.

## References

Prior GPU-based ANS implementations [to my knowledge](https://encode.su/threads/2078-List-of-Asymmetric-Numeral-Systems-implementations) include:

- [GST: GPU-decodable Supercompressed Textures](https://gamma.cs.unc.edu/GST/) (not a generalized ANS codec; meant as part of a texture compression scheme)
- Weissenberger and Schmidt, [Massively Parallel ANS Decoding on GPUs](https://dl.acm.org/doi/10.1145/3337821.3337888) (a decoder only)

Related GPU entropy coder works include:

- Yamamoto et al., [Huffman Coding with Gap Arrays for GPU Acceleration](https://dl.acm.org/doi/10.1145/3404397.3404429)

Related lossless floating point compression works include:

- Lindstrom and Isenburg, [Fast and Efficient Compression of Floating-Point Data](https://computing.llnl.gov/projects/fpzip) (CPU-based)
- Various GPU works from Martin Burtscher's group at Texas State such as Yang et al., [MPC: A Massively Parallel Compression Algorithm for Scientific Data](https://www.semanticscholar.org/paper/MPC%3A-A-Massively-Parallel-Compression-Algorithm-for-Yang-Mukka/1ab6910c90ad714e29954ccd69d569eb2003eb20)

These works are sometimes oriented at compressing HPC-type data (e.g., 2d/3d/Nd grid data) where there may be local/dimensional correlations that can be exploited.

- [nvCOMP](https://github.com/NVIDIA/nvcomp), Nvidia's GPU lossless compression library.
- [dietGPU](https://github.com/facebookresearch/dietgpu), DietGPU: GPU-based lossless compression for numerical data.

## License

hipANS is licensed with the MIT license, available in the LICENSE file at the top level.
