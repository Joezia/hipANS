#pragma once

#include "hipans/ans/GpuANSDecode.h"

namespace hipans {

// Copy one aligned compressed block from global memory into a wave-private
// LDS staging region. The caller publishes the completed writes with one
// workgroup barrier before passing staging to ANSDecodeWarpFullBlock.
inline __device__ void stageCompressedBlockToLds(
    const ANSEncodedT* input, ANSEncodedT* staging,
    uint32_t compressedWords, unsigned lane) {
  constexpr unsigned wordsPerVector = sizeof(uint4) / sizeof(ANSEncodedT);
  const auto* src4 = reinterpret_cast<const uint4*>(input);
  auto* dst4 = reinterpret_cast<uint4*>(staging);
  const unsigned vectorCount = compressedWords / wordsPerVector;
  unsigned base = 0;
  for (; base + 4 * 64 <= vectorCount; base += 4 * 64) {
    const uint4 v0 = src4[base + lane];
    const uint4 v1 = src4[base + 64 + lane];
    const uint4 v2 = src4[base + 128 + lane];
    const uint4 v3 = src4[base + 192 + lane];
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    dst4[base + lane] = v0;
    dst4[base + 64 + lane] = v1;
    dst4[base + 128 + lane] = v2;
    dst4[base + 192 + lane] = v3;
  }
  const unsigned vectorTail = vectorCount - vectorCount % (4 * 64);
  for (unsigned i = vectorTail + lane; i < vectorCount; i += 64)
    dst4[i] = src4[i];
  const unsigned fullWords = vectorCount * wordsPerVector;
  for (unsigned i = fullWords + lane; i < compressedWords; i += 64)
    staging[i] = input[i];
}

} // namespace hipans
