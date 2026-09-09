#pragma once
#include <poplar/HalfFloat.hpp>
#include <poplar/QuarterFloat.hpp>
using namespace poplar;

// Keep each converted pair in a spare ARF register. Conversion of the
// first pair overlaps the second input load; no eight-value staging is needed.
static inline void normFp8Quads(const half *input, const half *scale, const half *bias,
                                unsigned char *output, unsigned rounds,
                                const float *constants) {
  asm volatile(
      "ld32 $a6, %[constants], $mzero, 0\n"
      "ld32 $a7, %[constants], $mzero, 1\n"
      "{ rpt %[rounds], 20; fnop }\n"
      "{ ld32step $a0, $mzero, %[in]+=, 1; fnop }\n"
      "{ nop; f16v2tof32 $a0:1, $a0 }\n"
      "{ nop; f32v2add $a0:1, $a6:B, $a0:1 }\n"
      "{ ld32step $a2, $mzero, %[scale]+=, 1; f32v2mul $a0:1, $a7:B, $a0:1 }\n"
      "{ nop; f16v2tof32 $a2:3, $a2 }\n"
      "{ ld32step $a4, $mzero, %[bias]+=, 1; f32v2mul $a0:1, $a0:1, $a2:3 }\n"
      "{ nop; f16v2tof32 $a2:3, $a4 }\n"
      "{ nop; f32v2add $a0:1, $a0:1, $a2:3 }\n"
      "{ nop; f32v2tof16 $a5, $a0:1 }\n"
      "{ ld32step $a0, $mzero, %[in]+=, 11; f16v2tof8 $a5, $a5 }\n"
      "{ nop; f16v2tof32 $a0:1, $a0 }\n"
      "{ nop; f32v2add $a0:1, $a6:B, $a0:1 }\n"
      "{ ld32step $a2, $mzero, %[scale]+=, 11; f32v2mul $a0:1, $a7:B, $a0:1 }\n"
      "{ nop; f16v2tof32 $a2:3, $a2 }\n"
      "{ ld32step $a4, $mzero, %[bias]+=, 11; f32v2mul $a0:1, $a0:1, $a2:3 }\n"
      "{ nop; f16v2tof32 $a2:3, $a4 }\n"
      "{ nop; f32v2add $a0:1, $a0:1, $a2:3 }\n"
      "{ nop; f32v2tof16 $a0, $a0:1 }\n"
      "{ nop; f16v2tof8 $a0, $a0 }\n"
      "{ nop; sort4x16lo $a0, $a5, $a0 }\n"
      "{ st32step $a0, $mzero, %[out]+=, 6; fnop }\n"
      : [in] "+&r"(input), [scale] "+&r"(scale), [bias] "+&r"(bias), [out] "+&r"(output)
      : [rounds] "r"(rounds), [constants] "r"(constants)
      : "$a0:1", "$a2:3", "$a4:5", "$a6:7", "memory");
}

static inline void normApplyFp8(const half *input, const half *scale, const half *bias,
                                unsigned char *output, unsigned width, unsigned worker,
                                float mean, float inverse, unsigned row, unsigned rows,
                                bool packed) {
  const float constants[2] = {-mean, inverse};
  if (!packed || rows == 1) {
    if (worker < width / 4)
      normFp8Quads(input + worker * 4, scale + worker * 4, bias + worker * 4,
                   output + row * width + worker * 4, (width / 4 + 5 - worker) / 6, constants);
  } else {
    for (unsigned column = worker * 4; column < width; column += 24) {
      const unsigned offset = (column / 32 * rows + row) * 32 + column % 32;
      normFp8Quads(input + column, scale + column, bias + column,
                   output + offset, 1, constants);
    }
  }
  if (packed) {
    for (unsigned column = width + worker * 4; column < (width + 31) / 32 * 32; column += 24) {
      const unsigned offset = (column / 32 * rows + row) * 32 + column % 32;
      *reinterpret_cast<unsigned *>(output + offset) = 0;
    }
  }
}
