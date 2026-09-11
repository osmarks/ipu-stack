#pragma once
#include <poplar/HalfFloat.hpp>
using namespace poplar;

// Six workers traverse complete half pairs. Keep descriptors and address
// arithmetic outside the repeated bodies; tails never read a following pair.
static inline void addQuads(const half *left, const half *right, half *output,
                            unsigned quads, unsigned worker) {
  if (worker >= quads) return;
  left += worker * 4;
  right += worker * 4;
  output += worker * 4;
  const unsigned rounds = (quads + 5 - worker) / 6;
  asm volatile(
      "{ rpt %[rounds], 3; fnop }\n"
      "{ ld64step $a0:1, $mzero, %[left]+=, 6; fnop }\n"
      "{ ld64step $a2:3, $mzero, %[right]+=, 6; fnop }\n"
      "{ nop; f16v4add $a0:1, $a0:1, $a2:3 }\n"
      "{ st64step $a0:1, $mzero, %[out]+=, 6; fnop }\n"
      : [left] "+&r"(left), [right] "+&r"(right), [out] "+&r"(output)
      : [rounds] "r"(rounds)
      : "$a0:1", "$a2:3", "memory");
}

#ifdef NORM_WITH_ADD
#define NORM_LOAD_RIGHT \
  "{ ld32step $a2, $mzero, %[right]+=, 6; fnop }\n" \
  "{ nop; f16v2add $a0, $a0, $a2 }\n"
#define NORM_EXTRA_BUNDLES 2
#else
#define NORM_LOAD_RIGHT ""
#define NORM_EXTRA_BUNDLES 0
#endif

// Accumulate aligned groups directly in FP32 AACC registers. Centering
// remains FP32; only the sum pass consumes halves directly. GINA reads the
// active lanes before any later kernel can reuse the accumulators.
static inline float2 normSumWide(const half *input, const half *right,
                                 unsigned width, unsigned worker, float mean,
                                 bool centered) {
  float2 sum;
  const unsigned group = centered ? 4 : 8;
  input += worker * group;
#ifdef NORM_WITH_ADD
  right += worker * group;
#endif
  const unsigned rounds = (width / group + 5 - worker) / 6;
  if (centered) {
    asm volatile(
        "setzi $a0, (1 << 3)\n"
        "uput $FP_CLR, $a0\n"
        "mov $a6, %[mean]\n"
        "{ rpt %[rounds], %[bundles]; fnop }\n"
        "{ ld64step $a4:5, $mzero, %[in]+=, 6; fnop }\n"
#ifdef NORM_WITH_ADD
        "{ ld64step $a0:1, $mzero, %[right]+=, 6; fnop }\n"
        "{ nop; f16v4add $a4:5, $a4:5, $a0:1 }\n"
#endif
        "{ nop; f16v2tof32 $a0:1, $a4 }\n"
        "{ nop; f16v2tof32 $a2:3, $a5 }\n"
        "{ nop; f32v2add $a0:1, $a6:B, $a0:1 }\n"
        "{ nop; f32v2add $a2:3, $a6:B, $a2:3 }\n"
        "{ nop; f32v4sqacc $a0:3 }\n"
        "f32v2gina $a0:1, $azeros, 0\n"
        "f32v2gina $a2:3, $azeros, 0\n"
        "f32v2add $a0:1, $a0:1, $a2:3\n"
        "st64 $a0:1, %[sum], $mzero, 0\n"
        : [in] "+&r"(input), [right] "+&r"(right)
        : [sum] "r"(&sum), [rounds] "r"(rounds), [mean] "r"(-mean), [bundles] "i"(5 + NORM_EXTRA_BUNDLES)
        : "$a0:1", "$a2:3", "$a4:5", "$a6", "memory");
  } else {
    asm volatile(
        "setzi $a0, (1 << 3)\n"
        "uput $FP_CLR, $a0\n"
        "{ rpt %[rounds], %[bundles]; fnop }\n"
        "{ ld64step $a0:1, $mzero, %[in]+=, 1; fnop }\n"
        "{ ld64step $a2:3, $mzero, %[in]+=, 11; fnop }\n"
#ifdef NORM_WITH_ADD
        "{ ld64step $a4:5, $mzero, %[right]+=, 1; fnop }\n"
        "{ ld64step $a6:7, $mzero, %[right]+=, 11; f16v4add $a0:1, $a0:1, $a4:5 }\n"
        "{ nop; f16v4add $a2:3, $a2:3, $a6:7 }\n"
#endif
        "{ nop; f16v8acc $a0:3 }\n"
        "f32v2gina $a0:1, $azeros, 0\n"
        "f32v2gina $a2:3, $azeros, 0\n"
        "f32v2add $a0:1, $a0:1, $a2:3\n"
        "f32v2gina $a2:3, $azeros, 0\n"
        "f32v2add $a0:1, $a0:1, $a2:3\n"
        "f32v2gina $a2:3, $azeros, 0\n"
        "f32v2add $a0:1, $a0:1, $a2:3\n"
        "st64 $a0:1, %[sum], $mzero, 0\n"
        : [in] "+&r"(input), [right] "+&r"(right)
        : [sum] "r"(&sum), [rounds] "r"(rounds), [bundles] "i"(2 + NORM_EXTRA_BUNDLES * 3 / 2)
        : "$a0:1", "$a2:3", "$a4:5", "$a6:7", "memory");
  }
  return sum;
}

static inline float2 normSum(const half *input, const half *right,
                             unsigned width, unsigned worker, float mean,
                             bool centered) {
  if (width >= 96 && !(width % 8) &&
      !((reinterpret_cast<unsigned>(input) | reinterpret_cast<unsigned>(right)) & 7))
    return normSumWide(input, right, width, worker, mean, centered);
  float2 sum = {0, 0};
  if (worker >= width / 2) return sum;
  input += worker * 2;
#ifdef NORM_WITH_ADD
  right += worker * 2;
#endif
  const unsigned rounds = (width / 2 + 5 - worker) / 6;
  if (centered) {
    asm volatile(
        "mov $a4, %[mean]\n"
        "{ rpt %[rounds], %[bundles]; fnop }\n"
        "{ ld32step $a0, $mzero, %[in]+=, 6; fnop }\n"
        NORM_LOAD_RIGHT
        "{ nop; f16v2tof32 $a0:1, $a0 }\n"
        "{ nop; f32v2add $a0:1, $a4:B, $a0:1 }\n"
        "{ nop; f32v2mul $a0:1, $a0:1, $a0:1 }\n"
        "{ nop; f32v2add %[sum], %[sum], $a0:1 }\n"
        : [in] "+&r"(input), [right] "+&r"(right), [sum] "+r"(sum)
        : [rounds] "r"(rounds), [mean] "r"(-mean), [bundles] "i"(4 + NORM_EXTRA_BUNDLES)
        : "$a0:1", "$a2", "$a4", "memory");
  } else {
    asm volatile(
        "{ rpt %[rounds], %[bundles]; fnop }\n"
        "{ ld32step $a0, $mzero, %[in]+=, 6; fnop }\n"
        NORM_LOAD_RIGHT
        "{ nop; f16v2tof32 $a0:1, $a0 }\n"
        "{ nop; f32v2add %[sum], %[sum], $a0:1 }\n"
        : [in] "+&r"(input), [right] "+&r"(right), [sum] "+r"(sum)
        : [rounds] "r"(rounds), [bundles] "i"(2 + NORM_EXTRA_BUNDLES)
        : "$a0:1", "$a2", "memory");
  }
  return sum;
}

static inline float normInverse(float variance) {
  float result;
  asm("f32oorx %[out], %[in]" : [out] "=r"(result) : [in] "r"(variance));
  return result;
}

static inline void normApply(const half *input, const half *right,
                            const half *scale, const half *bias, half *output,
                            unsigned width, unsigned worker, float mean, float inverse) {
  // Keep centering and normalization in FP32. Scale/bias are FP16 model
  // parameters: applying them four-wide avoids widening both on every pair.
  if (!(width % 4) && !((reinterpret_cast<unsigned>(input) |
      reinterpret_cast<unsigned>(right) | reinterpret_cast<unsigned>(scale) |
      reinterpret_cast<unsigned>(bias) | reinterpret_cast<unsigned>(output)) & 7)) {
    if (worker >= width / 4) return;
    input += worker * 4;
#ifdef NORM_WITH_ADD
    right += worker * 4;
#endif
    scale += worker * 4;
    bias += worker * 4;
    output += worker * 4;
    const unsigned rounds = (width / 4 + 5 - worker) / 6;
    const float constants[2] = {-mean, inverse};
    asm volatile(
        "ld32 $a6, %[constants], $mzero, 0\n"
        "ld32 $a7, %[constants], $mzero, 1\n"
        "{ rpt %[rounds], %[bundles]; fnop }\n"
        "{ ld64step $a4:5, $mzero, %[in]+=, 6; fnop }\n"
#ifdef NORM_WITH_ADD
        "{ ld64step $a0:1, $mzero, %[right]+=, 6; fnop }\n"
        "{ nop; f16v4add $a4:5, $a4:5, $a0:1 }\n"
#endif
        "{ nop; f16v2tof32 $a0:1, $a4 }\n"
        "{ nop; f16v2tof32 $a2:3, $a5 }\n"
        "{ nop; f32v2add $a0:1, $a6:B, $a0:1 }\n"
        "{ nop; f32v2add $a2:3, $a6:B, $a2:3 }\n"
        "{ nop; f32v2mul $a0:1, $a7:B, $a0:1 }\n"
        "{ nop; f32v2mul $a2:3, $a7:B, $a2:3 }\n"
        "{ ld64step $a4:5, $mzero, %[scale]+=, 6; f32v2tof16 $a0, $a0:1 }\n"
        "{ ld64step $a2:3, $mzero, %[bias]+=, 6; f32v2tof16 $a1, $a2:3 }\n"
        "{ nop; f16v4mul $a0:1, $a0:1, $a4:5 }\n"
        "{ nop; f16v4add $a0:1, $a0:1, $a2:3 }\n"
        "{ st64step $a0:1, $mzero, %[out]+=, 6; fnop }\n"
        : [in] "+&r"(input), [right] "+&r"(right), [scale] "+&r"(scale),
          [bias] "+&r"(bias), [out] "+&r"(output)
        : [rounds] "r"(rounds), [constants] "r"(constants),
          [bundles] "i"(11 + NORM_EXTRA_BUNDLES)
        : "$a0:1", "$a2:3", "$a4:5", "$a6:7", "memory");
    return;
  }
  if (worker >= width / 2) return;
  input += worker * 2;
#ifdef NORM_WITH_ADD
  right += worker * 2;
#endif
  scale += worker * 2;
  bias += worker * 2;
  output += worker * 2;
  const unsigned rounds = (width / 2 + 5 - worker) / 6;
  const float constants[2] = {-mean, inverse};
  asm volatile(
      "ld32 $a6, %[constants], $mzero, 0\n"
      "ld32 $a7, %[constants], $mzero, 1\n"
      "{ rpt %[rounds], %[bundles]; fnop }\n"
      "{ ld32step $a0, $mzero, %[in]+=, 6; fnop }\n"
      NORM_LOAD_RIGHT
      "{ nop; f16v2tof32 $a0:1, $a0 }\n"
      "{ nop; f32v2add $a0:1, $a6:B, $a0:1 }\n"
      "{ ld32step $a2, $mzero, %[scale]+=, 6; f32v2mul $a0:1, $a7:B, $a0:1 }\n"
      "{ nop; f16v2tof32 $a2:3, $a2 }\n"
      "{ ld32step $a4, $mzero, %[bias]+=, 6; f32v2mul $a0:1, $a0:1, $a2:3 }\n"
      "{ nop; f16v2tof32 $a2:3, $a4 }\n"
      "{ nop; f32v2add $a0:1, $a0:1, $a2:3 }\n"
      "{ nop; f32v2tof16 $a0, $a0:1 }\n"
      "{ st32step $a0, $mzero, %[out]+=, 6; fnop }\n"
      : [in] "+&r"(input), [right] "+&r"(right), [scale] "+&r"(scale),
        [bias] "+&r"(bias), [out] "+&r"(output)
      : [rounds] "r"(rounds), [constants] "r"(constants),
        [bundles] "i"(9 + NORM_EXTRA_BUNDLES)
      : "$a0:1", "$a2:3", "$a4", "$a6:7", "memory");
}
#undef NORM_LOAD_RIGHT
#undef NORM_EXTRA_BUNDLES

// Write the rounded residual once while accumulating its FP32 sum. Subsequent
// centered-variance/application passes read this explicit result.
static inline float2 normAddAndStore(const half *left, const half *right, half *output,
                                     unsigned width, unsigned worker) {
  float2 result = {0, 0};
  if (worker >= (width + 7) / 8) return result;
  const unsigned rounds = worker < width / 8 ? (width / 8 + 5 - worker) / 6 : 0;
  left += worker * 8;
  right += worker * 8;
  output += worker * 8;
  asm volatile("setzi $a0, (1 << 3)\nuput $FP_CLR, $a0" : : : "$a0", "memory");
  if (rounds) {
    asm volatile(
        "{ rpt %[rounds], 6; fnop }\n"
        "{ ld64step $a0:1, $mzero, %[left]+=, 1; fnop }\n"
        "{ ld64step $a2:3, $mzero, %[left]+=, 11; fnop }\n"
        "{ ld64step $a4:5, $mzero, %[right]+=, 1; fnop }\n"
        "{ ld64step $a6:7, $mzero, %[right]+=, 11; f16v4add $a0:1, $a0:1, $a4:5 }\n"
        "{ nop; f16v4add $a2:3, $a2:3, $a6:7 }\n"
        "{ st64step $a0:1, $mzero, %[out]+=, 1; fnop }\n"
        "{ st64step $a2:3, $mzero, %[out]+=, 11; f16v8acc $a0:3 }\n"
        : [left] "+&r"(left), [right] "+&r"(right), [out] "+&r"(output)
        : [rounds] "r"(rounds)
        : "$a0:1", "$a2:3", "$a4:5", "$a6:7", "memory");
  }
  if (worker * 8 + rounds * 48 < width) {
    asm volatile(
        "ld64 $a0:1, %[left], $mzero, 0\n"
        "ld64 $a4:5, %[right], $mzero, 0\n"
        "f16v4add $a0:1, $a0:1, $a4:5\n"
        "zero $a2:3\n"
        "{ st64 $a0:1, %[out], $mzero, 0; f16v8acc $a0:3 }\n"
        : : [left] "r"(left), [right] "r"(right), [out] "r"(output)
        : "$a0:1", "$a2:3", "$a4:5", "memory");
  }
  asm volatile(
      "f32v2gina $a0:1, $azeros, 0\n"
      "f32v2gina $a2:3, $azeros, 0\n"
      "f32v2add $a0:1, $a0:1, $a2:3\n"
      "f32v2gina $a2:3, $azeros, 0\n"
      "f32v2add $a0:1, $a0:1, $a2:3\n"
      "f32v2gina $a2:3, $azeros, 0\n"
      "f32v2add $a0:1, $a0:1, $a2:3\n"
      "st64 $a0:1, %[result], $mzero, 0\n"
      : : [result] "r"(&result) : "$a0:1", "$a2:3", "memory");
  return result;
}
