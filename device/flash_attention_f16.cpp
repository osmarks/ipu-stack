#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

#ifndef ATTENTION_HEAD_DIMENSION
#define ATTENTION_HEAD_DIMENSION 64
#endif
#ifndef ATTENTION_PADDED_HEAD_DIMENSION
#define ATTENTION_PADDED_HEAD_DIMENSION ATTENTION_HEAD_DIMENSION
#endif
#ifndef ATTENTION_KEY_BLOCK_COLUMNS
#define ATTENTION_KEY_BLOCK_COLUMNS 64
#endif
#ifndef ATTENTION_SMALL_QUERY_ROWS
#define ATTENTION_SMALL_QUERY_ROWS 1
#endif
#ifndef ATTENTION_LARGE_QUERY_ROWS
#define ATTENTION_LARGE_QUERY_ROWS ATTENTION_SMALL_QUERY_ROWS
#endif
#ifndef ATTENTION_SMALL_KEY_ROWS
#define ATTENTION_SMALL_KEY_ROWS ATTENTION_KEY_BLOCK_COLUMNS
#endif
#ifndef ATTENTION_LARGE_KEY_ROWS
#define ATTENTION_LARGE_KEY_ROWS ATTENTION_KEY_BLOCK_COLUMNS
#endif

using namespace poplar;

static_assert(ATTENTION_HEAD_DIMENSION > 0);
static_assert(ATTENTION_KEY_BLOCK_COLUMNS % 16 == 0);
static_assert(ATTENTION_PADDED_HEAD_DIMENSION % 16 == 0);

static __attribute__((always_inline)) void expHalf16(half *values) {
  unsigned loopCount = 4;
  const half4 *input = reinterpret_cast<const half4 *>(values);
  half4 *output = reinterpret_cast<half4 *>(values);
  asm volatile(R"(
    .macro ipu_stack_exp_half2 OPERANDS:vararg
      f16v2exp \OPERANDS
    .endm
    brnzdec %[count], 3f
    bri 4f
    .align 8
    nop
  3:
    ld64step $a0:1, $mzero, %[input]+=, 1
    { rpt %[count], (2f-1f)/8 - 1
      ipu_stack_exp_half2 $a2, $a0 }
  1:
    { ld64step $a0:1, $mzero, %[input]+=, 1
      ipu_stack_exp_half2 $a3, $a1 }
    { st64step $a2:3, $mzero, %[output]+=, 1
      ipu_stack_exp_half2 $a2, $a0 }
  2:
    ipu_stack_exp_half2 $a3, $a1
    st64step $a2:3, $mzero, %[output]+=, 1
  4:
    .purgem ipu_stack_exp_half2
  )"
               : [count] "+r"(loopCount), [input] "+r"(input),
                 [output] "+r"(output)
               :
               : "$a0:1", "$a2:3", "memory");
}

constexpr unsigned logicalPairForPhysical(unsigned physicalPair) {
  return physicalPair % 2 == 0 ? physicalPair / 2
                               : 4 + physicalPair / 2;
}

template <unsigned QueryRows, unsigned KeyRows, typename Scores,
          typename Weights>
__attribute__((always_inline)) bool softmaxBlock(const Scores &scores,
                                                 Weights &weights,
                                                 unsigned worker) {
  constexpr unsigned dimension = ATTENTION_HEAD_DIMENSION;
  const float scale = 1.0f / __builtin_sqrtf(float(dimension));
  float *maxima = reinterpret_cast<float *>(
      &weights[QueryRows * ATTENTION_KEY_BLOCK_COLUMNS]);
  float *denominators = maxima + QueryRows;

  for (unsigned row = worker; row < QueryRows; row += 6) {
    float maximum = -65504.0f;
    for (unsigned panel = 0; panel < ATTENTION_KEY_BLOCK_COLUMNS / 16;
         ++panel) {
#pragma unroll
      for (unsigned physicalPair = 0; physicalPair < 8; ++physicalPair) {
        const unsigned logicalPair = logicalPairForPhysical(physicalPair);
        const unsigned column = panel * 16 + logicalPair * 2;
        if (column < KeyRows) {
          const unsigned source = panel * QueryRows * 16 + row * 16 +
                                  physicalPair * 2;
          const half2 packed =
              *reinterpret_cast<const half2 *>(&scores[source]);
          const float2 unpacked = __builtin_convertvector(packed, float2);
          maximum = __builtin_fmaxf(maximum, unpacked[0]);
          if (column + 1 < KeyRows)
            maximum = __builtin_fmaxf(maximum, unpacked[1]);
        }
      }
    }

    const float scaledMaximum = maximum * scale;
    for (unsigned panel = 0; panel < ATTENTION_KEY_BLOCK_COLUMNS / 16;
         ++panel) {
#pragma unroll
      for (unsigned physicalPair = 0; physicalPair < 8; ++physicalPair) {
        const unsigned logicalPair = logicalPairForPhysical(physicalPair);
        const unsigned column = panel * 16 + logicalPair * 2;
        float2 normalized = {-65504.0f, -65504.0f};
        if (column < KeyRows) {
          const unsigned source = panel * QueryRows * 16 + row * 16 +
                                  physicalPair * 2;
          const half2 packed =
              *reinterpret_cast<const half2 *>(&scores[source]);
          const float2 unpacked = __builtin_convertvector(packed, float2);
          normalized[0] = unpacked[0] * scale - scaledMaximum;
          if (column + 1 < KeyRows)
            normalized[1] = unpacked[1] * scale - scaledMaximum;
        }
        const unsigned destination =
            panel * QueryRows * 16 + row * 16 + logicalPair * 2;
        *reinterpret_cast<half2 *>(&weights[destination]) =
            __builtin_convertvector(normalized, half2);
      }
      expHalf16(&weights[panel * QueryRows * 16 + row * 16]);
    }

    float denominator = 0.0f;
    for (unsigned panel = 0; panel < ATTENTION_KEY_BLOCK_COLUMNS / 16;
         ++panel) {
#pragma unroll
      for (unsigned logicalPair = 0; logicalPair < 8; ++logicalPair) {
        const unsigned column = panel * 16 + logicalPair * 2;
        if (column < KeyRows) {
          const unsigned source =
              panel * QueryRows * 16 + row * 16 + logicalPair * 2;
          const half2 packed =
              *reinterpret_cast<const half2 *>(&weights[source]);
          const float2 unpacked = __builtin_convertvector(packed, float2);
          denominator += unpacked[0];
          if (column + 1 < KeyRows)
            denominator += unpacked[1];
        }
      }
    }
    maxima[row] = scaledMaximum;
    denominators[row] = denominator;
  }
  return true;
}

#define DEFINE_SOFTMAX_VERTEX(Name, QueryRows, KeyRows)                         \
  class Name : public MultiVertex {                                            \
  public:                                                                      \
    Input<Vector<half, VectorLayout::ONE_PTR>> scores;                          \
    Input<Vector<half, VectorLayout::ONE_PTR>> unused;                          \
    Output<Vector<half, VectorLayout::ONE_PTR>> weights;                        \
                                                                               \
    bool compute(unsigned worker) {                                             \
      return softmaxBlock<QueryRows, KeyRows>(scores, weights, worker);         \
    }                                                                          \
  }

DEFINE_SOFTMAX_VERTEX(AttentionSoftmaxSmallQuerySmallKeyF16,
                      ATTENTION_SMALL_QUERY_ROWS, ATTENTION_SMALL_KEY_ROWS);
DEFINE_SOFTMAX_VERTEX(AttentionSoftmaxSmallQueryLargeKeyF16,
                      ATTENTION_SMALL_QUERY_ROWS, ATTENTION_LARGE_KEY_ROWS);
DEFINE_SOFTMAX_VERTEX(AttentionSoftmaxLargeQuerySmallKeyF16,
                      ATTENTION_LARGE_QUERY_ROWS, ATTENTION_SMALL_KEY_ROWS);
DEFINE_SOFTMAX_VERTEX(AttentionSoftmaxLargeQueryLargeKeyF16,
                      ATTENTION_LARGE_QUERY_ROWS, ATTENTION_LARGE_KEY_ROWS);

template <unsigned QueryRows, bool InitialBlock, bool FinalBlock,
          typename BlockValues, typename BlockState, typename Accumulator>
__attribute__((always_inline)) bool mergeBlock(const BlockValues &blockValues,
                                               const BlockState &blockState,
                                               Accumulator &accumulator,
                                               unsigned worker) {
  constexpr unsigned dimension = ATTENTION_HEAD_DIMENSION;
  const float *blockMaxima = reinterpret_cast<const float *>(
      &blockState[QueryRows * ATTENTION_KEY_BLOCK_COLUMNS]);
  const float *blockDenominators = blockMaxima + QueryRows;
  float *maxima = &accumulator[QueryRows * dimension];
  float *denominators = maxima + QueryRows;

  for (unsigned row = worker; row < QueryRows; row += 6) {
    float *output = &accumulator[row * dimension];
    const float blockMaximum = blockMaxima[row];
    const float blockDenominator = blockDenominators[row];
    float previousScale = 0.0f;
    float blockScale = 1.0f;
    if (InitialBlock) {
      maxima[row] = blockMaximum;
      denominators[row] = blockDenominator;
    } else {
      const float maximum = __builtin_fmaxf(maxima[row], blockMaximum);
      previousScale = __builtin_expf(maxima[row] - maximum);
      blockScale = __builtin_expf(blockMaximum - maximum);
      denominators[row] = denominators[row] * previousScale +
                          blockDenominator * blockScale;
      maxima[row] = maximum;
    }

    for (unsigned panel = 0; panel < ATTENTION_PADDED_HEAD_DIMENSION / 16;
         ++panel) {
#pragma unroll
      for (unsigned physicalPair = 0; physicalPair < 8; ++physicalPair) {
        const unsigned logicalPair = logicalPairForPhysical(physicalPair);
        const unsigned column = panel * 16 + logicalPair * 2;
        if (column < dimension) {
          const unsigned source = panel * QueryRows * 16 + row * 16 +
                                  physicalPair * 2;
          const half2 packed =
              *reinterpret_cast<const half2 *>(&blockValues[source]);
          const float2 values = __builtin_convertvector(packed, float2);
          float2 *destination = reinterpret_cast<float2 *>(&output[column]);
          if (InitialBlock)
            *destination = values;
          else
            *destination = *destination * previousScale + values * blockScale;
        }
      }
    }
    if (FinalBlock) {
      const float reciprocal = 1.0f / denominators[row];
      for (unsigned column = 0; column < dimension; column += 2) {
        float2 *values = reinterpret_cast<float2 *>(&output[column]);
        *values *= reciprocal;
      }
    }
  }
  return true;
}

#define DEFINE_MERGE_VERTEX(Name, QueryRows, InitialBlock, FinalBlock)          \
  class Name : public MultiVertex {                                            \
  public:                                                                      \
    Input<Vector<half, VectorLayout::ONE_PTR>> blockValues;                     \
    Input<Vector<half, VectorLayout::ONE_PTR>> blockState;                      \
    Output<Vector<float, VectorLayout::ONE_PTR>> accumulator;                   \
    bool compute(unsigned worker) {                                             \
      return mergeBlock<QueryRows, InitialBlock, FinalBlock>(                   \
          blockValues, blockState, accumulator, worker);                        \
    }                                                                          \
  }

DEFINE_MERGE_VERTEX(AttentionMergeSmallQuerySingleBlockF16,
                    ATTENTION_SMALL_QUERY_ROWS, true, true);
DEFINE_MERGE_VERTEX(AttentionMergeSmallQueryInitialBlockF16,
                    ATTENTION_SMALL_QUERY_ROWS, true, false);
DEFINE_MERGE_VERTEX(AttentionMergeSmallQueryMiddleBlockF16,
                    ATTENTION_SMALL_QUERY_ROWS, false, false);
DEFINE_MERGE_VERTEX(AttentionMergeSmallQueryFinalBlockF16,
                    ATTENTION_SMALL_QUERY_ROWS, false, true);
DEFINE_MERGE_VERTEX(AttentionMergeLargeQuerySingleBlockF16,
                    ATTENTION_LARGE_QUERY_ROWS, true, true);
DEFINE_MERGE_VERTEX(AttentionMergeLargeQueryInitialBlockF16,
                    ATTENTION_LARGE_QUERY_ROWS, true, false);
DEFINE_MERGE_VERTEX(AttentionMergeLargeQueryMiddleBlockF16,
                    ATTENTION_LARGE_QUERY_ROWS, false, false);
DEFINE_MERGE_VERTEX(AttentionMergeLargeQueryFinalBlockF16,
                    ATTENTION_LARGE_QUERY_ROWS, false, true);
