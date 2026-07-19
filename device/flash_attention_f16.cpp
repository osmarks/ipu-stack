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

template <unsigned QueryRows>
__attribute__((always_inline)) unsigned c16Index(unsigned row,
                                                 unsigned column) {
  const unsigned panel = column / 16;
  const unsigned logicalPair = (column % 16) / 2;
  const unsigned physicalPair = (logicalPair % 4) * 2 + logicalPair / 4;
  return panel * QueryRows * 16 + row * 16 + physicalPair * 2 + column % 2;
}

template <unsigned QueryRows>
__attribute__((always_inline)) unsigned a16Index(unsigned row,
                                                 unsigned column) {
  return (column / 16) * QueryRows * 16 + row * 16 + column % 16;
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
    float maximum = float(scores[c16Index<QueryRows>(row, 0)]) * scale;
    for (unsigned column = 1; column < KeyRows; ++column)
      maximum = __builtin_fmaxf(
          maximum,
          float(scores[c16Index<QueryRows>(row, column)]) * scale);
    float denominator = 0.0f;
    for (unsigned column = 0; column < ATTENTION_KEY_BLOCK_COLUMNS;
         column += 2) {
      const float first = column < KeyRows
                              ? __builtin_expf(float(scores[c16Index<QueryRows>(
                                                   row, column)]) *
                                                   scale -
                                               maximum)
                              : 0.0f;
      const float second = column + 1 < KeyRows
                               ? __builtin_expf(float(scores[c16Index<QueryRows>(
                                                    row, column + 1)]) *
                                                    scale -
                                                maximum)
                               : 0.0f;
      const float2 unpacked = {first, second};
      const half2 packed = __builtin_convertvector(unpacked, half2);
      *reinterpret_cast<half2 *>(
          &weights[a16Index<QueryRows>(row, column)]) = packed;
      denominator += first + second;
    }
    maxima[row] = maximum;
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

template <unsigned QueryRows, typename BlockValues, typename BlockState,
          typename Accumulator>
__attribute__((always_inline)) bool mergeBlock(const BlockValues &blockValues,
                                               const BlockState &blockState,
                                               Accumulator &accumulator,
                                               unsigned initialBlock,
                                               unsigned finalBlock,
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
    if (initialBlock) {
      for (unsigned column = 0; column < dimension; ++column)
        output[column] =
            float(blockValues[c16Index<QueryRows>(row, column)]);
      maxima[row] = blockMaximum;
      denominators[row] = blockDenominator;
    } else {
      const float maximum = __builtin_fmaxf(maxima[row], blockMaximum);
      const float previousScale = __builtin_expf(maxima[row] - maximum);
      const float blockScale = __builtin_expf(blockMaximum - maximum);
      for (unsigned column = 0; column < dimension; ++column)
        output[column] = output[column] * previousScale +
                         float(blockValues[c16Index<QueryRows>(row, column)]) *
                             blockScale;
      denominators[row] = denominators[row] * previousScale +
                          blockDenominator * blockScale;
      maxima[row] = maximum;
    }
    if (finalBlock) {
      const float reciprocal = 1.0f / denominators[row];
      for (unsigned column = 0; column < dimension; ++column)
        output[column] *= reciprocal;
    }
  }
  return true;
}

#define DEFINE_MERGE_VERTEX(Name, QueryRows)                                   \
  class Name : public MultiVertex {                                            \
  public:                                                                      \
    Input<Vector<half, VectorLayout::ONE_PTR>> blockValues;                     \
    Input<Vector<half, VectorLayout::ONE_PTR>> blockState;                      \
    Output<Vector<float, VectorLayout::ONE_PTR>> accumulator;                   \
    unsigned initialBlock;                                                      \
    unsigned finalBlock;                                                        \
                                                                               \
    bool compute(unsigned worker) {                                             \
      return mergeBlock<QueryRows>(blockValues, blockState, accumulator,        \
                                   initialBlock, finalBlock, worker);            \
    }                                                                          \
  }

DEFINE_MERGE_VERTEX(AttentionMergeSmallQueryF16,
                    ATTENTION_SMALL_QUERY_ROWS);
DEFINE_MERGE_VERTEX(AttentionMergeLargeQueryF16,
                    ATTENTION_LARGE_QUERY_ROWS);
