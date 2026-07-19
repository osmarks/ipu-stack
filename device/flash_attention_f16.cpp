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

using namespace poplar;

static_assert(ATTENTION_HEAD_DIMENSION > 0);

class AttentionSoftmaxF16 : public MultiVertex {
public:
  Input<Vector<half, VectorLayout::ONE_PTR>> scores;
  Input<Vector<half, VectorLayout::ONE_PTR>> unused;
  Output<Vector<half, VectorLayout::ONE_PTR>> weights;
  unsigned queryRows;
  unsigned keyRows;

  bool compute(unsigned worker) {
    constexpr unsigned dimension = ATTENTION_HEAD_DIMENSION;
    const float scale = 1.0f / __builtin_sqrtf(float(dimension));
    float *maxima = reinterpret_cast<float *>(
        &weights[queryRows * ATTENTION_KEY_BLOCK_COLUMNS]);
    float *denominators = maxima + queryRows;

    for (unsigned row = worker; row < queryRows; row += 6) {
      float maximum = float(scores[scoreIndex(row, 0)]) * scale;
      for (unsigned column = 1; column < keyRows; ++column)
        maximum = __builtin_fmaxf(
            maximum, float(scores[scoreIndex(row, column)]) * scale);
      float denominator = 0.0f;
      for (unsigned column = 0; column < ATTENTION_KEY_BLOCK_COLUMNS;
           column += 2) {
        const float first = column < keyRows
                                ? __builtin_expf(float(scores[scoreIndex(
                                                     row, column)]) *
                                                     scale -
                                                 maximum)
                                : 0.0f;
        const float second = column + 1 < keyRows
                                 ? __builtin_expf(float(scores[scoreIndex(
                                                      row, column + 1)]) *
                                                      scale -
                                                  maximum)
                                 : 0.0f;
        const float2 unpacked = {first, second};
        const half2 packed = __builtin_convertvector(unpacked, half2);
        *reinterpret_cast<half2 *>(&weights[weightIndex(row, column)]) = packed;
        denominator += first + second;
      }
      maxima[row] = maximum;
      denominators[row] = denominator;
    }
    return true;
  }

private:
  __attribute__((always_inline)) unsigned scoreIndex(unsigned row,
                                                     unsigned column) const {
    const unsigned panel = column / 16;
    const unsigned logicalPair = (column % 16) / 2;
    const unsigned physicalPair = (logicalPair % 4) * 2 + logicalPair / 4;
    return panel * queryRows * 16 + row * 16 + physicalPair * 2 + column % 2;
  }

  __attribute__((always_inline)) unsigned weightIndex(unsigned row,
                                                      unsigned column) const {
    return (column / 16) * queryRows * 16 + row * 16 + column % 16;
  }
};

class AttentionMergeF16 : public MultiVertex {
public:
  Input<Vector<half, VectorLayout::ONE_PTR>> blockValues;
  Input<Vector<half, VectorLayout::ONE_PTR>> blockState;
  Output<Vector<float, VectorLayout::ONE_PTR>> accumulator;
  unsigned queryRows;
  unsigned initialBlock;
  unsigned finalBlock;

  bool compute(unsigned worker) {
    constexpr unsigned dimension = ATTENTION_HEAD_DIMENSION;
    const float *blockMaxima = reinterpret_cast<const float *>(
        &blockState[queryRows * ATTENTION_KEY_BLOCK_COLUMNS]);
    const float *blockDenominators = blockMaxima + queryRows;
    float *maxima = &accumulator[queryRows * dimension];
    float *denominators = maxima + queryRows;

    for (unsigned row = worker; row < queryRows; row += 6) {
      float *output = &accumulator[row * dimension];
      const float blockMaximum = blockMaxima[row];
      const float blockDenominator = blockDenominators[row];
      if (initialBlock) {
        for (unsigned column = 0; column < dimension; ++column)
          output[column] = float(blockValues[valueIndex(row, column)]);
        maxima[row] = blockMaximum;
        denominators[row] = blockDenominator;
      } else {
        const float maximum = __builtin_fmaxf(maxima[row], blockMaximum);
        const float previousScale = __builtin_expf(maxima[row] - maximum);
        const float blockScale = __builtin_expf(blockMaximum - maximum);
        for (unsigned column = 0; column < dimension; ++column)
          output[column] = output[column] * previousScale +
                           float(blockValues[valueIndex(row, column)]) *
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

private:
  __attribute__((always_inline)) unsigned valueIndex(unsigned row,
                                                     unsigned column) const {
    const unsigned panel = column / 16;
    const unsigned logicalPair = (column % 16) / 2;
    const unsigned physicalPair = (logicalPair % 4) * 2 + logicalPair / 4;
    return panel * queryRows * 16 + row * 16 + physicalPair * 2 + column % 2;
  }
};
