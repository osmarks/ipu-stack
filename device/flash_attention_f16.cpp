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

class FlashAttentionF16 : public MultiVertex {
public:
  Input<Vector<half, VectorLayout::ONE_PTR>> scores;
  Input<Vector<half, VectorLayout::ONE_PTR>> keyValue;
  Output<Vector<float, VectorLayout::ONE_PTR>> accumulator;
  unsigned queryRows;
  unsigned keyRows;
  unsigned initialBlock;
  unsigned finalBlock;

  bool compute(unsigned worker) {
    constexpr unsigned dimension = ATTENTION_HEAD_DIMENSION;
    const float scale = 1.0f / __builtin_sqrtf(float(dimension));
    float *maxima = &accumulator[queryRows * dimension];
    float *denominators = &maxima[queryRows];

    for (unsigned row = worker; row < queryRows; row += 6) {
      float *output = &accumulator[row * dimension];
      unsigned keyRow = 0;
      float maximum;
      float denominator;
      if (initialBlock) {
        maximum = float(scores[scoreIndex(row, 0)]) * scale;
        denominator = 1.0f;
        const half *firstValue = values();
        for (unsigned column = 0; column < dimension; ++column)
          output[column] = float(firstValue[column]);
        keyRow = 1;
      } else {
        maximum = maxima[row];
        denominator = denominators[row];
      }

      for (; keyRow < keyRows; ++keyRow) {
        const half *value = &values()[keyRow * dimension];
        const float score = float(scores[scoreIndex(row, keyRow)]) * scale;
        if (score <= maximum) {
          const float weight = __builtin_expf(score - maximum);
          denominator += weight;
          for (unsigned column = 0; column < dimension; ++column)
            output[column] += weight * float(value[column]);
        } else {
          const float previousScale = __builtin_expf(maximum - score);
          denominator = denominator * previousScale + 1.0f;
          for (unsigned column = 0; column < dimension; ++column)
            output[column] =
                output[column] * previousScale + float(value[column]);
          maximum = score;
        }
      }

      maxima[row] = maximum;
      denominators[row] = denominator;
      if (finalBlock) {
        const float reciprocal = 1.0f / denominator;
        for (unsigned column = 0; column < dimension; ++column)
          output[column] *= reciprocal;
      }
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

  const half *values() const {
    return &keyValue[ATTENTION_PADDED_HEAD_DIMENSION *
                     ATTENTION_KEY_BLOCK_COLUMNS];
  }
};
