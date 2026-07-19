#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

#ifndef ATTENTION_HEAD_DIMENSION
#define ATTENTION_HEAD_DIMENSION 64
#endif

using namespace poplar;

static_assert(ATTENTION_HEAD_DIMENSION > 0);

extern "C" float ipu_stack_attention_dot(const half *, const half *);

class FlashAttentionF16 : public MultiVertex {
public:
  Input<Vector<half, VectorLayout::ONE_PTR>> query;
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
      const half *q = &query[row * dimension];
      float *output = &accumulator[row * dimension];
      unsigned keyRow = 0;
      float maximum;
      float denominator;
      if (initialBlock) {
        maximum = dot(q, &keyValue[0], scale);
        denominator = 1.0f;
        const half *firstValue = &keyValue[keyRows * dimension];
        for (unsigned column = 0; column < dimension; ++column)
          output[column] = float(firstValue[column]);
        keyRow = 1;
      } else {
        maximum = maxima[row];
        denominator = denominators[row];
      }

      for (; keyRow < keyRows; ++keyRow) {
        const half *key = &keyValue[keyRow * dimension];
        const half *value = &keyValue[(keyRows + keyRow) * dimension];
        const float score = dot(q, key, scale);
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
  static __attribute__((always_inline)) float dot(const half *left,
                                                   const half *right,
                                                   float scale) {
    static_assert(ATTENTION_HEAD_DIMENSION % 4 == 0);
    return ipu_stack_attention_dot(left, right) * scale;
  }
};
