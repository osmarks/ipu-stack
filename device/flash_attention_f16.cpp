#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

#ifndef ATTENTION_SEQUENCE_LENGTH
#define ATTENTION_SEQUENCE_LENGTH 128
#endif
#ifndef ATTENTION_HEAD_DIMENSION
#define ATTENTION_HEAD_DIMENSION 64
#endif

using namespace poplar;

static_assert(ATTENTION_SEQUENCE_LENGTH > 0);
static_assert(ATTENTION_HEAD_DIMENSION > 0);

class FlashAttentionF16 : public MultiVertex {
public:
  Input<Vector<half, VectorLayout::ONE_PTR>> query;
  Input<Vector<half, VectorLayout::ONE_PTR>> keyValue;
  Output<Vector<float, VectorLayout::ONE_PTR>> accumulator;

  bool compute(unsigned worker) {
    constexpr unsigned sequence = ATTENTION_SEQUENCE_LENGTH;
    constexpr unsigned dimension = ATTENTION_HEAD_DIMENSION;
    const float scale = 1.0f / __builtin_sqrtf(float(dimension));

    for (unsigned row = worker; row < sequence; row += 6) {
      const half *q = &query[row * dimension];
      float *output = &accumulator[row * dimension];

      float maximum = dot(q, &keyValue[0], scale);
      float denominator = 1.0f;
      const half *firstValue = &keyValue[sequence * dimension];
      for (unsigned column = 0; column < dimension; ++column)
        output[column] = float(firstValue[column]);

      for (unsigned keyRow = 1; keyRow < sequence; ++keyRow) {
        const half *key = &keyValue[keyRow * dimension];
        const half *value =
            &keyValue[(sequence + keyRow) * dimension];
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

      const float reciprocal = 1.0f / denominator;
      for (unsigned column = 0; column < dimension; ++column)
        output[column] *= reciprocal;
    }
    return true;
  }

private:
  static __attribute__((always_inline)) float dot(const half *left,
                                                   const half *right,
                                                   float scale) {
    float result = 0.0f;
    for (unsigned column = 0; column < ATTENTION_HEAD_DIMENSION; ++column)
      result += float(left[column]) * float(right[column]);
    return result * scale;
  }
};
