#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

#ifndef ATTENTION_VERTEX_NAME
#define ATTENTION_VERTEX_NAME FlashAttentionOnlineF16
#endif
#ifndef ATTENTION_MATRICES
#define ATTENTION_MATRICES 1
#endif
#ifndef ATTENTION_QUERY_ROWS
#define ATTENTION_QUERY_ROWS 16
#endif
#ifndef ATTENTION_KEY_ROWS
#define ATTENTION_KEY_ROWS ATTENTION_QUERY_ROWS
#endif
#ifndef ATTENTION_QUERY_DIMENSION
#define ATTENTION_QUERY_DIMENSION 64
#endif
#ifndef ATTENTION_VALUE_DIMENSION
#define ATTENTION_VALUE_DIMENSION ATTENTION_QUERY_DIMENSION
#endif
#ifndef ATTENTION_SCALE
#define ATTENTION_SCALE (1.0f / __builtin_sqrtf(float(ATTENTION_QUERY_DIMENSION)))
#endif

using namespace poplar;

static_assert(ATTENTION_MATRICES > 0);
static_assert(ATTENTION_QUERY_ROWS > 0);
static_assert(ATTENTION_KEY_ROWS > 0);
static_assert(ATTENTION_QUERY_DIMENSION > 0);
static_assert(ATTENTION_VALUE_DIMENSION > 0);

// Exact, non-causal online softmax attention. Each worker owns complete query
// rows, so the running maximum, denominator, and output vector never need to
// be synchronized. The kernel materializes no QK matrix.
class ATTENTION_VERTEX_NAME : public MultiVertex {
public:
  Input<Vector<half, VectorLayout::ONE_PTR>> query;
  Input<Vector<half, VectorLayout::ONE_PTR>> key;
  Input<Vector<half, VectorLayout::ONE_PTR>> value;
  Output<Vector<float, VectorLayout::ONE_PTR>> output;

  bool compute(unsigned worker) {
    constexpr unsigned matrices = ATTENTION_MATRICES;
    constexpr unsigned queryRows = ATTENTION_QUERY_ROWS;
    constexpr unsigned keyRows = ATTENTION_KEY_ROWS;
    constexpr unsigned queryDimension = ATTENTION_QUERY_DIMENSION;
    constexpr unsigned valueDimension = ATTENTION_VALUE_DIMENSION;
    constexpr float scale = ATTENTION_SCALE;

    for (unsigned flatQuery = worker; flatQuery < matrices * queryRows;
         flatQuery += 6) {
      const unsigned matrix = flatQuery / queryRows;
      const unsigned queryRow = flatQuery % queryRows;
      const half *queryVector =
          &query[(matrix * queryRows + queryRow) * queryDimension];
      const half *keys = &key[matrix * keyRows * queryDimension];
      const half *values = &value[matrix * keyRows * valueDimension];
      volatile float *destination =
          &output[(matrix * queryRows + queryRow) * valueDimension];

      for (unsigned column = 0; column < valueDimension; ++column)
        destination[column] = 0.0f;

      float maximum = -__builtin_inff();
      float denominator = 0.0f;
      for (unsigned keyRow = 0; keyRow < keyRows; ++keyRow) {
        float score = 0.0f;
        const half *keyVector = &keys[keyRow * queryDimension];
        for (unsigned column = 0; column < queryDimension; ++column)
          score += float(queryVector[column]) * float(keyVector[column]);
        score *= scale;

        const float nextMaximum = __builtin_fmaxf(maximum, score);
        const float previousScale =
            maximum == -__builtin_inff() ? 0.0f
                                         : __builtin_expf(maximum - nextMaximum);
        const float weight = __builtin_expf(score - nextMaximum);
        denominator = denominator * previousScale + weight;
        const half *valueVector = &values[keyRow * valueDimension];
        for (unsigned column = 0; column < valueDimension; ++column) {
          destination[column] = destination[column] * previousScale +
                                weight * float(valueVector[column]);
        }
        maximum = nextMaximum;
      }

      const float reciprocal = 1.0f / denominator;
      for (unsigned column = 0; column < valueDimension; ++column)
        destination[column] *= reciprocal;
    }
    return true;
  }
};
