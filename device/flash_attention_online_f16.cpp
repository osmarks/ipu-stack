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
#ifndef ATTENTION_KEY_BLOCK_ROWS
#define ATTENTION_KEY_BLOCK_ROWS 32
#endif

using namespace poplar;

static_assert(ATTENTION_MATRICES > 0);
static_assert(ATTENTION_QUERY_ROWS > 0);
static_assert(ATTENTION_KEY_ROWS > 0);
static_assert(ATTENTION_QUERY_DIMENSION > 0);
static_assert(ATTENTION_VALUE_DIMENSION > 0);
static_assert(ATTENTION_KEY_BLOCK_ROWS > 0);

static __attribute__((always_inline)) float attentionDot(const half *query,
                                                         const half *key) {
  constexpr unsigned dimension = ATTENTION_QUERY_DIMENSION;
  float score = 0.0f;
#if ATTENTION_QUERY_DIMENSION % 2 == 0
  for (unsigned column = 0; column < dimension; column += 2) {
    const half2 packedQuery =
        *reinterpret_cast<const half2 *>(&query[column]);
    const half2 packedKey = *reinterpret_cast<const half2 *>(&key[column]);
    const float2 queryPair = __builtin_convertvector(packedQuery, float2);
    const float2 keyPair = __builtin_convertvector(packedKey, float2);
    score += queryPair[0] * keyPair[0] + queryPair[1] * keyPair[1];
  }
#else
  for (unsigned column = 0; column < dimension; ++column)
    score += float(query[column]) * float(key[column]);
#endif
  return score;
}

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
      for (unsigned keyStart = 0; keyStart < keyRows;
           keyStart += ATTENTION_KEY_BLOCK_ROWS) {
        const unsigned remainingRows = keyRows - keyStart;
        const unsigned blockRows = remainingRows < ATTENTION_KEY_BLOCK_ROWS
                                       ? remainingRows
                                       : ATTENTION_KEY_BLOCK_ROWS;
        alignas(8) float scores[ATTENTION_KEY_BLOCK_ROWS];
        float blockMaximum = -__builtin_inff();
        for (unsigned blockRow = 0; blockRow < blockRows; ++blockRow) {
          const unsigned keyRow = keyStart + blockRow;
          scores[blockRow] =
              attentionDot(queryVector, &keys[keyRow * queryDimension]) * scale;
          blockMaximum = __builtin_fmaxf(blockMaximum, scores[blockRow]);
        }

        const float nextMaximum = __builtin_fmaxf(maximum, blockMaximum);
        const float previousScale = maximum == -__builtin_inff()
                                        ? 0.0f
                                        : __builtin_expf(maximum - nextMaximum);
        denominator *= previousScale;
        for (unsigned column = 0; column < valueDimension; ++column)
          destination[column] *= previousScale;

        for (unsigned blockRow = 0; blockRow < blockRows; ++blockRow) {
          const unsigned keyRow = keyStart + blockRow;
          const float weight = __builtin_expf(scores[blockRow] - nextMaximum);
          denominator += weight;
          const half *valueVector = &values[keyRow * valueDimension];
#if ATTENTION_VALUE_DIMENSION % 2 == 0
          for (unsigned column = 0; column < valueDimension; column += 2) {
            const half2 packedValue =
                *reinterpret_cast<const half2 *>(&valueVector[column]);
            const float2 valuePair =
                __builtin_convertvector(packedValue, float2);
            volatile float2 *destinationPair =
                reinterpret_cast<volatile float2 *>(&destination[column]);
            *destinationPair = *destinationPair + valuePair * weight;
          }
#else
          for (unsigned column = 0; column < valueDimension; ++column)
            destination[column] += weight * float(valueVector[column]);
#endif
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
