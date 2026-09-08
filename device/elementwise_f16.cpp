#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>
#include <cmath>
using namespace poplar;

#ifdef VERTEX_LayerNormF16
#ifdef NORM_WITH_ADD
#define NORM_VERTEX AddLayerNormF16
#else
#define NORM_VERTEX LayerNormF16
#endif
class NORM_VERTEX : public MultiVertex {
public:
  Input<Vector<half, VectorLayout::ONE_PTR>> source;
#ifdef NORM_WITH_ADD
  Input<Vector<half, VectorLayout::ONE_PTR>> right;
#endif
  Input<Vector<half, VectorLayout::ONE_PTR>> scale, bias;
  Output<Vector<half, VectorLayout::ONE_PTR>> destination;
  unsigned rows, width;
  InOut<Vector<float, VectorLayout::ONE_PTR>> scratch;
  unsigned stage;
  half2 load(unsigned i) const {
    half2 value = reinterpret_cast<const half2 *>(&source[0])[i];
#ifdef NORM_WITH_ADD
    value += reinterpret_cast<const half2 *>(&right[0])[i];
#endif
    return value;
  }
  bool compute(unsigned worker) {
    auto *partials = reinterpret_cast<float2 *>(&scratch[0]);
    if (stage == 0) {
      float2 sum = {0, 0};
      for (unsigned i = worker; i < width / 2; i += 6)
        sum += __builtin_convertvector(load(i), float2);
      partials[worker] = sum;
      return true;
    }
    float2 sum = {0, 0};
    for (unsigned i = 0; i < 6; ++i) sum += partials[i];
    const float mean = (sum[0] + sum[1]) / width;
    if (stage == 1) {
      float2 variance = {0, 0};
      for (unsigned i = worker; i < width / 2; i += 6) {
        const float2 d = __builtin_convertvector(load(i), float2) - mean;
        variance += d * d;
      }
      partials[6 + worker] = variance;
      return true;
    }
    float2 variance = {0, 0};
    for (unsigned i = 6; i < 12; ++i) variance += partials[i];
    const float inverse = 1.0f / std::sqrt((variance[0] + variance[1]) / width + 1e-6f);
    const auto *gamma = reinterpret_cast<const half2 *>(&scale[0]);
    const auto *beta = reinterpret_cast<const half2 *>(&bias[0]);
    auto *y = reinterpret_cast<half2 *>(&destination[0]);
    for (unsigned i = worker; i < width / 2; i += 6) {
      const float2 normalized = (__builtin_convertvector(load(i), float2) - mean) * inverse;
      const float2 result = normalized * __builtin_convertvector(gamma[i], float2)
                            + __builtin_convertvector(beta[i], float2);
      y[i] = __builtin_convertvector(result, half2);
    }
    return true;
  }
};
#endif

#ifdef VERTEX_AddF16
class AddF16 : public MultiVertex {
public:
  Input<Vector<half, VectorLayout::ONE_PTR>> left, right;
  Output<Vector<half, VectorLayout::ONE_PTR>> destination;
  unsigned elements, leftElements, rightElements;
  static unsigned wrap(unsigned i, unsigned size) {
    while (i >= size) i -= size;
    return i;
  }
  bool compute(unsigned worker) {
    // Short column shards should not wait for inactive contexts to wrap
    // broadcast indices. Worker zero also owns a possible final halfword.
    if (worker != 0 && worker >= elements / 2) return true;
    // The common dense and vector-broadcast paths use complete half2 words.
    // Keep wrap/gather handling out of their inner loops.
    if (!(elements & 1) && !(leftElements & 1) && !(rightElements & 1)) {
      const auto *a = reinterpret_cast<const half2 *>(&left[0]);
      const auto *b = reinterpret_cast<const half2 *>(&right[0]);
      auto *out = reinterpret_cast<half2 *>(&destination[0]);
      if (leftElements == elements && rightElements == elements) {
        for (unsigned i = worker; i < elements / 2; i += 6)
          out[i] = a[i] + b[i];
      } else if ((leftElements == elements || rightElements == elements) &&
                 wrap(elements, leftElements < rightElements ? leftElements : rightElements) == 0) {
        const auto *dense = leftElements == elements ? a : b;
        const auto *bias = leftElements == elements ? b : a;
        const unsigned width = (leftElements == elements ? rightElements : leftElements) / 2;
        // Broadcast a contiguous suffix. Reset its pointer once per row;
        // both loads in the vector loop now advance linearly.
        for (unsigned row = 0; row < elements / 2; row += width)
          for (unsigned i = worker; i < width; i += 6)
            out[row + i] = dense[row + i] + bias[i];
      } else {
        unsigned l = wrap(worker, leftElements / 2);
        unsigned r = wrap(worker, rightElements / 2);
        for (unsigned i = worker; i < elements / 2; i += 6) {
          out[i] = a[l] + b[r];
          l = wrap(l + 6, leftElements / 2);
          r = wrap(r + 6, rightElements / 2);
        }
      }
      return true;
    }
    unsigned l = wrap(worker * 2, leftElements);
    unsigned r = wrap(worker * 2, rightElements);
    for (unsigned i = worker * 2; i + 1 < elements; i += 12) {
      const half2 a = {left[l], left[wrap(l+1, leftElements)]};
      const half2 b = {right[r], right[wrap(r+1, rightElements)]};
      *reinterpret_cast<half2 *>(&destination[i]) = a + b;
      l = wrap(l + 12, leftElements);
      r = wrap(r + 12, rightElements);
    }
    if (worker == 0 && (elements & 1)) {
      const unsigned i = elements - 1;
      union { half2 lanes; unsigned word; } tail;
      const half2 a = {left[wrap(i, leftElements)], half(0)};
      const half2 b = {right[wrap(i, rightElements)], half(0)};
      tail.lanes = a + b;
      auto *words = reinterpret_cast<volatile unsigned *>(&destination[0]);
      words[i / 2] = (words[i / 2] & 0xffff0000u) | (tail.word & 0xffffu);
    }
    return true;
  }
};
#endif
