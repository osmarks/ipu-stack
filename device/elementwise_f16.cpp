#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>
#include "elementwise_vector.hpp"
#ifdef NORM_FP8
#include "elementwise_fp8.hpp"
#endif
using namespace poplar;

#ifdef VERTEX_LayerNormF16
#ifdef NORM_WITH_ADD
#define NORM_VERTEX AddLayerNormF16
#elif defined(NORM_FP8)
#define NORM_VERTEX LayerNormF8
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
#ifdef NORM_FP8
  Output<Vector<unsigned char, VectorLayout::ONE_PTR>> destination;
#else
  Output<Vector<half, VectorLayout::ONE_PTR>> destination;
#endif
  unsigned rows, width;
  InOut<Vector<float, VectorLayout::ONE_PTR>> scratch;
  unsigned stage;
#ifdef NORM_FP8
  unsigned row;
  int outputScale;
  unsigned packed;
#endif
  bool compute(unsigned worker) {
    auto *partials = reinterpret_cast<float2 *>(&scratch[0]);
    const half *x = &source[0];
#ifdef NORM_WITH_ADD
    const half *r = &right[0];
#else
    const half *r = nullptr;
#endif
    const unsigned width = this->width;
    if (stage == 0) {
      partials[worker] = normSum(x, r, width, worker, 0, false);
      return true;
    }
    float2 sum = {0, 0};
    for (unsigned i = 0; i < 6; ++i) sum += partials[i];
    const float mean = (sum[0] + sum[1]) / width;
    if (stage == 1) {
      partials[6 + worker] = normSum(x, r, width, worker, mean, true);
      return true;
    }
    float2 variance = {0, 0};
    for (unsigned i = 6; i < 12; ++i) variance += partials[i];
    const float inverse = normInverse((variance[0] + variance[1]) / width + 1e-6f);
#ifdef NORM_FP8
    setQuarterConfig({quarter_metadata::f143, static_cast<signed char>(-outputScale)});
    normApplyFp8(x, &scale[0], &bias[0], &destination[0], width, worker, mean, inverse, row, rows, packed);
#else
    normApply(x, r, &scale[0], &bias[0], &destination[0], width, worker, mean, inverse);
#endif
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
    const unsigned elements = this->elements;
    const unsigned leftElements = this->leftElements;
    const unsigned rightElements = this->rightElements;
    // Four-wide dense/suffix-broadcast loop; leave irregular halfword tails
    // and general broadcasting to the pair path below.
    if (elements && !(elements % 4) && !(leftElements % 4) && !(rightElements % 4) &&
        !((reinterpret_cast<unsigned>(&left[0]) | reinterpret_cast<unsigned>(&right[0]) |
           reinterpret_cast<unsigned>(&destination[0])) & 7)) {
      if (leftElements == elements && rightElements == elements) {
        addQuads(&left[0], &right[0], &destination[0], elements / 4, worker);
        return true;
      }
      if ((leftElements == elements || rightElements == elements) &&
          wrap(elements, leftElements < rightElements ? leftElements : rightElements) == 0) {
        const half *dense = leftElements == elements ? &left[0] : &right[0];
        const half *bias = leftElements == elements ? &right[0] : &left[0];
        const unsigned width = leftElements == elements ? rightElements : leftElements;
        half *out = &destination[0];
        for (unsigned row = 0; row < elements; row += width)
          addQuads(dense + row, bias, out + row, width / 4, worker);
        return true;
      }
    }
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
