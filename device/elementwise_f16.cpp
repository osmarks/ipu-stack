#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>
#include <cmath>
using namespace poplar;

#ifdef VERTEX_LayerNormF16
class LayerNormF16 : public MultiVertex {
public:
  Input<Vector<half, VectorLayout::ONE_PTR>> source, scale, bias;
  Output<Vector<half, VectorLayout::ONE_PTR>> destination;
  unsigned rows, width;
  bool compute(unsigned worker) {
    // Complete even-width rows avoid halfword read/modify/write races.
    for (unsigned row = worker; row < rows; row += 6) {
      const unsigned base = row * width;
      float sum = 0;
      for (unsigned i = 0; i < width; ++i) sum += float(source[base + i]);
      const float mean = sum / width;
      float variance = 0;
      for (unsigned i = 0; i < width; ++i) {
        const float d = float(source[base + i]) - mean;
        variance += d * d;
      }
      const float inverse = 1.0f / std::sqrt(variance / width + 1e-6f);
      for (unsigned i = 0; i < width; i += 2) {
        float2 result = {
          (float(source[base+i]) - mean) * inverse * float(scale[i]) + float(bias[i]),
          (float(source[base+i+1]) - mean) * inverse * float(scale[i+1]) + float(bias[i+1])};
        *reinterpret_cast<half2 *>(&destination[base+i]) = __builtin_convertvector(result, half2);
      }
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
    unsigned l = wrap(worker * 2, leftElements);
    unsigned r = wrap(worker * 2, rightElements);
    for (unsigned i = worker * 2; i + 1 < elements; i += 12) {
      float2 result = {float(left[l]) + float(right[r]),
                      float(left[wrap(l+1, leftElements)]) + float(right[wrap(r+1, rightElements)])};
      *reinterpret_cast<half2 *>(&destination[i]) = __builtin_convertvector(result, half2);
      l = wrap(l + 12, leftElements);
      r = wrap(r + 12, rightElements);
    }
    if (worker == 0 && (elements & 1)) {
      const unsigned i = elements - 1;
      union { half2 lanes; unsigned word; } tail;
      const float2 value = {float(left[wrap(i, leftElements)]) + float(right[wrap(i, rightElements)]), 0.0f};
      tail.lanes = __builtin_convertvector(value, half2);
      auto *words = reinterpret_cast<volatile unsigned *>(&destination[0]);
      words[i / 2] = (words[i / 2] & 0xffff0000u) | (tail.word & 0xffffu);
    }
    return true;
  }
};
#endif
