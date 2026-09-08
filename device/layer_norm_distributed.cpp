#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>
#include <cmath>
using namespace poplar;

#ifdef VERTEX_LayerNormMoments
class LayerNormMoments : public MultiVertex {
public:
  Input<Vector<half, VectorLayout::ONE_PTR>> source;
  Output<Vector<float, VectorLayout::ONE_PTR>> destination;
  unsigned width;
  InOut<Vector<float, VectorLayout::ONE_PTR>> scratch;
  unsigned stage;
  bool compute(unsigned worker) {
    auto *partials = reinterpret_cast<float2 *>(&scratch[0]);
    const auto *x = reinterpret_cast<const half2 *>(&source[0]);
    if (stage == 0) {
      float2 sum = {0, 0};
      for (unsigned i = worker; i < width / 2; i += 6)
        sum += __builtin_convertvector(x[i], float2);
      partials[worker] = sum;
      return true;
    }
    float2 sum = {0, 0};
    for (unsigned i = 0; i < 6; ++i) sum += partials[i];
    const float mean = (sum[0] + sum[1]) / width;
    if (stage == 1) {
      float2 variance = {0, 0};
      for (unsigned i = worker; i < width / 2; i += 6) {
        const float2 d = __builtin_convertvector(x[i], float2) - mean;
        variance += d * d;
      }
      partials[6 + worker] = variance;
    } else if (worker == 0) {
      float2 variance = {0, 0};
      for (unsigned i = 6; i < 12; ++i) variance += partials[i];
      *reinterpret_cast<float2 *>(&destination[0]) = {mean, variance[0] + variance[1]};
    }
    return true;
  }
};
#endif

#ifdef VERTEX_LayerNormApply
class LayerNormApply : public MultiVertex {
public:
  Input<Vector<half, VectorLayout::ONE_PTR>> source, scale, bias;
  Input<Vector<float, VectorLayout::ONE_PTR>> moments;
  Output<Vector<half, VectorLayout::ONE_PTR>> destination;
  unsigned rows, width, parts;
  bool compute(unsigned worker) {
    const auto *x = reinterpret_cast<const half2 *>(&source[0]);
    const auto *gamma = reinterpret_cast<const half2 *>(&scale[0]);
    const auto *beta = reinterpret_cast<const half2 *>(&bias[0]);
    const auto *stats = reinterpret_cast<const float2 *>(&moments[0]);
    auto *y = reinterpret_cast<half2 *>(&destination[0]);
    for (unsigned row = 0; row < rows; ++row) {
      float mean = 0;
      for (unsigned part = 0; part < parts; ++part) mean += stats[row * parts + part][0];
      mean /= parts;
      float variance = 0;
      for (unsigned part = 0; part < parts; ++part) {
        const float2 group = stats[row * parts + part];
        const float delta = group[0] - mean;
        variance += group[1] + width * delta * delta;
      }
      const float inverse = 1.0f / std::sqrt(variance / (width * parts) + 1e-6f);
      for (unsigned i = worker; i < width / 2; i += 6) {
        const unsigned at = row * (width / 2) + i;
        const float2 value = (__builtin_convertvector(x[at], float2) - mean) * inverse;
        y[at] = __builtin_convertvector(value * __builtin_convertvector(gamma[i], float2)
                                      + __builtin_convertvector(beta[i], float2), half2);
      }
    }
    return true;
  }
};
#endif
