#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>
#include "elementwise_vector.hpp"
using namespace poplar;

#ifdef VERTEX_LayerNormMoments
#ifdef NORM_STORE_SUM
#define MOMENTS_VERTEX AddLayerNormMoments
#else
#define MOMENTS_VERTEX LayerNormMoments
#endif
class MOMENTS_VERTEX : public MultiVertex {
public:
  Input<Vector<half, VectorLayout::ONE_PTR>> source;
  Output<Vector<float, VectorLayout::ONE_PTR>> destination;
  unsigned width;
  InOut<Vector<float, VectorLayout::ONE_PTR>> scratch;
  unsigned stage;
#ifdef NORM_STORE_SUM
  Input<Vector<half, VectorLayout::ONE_PTR>> right;
  Output<Vector<half, VectorLayout::ONE_PTR>> residual;
#endif
  bool compute(unsigned worker) {
    auto *partials = reinterpret_cast<float2 *>(&scratch[0]);
    const half *x = &source[0];
    const unsigned width = this->width;
    if (stage == 0) {
#ifdef NORM_STORE_SUM
      partials[worker] = normAddAndStore(x, &right[0], &residual[0], width, worker);
#else
      partials[worker] = normSum(x, nullptr, width, worker, 0, false);
#endif
      return true;
    }
#ifdef NORM_STORE_SUM
    x = &residual[0];
#endif
    float2 sum = {0, 0};
    for (unsigned i = 0; i < 6; ++i) sum += partials[i];
    const float mean = (sum[0] + sum[1]) / width;
    if (stage == 1) {
      partials[6 + worker] = normSum(x, nullptr, width, worker, mean, true);
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
    const half *x = &source[0];
    const half *gamma = &scale[0];
    const half *beta = &bias[0];
    const auto *stats = reinterpret_cast<const float2 *>(&moments[0]);
    half *y = &destination[0];
    const unsigned rows = this->rows, width = this->width, parts = this->parts;
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
      const float inverse = normInverse(variance / (width * parts) + 1e-6f);
      normApply(x + row * width, nullptr, gamma, beta, y + row * width,
                width, worker, mean, inverse);
    }
    return true;
  }
};
#endif
