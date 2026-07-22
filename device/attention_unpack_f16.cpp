#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

using namespace poplar;

#ifndef ATTENTION_HEAD_DIMENSION
#define ATTENTION_HEAD_DIMENSION 64
#endif

class AttentionUnpackHeadF16 : public MultiVertex {
public:
  Input<Vector<half, VectorLayout::ONE_PTR>> source;
  Input<Vector<half, VectorLayout::ONE_PTR>> unused;
  Output<Vector<half, VectorLayout::ONE_PTR>> output;
  unsigned rows;
  unsigned columnStart;

  bool compute(unsigned worker) {
    for (unsigned row = worker; row < rows; row += 6) {
      for (unsigned column = 0; column < ATTENTION_HEAD_DIMENSION;
           column += 4) {
        const unsigned outputColumn = columnStart + column;
        const unsigned destination = (outputColumn / 16) * rows * 16 +
                                     row * 16 + outputColumn % 16;
        *reinterpret_cast<half4 *>(&output[destination]) =
            *reinterpret_cast<const half4 *>(
                &source[row * ATTENTION_HEAD_DIMENSION + column]);
      }
    }
    return true;
  }
};
