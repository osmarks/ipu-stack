#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

using namespace poplar;

class AddF32 : public MultiVertex {
public:
  Input<Vector<float, VectorLayout::ONE_PTR>> left;
  Input<Vector<float, VectorLayout::ONE_PTR>> right;
  Output<Vector<float, VectorLayout::ONE_PTR>> output;
  unsigned elements;

  bool compute(unsigned worker) {
    for (unsigned index = worker; index < elements; index += 6)
      output[index] = left[index] + right[index];
    return true;
  }
};

class GeluF32 : public MultiVertex {
public:
  Input<Vector<float, VectorLayout::ONE_PTR>> input;
  Output<Vector<float, VectorLayout::ONE_PTR>> output;
  unsigned elements;

  bool compute(unsigned worker) {
    for (unsigned index = worker; index < elements; index += 6) {
      const float value = input[index];
      const float cubic = value * value * value;
      output[index] =
          0.5f * value * (1.0f + __builtin_tanhf(0.7978846f *
                                                (value + 0.044715f * cubic)));
    }
    return true;
  }
};

class MatMulF16 : public MultiVertex {
public:
  Input<Vector<half, VectorLayout::ONE_PTR>> left;
  Input<Vector<half, VectorLayout::ONE_PTR>> right;
  Output<Vector<half, VectorLayout::ONE_PTR>> output;
  unsigned rows;
  unsigned inner;
  unsigned columns;

  bool compute(unsigned worker) {
    const unsigned elements = rows * columns;
    for (unsigned index = worker; index < elements; index += 6) {
      const unsigned row = index / columns;
      const unsigned column = index - row * columns;
      float sum = 0.0f;
      for (unsigned k = 0; k < inner; ++k)
        sum += static_cast<float>(left[row * inner + k]) *
               static_cast<float>(right[k * columns + column]);
      output[index] = static_cast<half>(sum);
    }
    return true;
  }
};

class LayerNormF32 : public MultiVertex {
public:
  Input<Vector<float, VectorLayout::ONE_PTR>> input;
  Output<Vector<float, VectorLayout::ONE_PTR>> output;
  unsigned rows;
  unsigned columns;
  float epsilon;

  bool compute(unsigned worker) {
    for (unsigned row = worker; row < rows; row += 6) {
      float mean = 0.0f;
      for (unsigned column = 0; column < columns; ++column)
        mean += input[row * columns + column];
      mean /= static_cast<float>(columns);
      float variance = 0.0f;
      for (unsigned column = 0; column < columns; ++column) {
        const float centered = input[row * columns + column] - mean;
        variance += centered * centered;
      }
      const float scale =
          1.0f / __builtin_sqrtf(variance / static_cast<float>(columns) +
                                 epsilon);
      for (unsigned column = 0; column < columns; ++column)
        output[row * columns + column] =
            (input[row * columns + column] - mean) * scale;
    }
    return true;
  }
};
