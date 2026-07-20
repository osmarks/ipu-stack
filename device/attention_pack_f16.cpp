#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

using namespace poplar;

#ifndef ATTENTION_HEAD_DIMENSION
#define ATTENTION_HEAD_DIMENSION 72
#endif
#ifndef ATTENTION_PADDED_HEAD_DIMENSION
#define ATTENTION_PADDED_HEAD_DIMENSION 80
#endif

static __attribute__((always_inline)) unsigned
sourceIndex(unsigned rows, unsigned row, unsigned column) {
  return (column / 16) * rows * 16 + row * 16 + column % 16;
}

static __attribute__((always_inline)) unsigned
logicalPairForPhysical(unsigned physicalPair) {
  return (physicalPair % 2) * 4 + physicalPair / 2;
}

class AttentionPackQueryF16 : public MultiVertex {
public:
  Input<Vector<half, VectorLayout::ONE_PTR>> source;
  Input<Vector<half, VectorLayout::ONE_PTR>> unused;
  Output<Vector<half, VectorLayout::ONE_PTR>> output;
  unsigned rows;
  unsigned sourceColumns;
  unsigned sourceOffset;

  bool compute(unsigned worker) {
    const unsigned headStart = sourceOffset;
    for (unsigned row = worker; row < rows; row += 6) {
      for (unsigned panel = 0; panel < ATTENTION_PADDED_HEAD_DIMENSION / 16;
           ++panel) {
        const unsigned outputBase = panel * rows * 16 + row * 16;
        for (unsigned column = 0; column < 16; column += 2) {
          const unsigned logicalColumn = panel * 16 + column;
          half2 packed = {};
          if (logicalColumn < ATTENTION_HEAD_DIMENSION) {
            const unsigned input =
                sourceIndex(rows, row, headStart + logicalColumn);
            packed = *reinterpret_cast<const half2 *>(&source[input]);
          }
          *reinterpret_cast<half2 *>(&output[outputBase + column]) = packed;
        }
      }
    }
    return true;
  }
};

class AttentionPackKeyF16 : public MultiVertex {
public:
  Input<Vector<half, VectorLayout::ONE_PTR>> source;
  Input<Vector<half, VectorLayout::ONE_PTR>> unused;
  Output<Vector<half, VectorLayout::ONE_PTR>> output;
  unsigned rows;
  unsigned sourceColumns;
  unsigned sourceOffset;

  bool compute(unsigned worker) {
    const unsigned headStart = sourceOffset;
    for (unsigned loadChannel = worker; loadChannel < 16; loadChannel += 6) {
      const unsigned physicalPair = loadChannel / 2;
      const unsigned keyColumn =
          logicalPairForPhysical(physicalPair) * 2 + loadChannel % 2;
      for (unsigned innerGroup = 0;
           innerGroup < ATTENTION_PADDED_HEAD_DIMENSION / 16; ++innerGroup) {
        const unsigned outputBase =
            innerGroup * 256 + loadChannel * 16;
        for (unsigned inner = 0; inner < 16; inner += 2) {
          const unsigned logicalInner = innerGroup * 16 + inner;
          half2 packed = {};
          if (keyColumn < rows && logicalInner < ATTENTION_HEAD_DIMENSION) {
            const unsigned input =
                sourceIndex(rows, keyColumn, headStart + logicalInner);
            packed = *reinterpret_cast<const half2 *>(&source[input]);
          }
          *reinterpret_cast<half2 *>(&output[outputBase + inner]) = packed;
        }
      }
    }
    return true;
  }
};

class AttentionPackValueF16 : public MultiVertex {
public:
  Input<Vector<half, VectorLayout::ONE_PTR>> source;
  Input<Vector<half, VectorLayout::ONE_PTR>> unused;
  Output<Vector<half, VectorLayout::ONE_PTR>> output;
  unsigned rows;
  unsigned sourceColumns;
  unsigned sourceOffset;

  bool compute(unsigned worker) {
    const unsigned headStart = sourceOffset;
    for (unsigned loadChannel = worker; loadChannel < 16; loadChannel += 6) {
      const unsigned physicalPair = loadChannel / 2;
      const unsigned logicalColumnInPanel =
          logicalPairForPhysical(physicalPair) * 2 + loadChannel % 2;
      for (unsigned panel = 0;
           panel < ATTENTION_PADDED_HEAD_DIMENSION / 16; ++panel) {
        const unsigned logicalColumn = panel * 16 + logicalColumnInPanel;
        const unsigned outputBase = panel * 256 + loadChannel * 16;
        for (unsigned row = 0; row < 16; row += 2) {
          float2 values = {0.0f, 0.0f};
          if (logicalColumn < ATTENTION_HEAD_DIMENSION) {
            if (row < rows)
              values[0] = static_cast<float>(
                  source[sourceIndex(rows, row, headStart + logicalColumn)]);
            if (row + 1 < rows)
              values[1] = static_cast<float>(source[
                  sourceIndex(rows, row + 1, headStart + logicalColumn)]);
          }
          *reinterpret_cast<half2 *>(&output[outputBase + row]) =
              __builtin_convertvector(values, half2);
        }
      }
    }
    return true;
  }
};
