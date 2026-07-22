#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

using namespace poplar;

#ifndef ATTENTION_HEAD_DIMENSION
#define ATTENTION_HEAD_DIMENSION 72
#endif
#ifndef ATTENTION_PADDED_HEAD_DIMENSION
#define ATTENTION_PADDED_HEAD_DIMENSION 80
#endif
#ifndef ATTENTION_KEY_BLOCK_COLUMNS
#define ATTENTION_KEY_BLOCK_COLUMNS 16
#endif

constexpr unsigned packedRowOffsetBits = 10;
constexpr unsigned packedRowOffsetMask = (1u << packedRowOffsetBits) - 1;

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
  unsigned sourceRows;
  unsigned sourceOffset;
  unsigned rowOffsets;
  unsigned copyRows;
  unsigned destinationRows;

  bool compute(unsigned worker) {
    const unsigned headStart = sourceOffset;
    const unsigned sourceRowStart = rowOffsets & packedRowOffsetMask;
    const unsigned destinationRowStart = rowOffsets >> packedRowOffsetBits;
    for (unsigned localRow = worker; localRow < copyRows; localRow += 6) {
      const unsigned sourceRow = sourceRowStart + localRow;
      const unsigned destinationRow = destinationRowStart + localRow;
      for (unsigned panel = 0; panel < ATTENTION_PADDED_HEAD_DIMENSION / 16;
           ++panel) {
        const unsigned outputBase =
            panel * destinationRows * 16 + destinationRow * 16;
        for (unsigned column = 0; column < 16; column += 4) {
          const unsigned logicalColumn = panel * 16 + column;
          half4 packed = {};
          if (logicalColumn + 3 < ATTENTION_HEAD_DIMENSION) {
            const unsigned input =
                sourceIndex(sourceRows, sourceRow, headStart + logicalColumn);
            packed = *reinterpret_cast<const half4 *>(&source[input]);
          }
          *reinterpret_cast<half4 *>(&output[outputBase + column]) = packed;
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
  unsigned sourceRows;
  unsigned sourceOffset;
  unsigned rowOffsets;
  unsigned copyRows;
  unsigned destinationRows;

  bool compute(unsigned worker) {
    const unsigned headStart = sourceOffset;
    const unsigned sourceRowStart = rowOffsets & packedRowOffsetMask;
    const unsigned destinationRowStart = rowOffsets >> packedRowOffsetBits;
    constexpr unsigned innerGroups = ATTENTION_PADDED_HEAD_DIMENSION / 16;
    for (unsigned loadChannel = worker; loadChannel < 16; loadChannel += 6) {
      const unsigned physicalPair = loadChannel / 2;
      const unsigned rowInGroup =
          logicalPairForPhysical(physicalPair) * 2 + loadChannel % 2;
      for (unsigned rowGroup = 0;
           rowGroup < ATTENTION_KEY_BLOCK_COLUMNS / 16; ++rowGroup) {
        const unsigned destinationRow = rowGroup * 16 + rowInGroup;
        if (destinationRow < destinationRowStart ||
            destinationRow >= destinationRowStart + copyRows)
          continue;
        const unsigned sourceRow =
            sourceRowStart + destinationRow - destinationRowStart;
        for (unsigned innerGroup = 0; innerGroup < innerGroups;
             ++innerGroup) {
          const unsigned outputBase =
              (rowGroup * innerGroups + innerGroup) * 256 + loadChannel * 16;
          for (unsigned inner = 0; inner < 16; inner += 4) {
            const unsigned logicalInner = innerGroup * 16 + inner;
            half4 packed = {};
            if (logicalInner + 3 < ATTENTION_HEAD_DIMENSION) {
              const unsigned input =
                  sourceIndex(sourceRows, sourceRow, headStart + logicalInner);
              packed = *reinterpret_cast<const half4 *>(&source[input]);
            }
            *reinterpret_cast<half4 *>(&output[outputBase + inner]) = packed;
          }
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
  unsigned sourceRows;
  unsigned sourceOffset;
  unsigned rowOffsets;
  unsigned copyRows;
  unsigned destinationRows;

  bool compute(unsigned worker) {
    const unsigned headStart = sourceOffset;
    const unsigned sourceRowStart = rowOffsets & packedRowOffsetMask;
    const unsigned destinationRowStart = rowOffsets >> packedRowOffsetBits;
    constexpr unsigned keyGroups = ATTENTION_KEY_BLOCK_COLUMNS / 16;
    for (unsigned loadChannel = worker; loadChannel < 16; loadChannel += 6) {
      const unsigned physicalPair = loadChannel / 2;
      const unsigned logicalColumnInPanel =
          logicalPairForPhysical(physicalPair) * 2 + loadChannel % 2;
      for (unsigned panel = 0;
           panel < ATTENTION_PADDED_HEAD_DIMENSION / 16; ++panel) {
        const unsigned logicalColumn = panel * 16 + logicalColumnInPanel;
        unsigned localRow = 0;
        if ((destinationRowStart & 1u) != 0 && localRow < copyRows) {
          const unsigned destinationRow = destinationRowStart + localRow;
          const unsigned keyGroup = destinationRow / 16;
          const unsigned row = destinationRow % 16;
          const unsigned outputBase =
              (panel * keyGroups + keyGroup) * 256 + loadChannel * 16;
          const unsigned pairRow = row & ~1u;
          half2 values =
              *reinterpret_cast<const half2 *>(&output[outputBase + pairRow]);
          const unsigned lane = row & 1u;
          values[lane] = 0.0;
          if (logicalColumn < ATTENTION_HEAD_DIMENSION) {
            const unsigned sourceRow = sourceRowStart + localRow;
            values[lane] = source[sourceIndex(
                sourceRows, sourceRow, headStart + logicalColumn)];
          }
          *reinterpret_cast<half2 *>(&output[outputBase + pairRow]) = values;
          ++localRow;
        }
        for (; localRow + 1 < copyRows; localRow += 2) {
          const unsigned destinationRow = destinationRowStart + localRow;
          const unsigned keyGroup = destinationRow / 16;
          const unsigned row = destinationRow % 16;
          const unsigned outputBase =
              (panel * keyGroups + keyGroup) * 256 + loadChannel * 16;
          half2 values = {};
          if (logicalColumn < ATTENTION_HEAD_DIMENSION) {
            const unsigned sourceRow = sourceRowStart + localRow;
            values[0] = source[sourceIndex(sourceRows, sourceRow,
                                           headStart + logicalColumn)];
            values[1] = source[sourceIndex(sourceRows, sourceRow + 1,
                                           headStart + logicalColumn)];
          }
          *reinterpret_cast<half2 *>(&output[outputBase + row]) = values;
        }
        if (localRow < copyRows) {
          const unsigned destinationRow = destinationRowStart + localRow;
          const unsigned keyGroup = destinationRow / 16;
          const unsigned row = destinationRow % 16;
          const unsigned outputBase =
              (panel * keyGroups + keyGroup) * 256 + loadChannel * 16;
          const unsigned pairRow = row & ~1u;
          half2 values =
              *reinterpret_cast<const half2 *>(&output[outputBase + pairRow]);
          const unsigned lane = row & 1u;
          values[lane] = 0.0;
          if (logicalColumn < ATTENTION_HEAD_DIMENSION) {
            const unsigned sourceRow = sourceRowStart + localRow;
            values[lane] = source[sourceIndex(
                sourceRows, sourceRow, headStart + logicalColumn)];
          }
          *reinterpret_cast<half2 *>(&output[outputBase + pairRow]) = values;
        }
      }
    }
    return true;
  }
};
