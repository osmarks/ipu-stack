#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

using namespace poplar;

#ifndef REARRANGE_LOGICAL_COLUMNS
#define REARRANGE_LOGICAL_COLUMNS 64
#endif
#ifndef REARRANGE_PHYSICAL_COLUMNS
#define REARRANGE_PHYSICAL_COLUMNS REARRANGE_LOGICAL_COLUMNS
#endif
#ifndef REARRANGE_VERTEX_NAME
#define REARRANGE_VERTEX_NAME RearrangeRowMajorToAmpF16
#endif
#ifndef REARRANGE_INNER_DIMENSION
#define REARRANGE_INNER_DIMENSION 16
#endif
#ifndef REARRANGE_LOGICAL_ROWS
#define REARRANGE_LOGICAL_ROWS 1
#endif
#ifndef REARRANGE_PHYSICAL_ROWS
#define REARRANGE_PHYSICAL_ROWS REARRANGE_LOGICAL_ROWS
#endif
#ifndef REARRANGE_TARGET_ORDER
#define REARRANGE_TARGET_ORDER 0
#endif
#ifndef REARRANGE_ROW_BLOCK
#define REARRANGE_ROW_BLOCK 64
#endif
#ifndef REARRANGE_COLUMN_BLOCK
#define REARRANGE_COLUMN_BLOCK 16
#endif

static_assert(REARRANGE_LOGICAL_COLUMNS % 2 == 0);

class REARRANGE_VERTEX_NAME : public MultiVertex {
public:
  Input<Vector<half, VectorLayout::ONE_PTR>> source;
  Output<Vector<half, VectorLayout::ONE_PTR>> destination;
  unsigned logicalRows;
  unsigned physicalRows;
  unsigned targetOrder;
  unsigned logicalColumns;
  unsigned physicalColumns;

  bool compute(unsigned worker) {
    constexpr unsigned inner = REARRANGE_INNER_DIMENSION;
    constexpr unsigned innerBlock = REARRANGE_ROW_BLOCK;
    constexpr unsigned columnBlock = REARRANGE_COLUMN_BLOCK;
#if REARRANGE_LOGICAL_ROWS == 0
    const unsigned logicalRowCount = logicalRows;
#else
    constexpr unsigned logicalRowCount = REARRANGE_LOGICAL_ROWS;
#endif
#if REARRANGE_PHYSICAL_ROWS == 0
    const unsigned physicalRowCount = physicalRows;
#else
    constexpr unsigned physicalRowCount = REARRANGE_PHYSICAL_ROWS;
#endif
#if REARRANGE_LOGICAL_COLUMNS == 0
    const unsigned logicalColumnCount = logicalColumns;
#else
    constexpr unsigned logicalColumnCount = REARRANGE_LOGICAL_COLUMNS;
#endif
#if REARRANGE_PHYSICAL_COLUMNS == 0
    const unsigned physicalColumnCount = physicalColumns;
#else
    constexpr unsigned physicalColumnCount = REARRANGE_PHYSICAL_COLUMNS;
#endif
    const unsigned *sourceWords = reinterpret_cast<const unsigned *>(&source[0]);
    unsigned *destinationWords = reinterpret_cast<unsigned *>(&destination[0]);
#if REARRANGE_TARGET_ORDER < 2
      for (unsigned row = worker; row < physicalRowCount; row += 6) {
#if REARRANGE_TARGET_ORDER == 1
        const unsigned panelRow =
            (row / columnBlock) * (physicalColumnCount / inner);
        for (unsigned column = 0; column < physicalColumnCount;
             column += inner) {
          const unsigned panel = panelRow + column / inner;
          const unsigned destinationWord =
              (panel * inner * columnBlock + row % columnBlock * inner) / 2;
          const unsigned sourceWord =
              (row * logicalColumnCount + column) / 2;
          for (unsigned word = 0; word < inner / 2; ++word) {
            destinationWords[destinationWord + word] =
                row < logicalRowCount &&
                        column + word * 2 < logicalColumnCount
                    ? sourceWords[sourceWord + word]
                    : 0;
          }
        }
#else
        for (unsigned column = 0; column < physicalColumnCount;
             column += 2) {
          unsigned low = 0;
          unsigned high = 0;
          if (row < logicalRowCount && column < logicalColumnCount) {
            const unsigned sourceElement =
                row * logicalColumnCount + column;
            low = sourceWords[sourceElement / 2] & 0xffff;
            high = sourceWords[sourceElement / 2] >> 16;
          }
          unsigned physical;
#if REARRANGE_TARGET_ORDER == 0
            physical = (column / inner) * physicalRowCount * inner + row * inner +
                       column % inner;
#else
            const unsigned panel =
                (row / columnBlock) * (physicalColumnCount / inner) +
                column / inner;
            physical = panel * inner * columnBlock + loadChannel * inner +
                       column % inner;
#endif
          destinationWords[physical / 2] = low | (high << 16);
        }
#endif
      }
      return true;
#else
    const unsigned rowPairs = (physicalRowCount + 1) / 2;
    for (unsigned rowPair = worker; rowPair < rowPairs; rowPair += 6) {
      const unsigned row = rowPair * 2;
      for (unsigned column = 0; column < physicalColumnCount;
           column += 2) {
        unsigned sourceRow = 0;
        unsigned sourceNextRow = 0;
        if (row < logicalRowCount && column < logicalColumnCount) {
          const unsigned sourceWord =
              (row * logicalColumnCount + column) / 2;
          sourceRow = sourceWords[sourceWord];
          if (row + 1 < logicalRowCount)
            sourceNextRow =
                sourceWords[sourceWord + logicalColumnCount / 2];
        }
        const unsigned packedEven =
            (sourceRow & 0xffff) | (sourceNextRow << 16);
        const unsigned packedOdd =
            (sourceRow >> 16) | (sourceNextRow & 0xffff0000);

        const unsigned innerGroup = row % innerBlock / inner;
        const unsigned panel =
          (row / innerBlock) * (physicalColumnCount / columnBlock) *
              (innerBlock / inner) +
          (column / columnBlock) * (innerBlock / inner) + innerGroup;
        const unsigned physicalBase = panel * inner * columnBlock + row % inner;
        destinationWords[(physicalBase + column % columnBlock * inner) / 2] =
            packedEven;
        destinationWords[(physicalBase + (column % columnBlock + 1) * inner) / 2] =
            packedOdd;
      }
    }
    return true;
#endif
  }
};
