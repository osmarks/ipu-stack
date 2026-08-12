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

static_assert(REARRANGE_LOGICAL_COLUMNS % 2 == 0);

class REARRANGE_VERTEX_NAME : public MultiVertex {
public:
  Input<Vector<half, VectorLayout::ONE_PTR>> source;
  Output<Vector<half, VectorLayout::ONE_PTR>> destination;
  unsigned logicalRows;
  unsigned physicalRows;
  unsigned targetOrder;

  bool compute(unsigned worker) {
    constexpr unsigned inner = REARRANGE_INNER_DIMENSION;
    constexpr unsigned innerBlock = 64;
    constexpr unsigned columnBlock = 16;
    const unsigned *sourceWords = reinterpret_cast<const unsigned *>(&source[0]);
    unsigned *destinationWords = reinterpret_cast<unsigned *>(&destination[0]);
#if REARRANGE_TARGET_ORDER < 2
      for (unsigned row = worker; row < REARRANGE_PHYSICAL_ROWS; row += 6) {
#if REARRANGE_TARGET_ORDER == 1
        const unsigned logicalPair = row % columnBlock / 2;
        const unsigned loadPair = logicalPair % 4 * 2 + logicalPair / 4;
        const unsigned loadChannel = loadPair * 2 + row % 2;
        const unsigned panelRow =
            (row / columnBlock) * (REARRANGE_PHYSICAL_COLUMNS / inner);
        for (unsigned column = 0; column < REARRANGE_PHYSICAL_COLUMNS;
             column += inner) {
          const unsigned panel = panelRow + column / inner;
          const unsigned destinationWord =
              (panel * inner * columnBlock + loadChannel * inner) / 2;
          const unsigned sourceWord =
              (row * REARRANGE_LOGICAL_COLUMNS + column) / 2;
          for (unsigned word = 0; word < inner / 2; ++word) {
            destinationWords[destinationWord + word] =
                row < REARRANGE_LOGICAL_ROWS &&
                        column + word * 2 < REARRANGE_LOGICAL_COLUMNS
                    ? sourceWords[sourceWord + word]
                    : 0;
          }
        }
#else
        const unsigned logicalPair = row % columnBlock / 2;
        const unsigned loadPair = logicalPair % 4 * 2 + logicalPair / 4;
        const unsigned loadChannel = loadPair * 2 + row % 2;
        for (unsigned column = 0; column < REARRANGE_PHYSICAL_COLUMNS;
             column += 2) {
          unsigned low = 0;
          unsigned high = 0;
          if (row < REARRANGE_LOGICAL_ROWS && column < REARRANGE_LOGICAL_COLUMNS) {
            const unsigned sourceElement =
                row * REARRANGE_LOGICAL_COLUMNS + column;
            low = sourceWords[sourceElement / 2] & 0xffff;
            high = sourceWords[sourceElement / 2] >> 16;
          }
          unsigned physical;
#if REARRANGE_TARGET_ORDER == 0
            physical = (column / inner) * REARRANGE_PHYSICAL_ROWS * inner + row * inner +
                       column % inner;
#else
            const unsigned panel =
                (row / columnBlock) * (REARRANGE_PHYSICAL_COLUMNS / inner) +
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
    constexpr unsigned rowPairs = (REARRANGE_PHYSICAL_ROWS + 1) / 2;
    for (unsigned rowPair = worker; rowPair < rowPairs; rowPair += 6) {
      const unsigned row = rowPair * 2;
      for (unsigned column = 0; column < REARRANGE_PHYSICAL_COLUMNS;
           column += 2) {
        unsigned sourceRow = 0;
        unsigned sourceNextRow = 0;
        if (row < REARRANGE_LOGICAL_ROWS &&
            column < REARRANGE_LOGICAL_COLUMNS) {
          const unsigned sourceWord =
              (row * REARRANGE_LOGICAL_COLUMNS + column) / 2;
          sourceRow = sourceWords[sourceWord];
          if (row + 1 < REARRANGE_LOGICAL_ROWS)
            sourceNextRow =
                sourceWords[sourceWord + REARRANGE_LOGICAL_COLUMNS / 2];
        }
        const unsigned packedEven =
            (sourceRow & 0xffff) | (sourceNextRow << 16);
        const unsigned packedOdd =
            (sourceRow >> 16) | (sourceNextRow & 0xffff0000);

        const unsigned logicalPair = column % columnBlock / 2;
        const unsigned loadPair = logicalPair % 4 * 2 + logicalPair / 4;
        const unsigned innerGroup = row % innerBlock / inner;
        const unsigned panel =
          (row / innerBlock) * (REARRANGE_PHYSICAL_COLUMNS / columnBlock) *
              (innerBlock / inner) +
          (column / columnBlock) * (innerBlock / inner) + innerGroup;
        const unsigned physicalBase = panel * inner * columnBlock + row % inner;
        destinationWords[(physicalBase + loadPair * 2 * inner) / 2] =
            packedEven;
        destinationWords[(physicalBase + (loadPair * 2 + 1) * inner) / 2] =
            packedOdd;
      }
    }
    return true;
#endif
  }
};
