#include <iostream>
#include <vector>
#include <limits.h>
#include <math.h>
#include <cassert>
#include <algorithm>
#include <map>

#include <mpi.h>

#define MASTER 0

// Forward declarations
void parallelFloydWarshall(
    std::vector<std::vector<int>> graph,
    int taskId,
    std::pair<int, int> upperLeftCoordinates,
    std::pair<int, int> bottomRightCoordinates,
    int rowId,
    int columnId,
    int sqrtP
);


// Replace INF with a big number, be careful of overflows in case you use INT_MAX (addition of 2 INT_MAX will cause overflows)
// Instead, maybe use INF = (INT_MAX - 2)/2 or INT_MAX/10 etc etc;
const int INF = INT_MAX / 10;

std::vector<std::vector<int>> theGraph = {
      {0,   1, INF, INF,   4, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF},
    {INF,   0,   2, INF,   3, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF},
    {INF, INF,   0,   3,   2, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF},
    {INF, INF, INF,   0,   4, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF},
    {INF, INF, INF, INF,   0,   5,   2, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,  20, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF},
     { 6, INF, INF, INF, INF,   0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF},
    {INF, INF, INF, INF, INF, INF,   0,   1, INF, INF, INF, INF,   6, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF},
    {INF, INF, INF, INF, INF, INF, INF,   0,   2, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF},
    {INF, INF, INF, INF, INF, INF, INF, INF,   0,   3,   2, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF},
    {INF, INF, INF, INF, INF, INF, INF, INF, INF,   0,   4, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF},
    {INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,   0,   5, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF},
    {INF, INF, INF, INF, INF, INF,   6, INF,   3, INF, INF,   0,   1, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF},
    {INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,   0,   1, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF},
    {INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,   0,   2, INF,   3, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF},
    {INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,   0,   3, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF},
    {INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,   0,   4, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF},
    {INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,   0,   5, INF, INF, INF, INF, INF,  10, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF},
    {INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,   6, INF, INF, INF, INF,   0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF},
    {INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,   0, INF,   2, INF, INF,   6, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF},
    {INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,   5,   0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF},
    {INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,   4,   0, INF,   2, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF},
    {INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,   3,   0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,  14},
    {INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,   2,   0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF},
    {INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,   2,   1,   0, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF},
    {INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,   0,   1, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF},
     {25, INF, INF,  10, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,   0,   2, INF, INF, INF, INF, INF, INF, INF, INF, INF},
    {INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,   0,   1,   2, INF, INF, INF, INF, INF, INF, INF},
    {INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,   0,   3, INF, INF, INF, INF, INF, INF, INF},
    {INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,   4,   3, INF, INF,   0, INF, INF, INF, INF, INF, INF, INF},
    {INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,   0, INF, INF, INF, INF,   6, INF},
    {INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,   2,   5,   0, INF, INF, INF, INF, INF},
    {INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,   4,   0, INF, INF, INF, INF},
    {INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,   2,   3,   0, INF, INF, INF},
    {INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,   2,   0, INF, INF},
    {INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,   3, INF,   1,   0, INF},
    {INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF, INF,   1,   0}
};

std::vector<std::vector<int>> answer = {
    { 0, 1, 3, 6, 4, 9, 6, 7, 9, 12, 11, 16, 12, 13, 15, 18, 16, 21, 40, 35, 31, 28, 27, 26, 24, 25, 27, 28, 29, 53, 48, 46, 46, 44, 43, 42       },
    { 14, 0, 2, 5, 3, 8, 5, 6, 8, 11, 10, 15, 11, 12, 14, 17, 15, 20, 39, 34, 30, 27, 26, 25, 23, 24, 26, 27, 28, 52, 47, 45, 45, 43, 42, 41      },
    { 13, 14, 0, 3, 2, 7, 4, 5, 7, 10, 9, 14, 10, 11, 13, 16, 14, 19, 38, 33, 29, 26, 25, 24, 22, 23, 25, 26, 27, 51, 46, 44, 44, 42, 41, 40      },
    { 15, 16, 18, 0, 4, 9, 6, 7, 9, 12, 11, 16, 12, 13, 15, 18, 16, 21, 40, 35, 31, 28, 27, 26, 24, 25, 27, 28, 29, 53, 48, 46, 46, 44, 43, 42    },
    { 11, 12, 14, 17, 0, 5, 2, 3, 5, 8, 7, 12, 8, 9, 11, 14, 12, 17, 36, 31, 27, 24, 23, 22, 20, 21, 23, 24, 25, 49, 44, 42, 42, 40, 39, 38       },
    { 6, 7, 9, 12, 10, 0, 12, 13, 15, 18, 17, 22, 18, 19, 21, 24, 22, 27, 46, 41, 37, 34, 33, 32, 30, 31, 33, 34, 35, 59, 54, 52, 52, 50, 49, 48  },
    { 72, 73, 75, 57, 61, 66, 0, 1, 3, 6, 5, 10, 6, 7, 9, 12, 10, 15, 34, 29, 25, 22, 21, 20, 48, 47, 49, 50, 44, 47, 42, 40, 40, 38, 37, 36      },
    { 76, 77, 79, 61, 65, 70, 15, 0, 2, 5, 4, 9, 10, 11, 13, 16, 14, 19, 38, 33, 29, 26, 25, 24, 52, 51, 53, 54, 48, 51, 46, 44, 44, 42, 41, 40   },
    { 74, 75, 77, 59, 63, 68, 13, 14, 0, 3, 2, 7, 8, 9, 11, 14, 12, 17, 36, 31, 27, 24, 23, 22, 50, 49, 51, 52, 46, 49, 44, 42, 42, 40, 39, 38    },
    { 76, 77, 79, 61, 65, 70, 15, 16, 12, 0, 4, 9, 10, 11, 13, 16, 14, 19, 38, 33, 29, 26, 25, 24, 52, 51, 53, 54, 48, 51, 46, 44, 44, 42, 41, 40 },
    { 72, 73, 75, 57, 61, 66, 11, 12, 8, 11, 0, 5, 6, 7, 9, 12, 10, 15, 34, 29, 25, 22, 21, 20, 48, 47, 49, 50, 44, 47, 42, 40, 40, 38, 37, 36    },
    { 67, 68, 70, 52, 56, 61, 6, 7, 3, 6, 5, 0, 1, 2, 4, 7, 5, 10, 29, 24, 20, 17, 16, 15, 43, 42, 44, 45, 39, 42, 37, 35, 35, 33, 32, 31         },
    { 66, 67, 69, 51, 55, 60, 57, 58, 60, 63, 62, 67, 0, 1, 3, 6, 4, 9, 28, 23, 19, 16, 15, 14, 42, 41, 43, 44, 38, 41, 36, 34, 34, 32, 31, 30    },
    { 65, 66, 68, 50, 54, 59, 56, 57, 59, 62, 61, 66, 14, 0, 2, 5, 3, 8, 27, 22, 18, 15, 14, 13, 41, 40, 42, 43, 37, 40, 35, 33, 33, 31, 30, 29   },
    { 69, 70, 72, 54, 58, 63, 60, 61, 63, 66, 65, 70, 18, 19, 0, 3, 7, 12, 31, 26, 22, 19, 18, 17, 45, 44, 46, 47, 41, 44, 39, 37, 37, 35, 34, 33 },
    { 66, 67, 69, 51, 55, 60, 57, 58, 60, 63, 62, 67, 15, 16, 18, 0, 4, 9, 28, 23, 19, 16, 15, 14, 42, 41, 43, 44, 38, 41, 36, 34, 34, 32, 31, 30 },
    { 62, 63, 65, 47, 51, 56, 53, 54, 56, 59, 58, 63, 11, 12, 14, 17, 0, 5, 24, 19, 15, 12, 11, 10, 38, 37, 39, 40, 34, 37, 32, 30, 30, 28, 27, 26},
    { 72, 73, 75, 57, 61, 66, 63, 64, 66, 69, 68, 73, 6, 7, 9, 12, 10, 0, 34, 29, 25, 22, 21, 20, 48, 47, 49, 50, 44, 47, 42, 40, 40, 38, 37, 36  },
    { 56, 57, 59, 41, 45, 50, 47, 48, 50, 53, 52, 57, 53, 54, 56, 59, 57, 62, 0, 6, 2, 6, 4, 6, 32, 31, 33, 34, 28, 31, 26, 24, 24, 22, 21, 20    },
    { 61, 62, 64, 46, 50, 55, 52, 53, 55, 58, 57, 62, 58, 59, 61, 64, 62, 67, 5, 0, 7, 11, 9, 11, 37, 36, 38, 39, 33, 36, 31, 29, 29, 27, 26, 25  },
    { 54, 55, 57, 39, 43, 48, 45, 46, 48, 51, 50, 55, 51, 52, 54, 57, 55, 60, 9, 4, 0, 4, 2, 15, 30, 29, 31, 32, 26, 29, 24, 22, 22, 20, 19, 18   },
    { 50, 51, 53, 35, 39, 44, 41, 42, 44, 47, 46, 51, 47, 48, 50, 53, 51, 56, 12, 7, 3, 0, 5, 18, 26, 25, 27, 28, 22, 25, 20, 18, 18, 16, 15, 14  },
    { 52, 53, 55, 37, 41, 46, 43, 44, 46, 49, 48, 53, 49, 50, 52, 55, 53, 58, 14, 9, 5, 2, 0, 20, 28, 27, 29, 30, 24, 27, 22, 20, 20, 18, 17, 16  },
    { 52, 53, 55, 37, 41, 46, 43, 44, 46, 49, 48, 53, 49, 50, 52, 55, 53, 58, 14, 9, 5, 2, 1, 0, 28, 27, 29, 30, 24, 27, 22, 20, 20, 18, 17, 16   },
    { 26, 27, 29, 11, 15, 20, 17, 18, 20, 23, 22, 27, 23, 24, 26, 29, 27, 32, 51, 46, 42, 39, 38, 37, 0, 1, 3, 4, 5, 64, 59, 57, 57, 55, 54, 53   },
    { 25, 26, 28, 10, 14, 19, 16, 17, 19, 22, 21, 26, 22, 23, 25, 28, 26, 31, 50, 45, 41, 38, 37, 36, 8, 0, 2, 3, 4, 63, 58, 56, 56, 54, 53, 52   },
    { 30, 31, 33, 15, 19, 24, 21, 22, 24, 27, 26, 31, 27, 28, 30, 33, 31, 36, 55, 50, 46, 43, 42, 41, 6, 5, 0, 1, 2, 68, 63, 61, 61, 59, 58, 57   },
    { 31, 32, 34, 16, 20, 25, 22, 23, 25, 28, 27, 32, 28, 29, 31, 34, 32, 37, 56, 51, 47, 44, 43, 42, 7, 6, 8, 0, 3, 69, 64, 62, 62, 60, 59, 58   },
    { 28, 29, 31, 13, 17, 22, 19, 20, 22, 25, 24, 29, 25, 26, 28, 31, 29, 34, 53, 48, 44, 41, 40, 39, 4, 3, 5, 6, 0, 66, 61, 59, 59, 57, 56, 55   },
    { 41, 42, 44, 26, 30, 35, 32, 33, 35, 38, 37, 42, 38, 39, 41, 44, 42, 47, 66, 61, 57, 54, 53, 52, 17, 16, 18, 19, 13, 0, 11, 9, 9, 7, 6, 68   },
    { 30, 31, 33, 15, 19, 24, 21, 22, 24, 27, 26, 31, 27, 28, 30, 33, 31, 36, 55, 50, 46, 43, 42, 41, 6, 5, 7, 8, 2, 5, 0, 14, 14, 12, 11, 57     },
    { 34, 35, 37, 19, 23, 28, 25, 26, 28, 31, 30, 35, 31, 32, 34, 37, 35, 40, 59, 54, 50, 47, 46, 45, 10, 9, 11, 12, 6, 9, 4, 0, 18, 16, 15, 61   },
    { 32, 33, 35, 17, 21, 26, 23, 24, 26, 29, 28, 33, 29, 30, 32, 35, 33, 38, 57, 52, 48, 45, 44, 43, 8, 7, 9, 10, 4, 7, 2, 3, 0, 14, 13, 59      },
    { 34, 35, 37, 19, 23, 28, 25, 26, 28, 31, 30, 35, 31, 32, 34, 37, 35, 40, 59, 54, 50, 47, 46, 45, 10, 9, 11, 12, 6, 9, 4, 5, 2, 0, 15, 61     },
    { 35, 36, 38, 20, 24, 29, 26, 27, 29, 32, 31, 36, 32, 33, 35, 38, 36, 41, 60, 55, 51, 48, 47, 46, 11, 10, 12, 13, 7, 10, 5, 3, 3, 1, 0, 62    },
    { 36, 37, 39, 21, 25, 30, 27, 28, 30, 33, 32, 37, 33, 34, 36, 39, 37, 42, 61, 56, 52, 49, 48, 47, 12, 11, 13, 14, 8, 11, 6, 4, 4, 2, 1, 0     }
};

/*
    Parameters:

    matrix: Submatrix that you want to compare
    referenceMatrix: correct matrix against which we compare the input matrix
    taskId: MPI rank of the process
    graphStartRow & graphStartColumn: starting coordinates of the submatrix in the bigger matrix


*/
void printVectorContentsWithAssertions(
    std::vector<std::vector<int>> matrix,
    std::vector<std::vector<int>> referenceMatrix,
    int taskId,
    int graphStartRow,
    int graphStartColumn
)
{
    std::cout << "Process #" << taskId << " matrix contents:\n";
    std::cout << "[\n";
    int j = graphStartRow;
    int k = graphStartColumn;
    for (std::vector<int> row : matrix)
    {
        for (int i : row)
        {
            assert(i == referenceMatrix[j][k++]);
            std::cout << i << ", ";
        }
        k = graphStartColumn;
        ++j;
    }
    std::cout << "]\n";
}

void printVectorContents(
    std::vector<std::vector<int>> matrix,
    int taskId
)
{
    std::cout << "Process #" << taskId << " matrix contents:\n";
    std::cout << "[\n";
    for (std::vector<int> row : matrix)
    {
        for (int i : row)
        {
            std::cout << i << ", ";
        }
    }
    std::cout << "]\n";
}

int main(int argc, char* argv[])
{
    int taskId;
    int numTasks; // The number of tasks must be a perfect square

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numTasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskId);

    if (numTasks / sqrt(numTasks) != sqrt(numTasks))
    {
        std::cout << "The number of processors must be a perfect square (1,4,9,25,36...)\n";
        return 1;
    }

    int sqrtP = (int)sqrt(numTasks);
    int subMatrixSize = theGraph.size() / sqrtP;

    std::vector<std::vector<int>> subMatrix(subMatrixSize, std::vector<int>(subMatrixSize));

    // Calculate the starting row and column indices from the original graph
    int startingRowId = taskId / sqrtP;
    int startingColumnId = taskId % sqrtP;

    int graphStartingRowIndex = startingRowId * (theGraph.size() / sqrtP);
    int graphStartingColumnIndex = startingColumnId * (theGraph.size() / sqrtP);

    int graphEndingRowIndex = (startingRowId + 1) * (theGraph.size() / sqrtP) - 1;
    int graphEndingColumnIndex = (startingColumnId + 1) * (theGraph.size() / sqrtP) - 1;

    std::cout << "Process #" << taskId << " start coordinates: (" << graphStartingRowIndex << ", " << graphStartingColumnIndex << ") end coordinates: (" << graphEndingRowIndex
        << ", " << graphEndingColumnIndex << ")\n";

    assert((graphEndingRowIndex - graphStartingRowIndex + 1) % subMatrixSize == 0);
    assert((graphEndingColumnIndex - graphStartingColumnIndex + 1) % subMatrixSize == 0);

    if (taskId == MASTER)
    {
        std::cout << "Submatrix size: " << subMatrixSize << "x" << subMatrixSize << std::endl;

        // Send 2-d submatrices to all other processes
        for (int otherProcessId = 1; otherProcessId < numTasks; ++otherProcessId)
        {
            // Calculate the starting row and column indices
            int startingRowId = otherProcessId / sqrtP;
            int startingColumnId = otherProcessId % sqrtP;

            int startingRowIndex = startingRowId * (theGraph.size() / sqrtP);
            int startingColumnIndex = startingColumnId * (theGraph.size() / sqrtP);

            // Send the submatrix by chunks to the other processor
            for (int i = 0; i < subMatrixSize; ++i)
            {
                MPI_Send((void*)&theGraph[startingRowIndex + i][startingColumnIndex], subMatrixSize, MPI_INT, otherProcessId, 0, MPI_COMM_WORLD);
            }
        }

        // Extract the submatrix for process 0
        for (int row = 0; row < subMatrixSize; ++row)
        {
            // copy the entire column
            for (int col = 0; col < subMatrixSize; ++col)
            {
                subMatrix[row][col] = theGraph[row][col];
            }
        }
    }
    else
    {
        // Every processor that is not the master process will wait to receive its chunk of the matrix
        MPI_Status receiveStatus;

        for (int i = 0; i < subMatrixSize; ++i)
        {
            MPI_Recv(&subMatrix[i][0], subMatrixSize, MPI_INT, MASTER, 0, MPI_COMM_WORLD, &receiveStatus);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD); // Wait for all processes to reach this point

    std::pair<int, int> upperLeftCoordinates(graphStartingRowIndex, graphStartingColumnIndex);
    std::pair<int, int> bottomRighCoordinates(graphEndingRowIndex, graphEndingColumnIndex);
    parallelFloydWarshall(subMatrix, taskId, upperLeftCoordinates, bottomRighCoordinates, startingRowId, startingColumnId, sqrtP);

    MPI_Finalize();

    return 0;
}

/*
    Parameters:

    graph: submatrix that the current process operates on
    taskId: id of the current process
    upperLeftCoordinates: upper left coordinates of the submatrix in the original matrix
    bottomRightCoordinates: bottom right coordinates of the submatrix in the original matrix
    rowId: Row in the 2-d grid where the processor is located
    columnId: Column in the 2-d grid where the processor is located
    sqrtP: number of processors along a row or column
*/
void parallelFloydWarshall(
    std::vector<std::vector<int>> graph,
    int taskId,
    std::pair<int, int> upperLeftCoordinates,
    std::pair<int, int> bottomRightCoordinates,
    int rowId,
    int columnId,
    int sqrtP
)
{
    int n = graph.size();

    std::vector<std::vector<int>> previousDkMatrix = graph; // Matrix that will store the costs of the matrix Dk-1

    for (int k = 0; k < theGraph.size(); ++k)
    {
        // k is the current vertex for which we are trying to find the shortest path to all other vertices

        // During the Kth iteration of the algorithm, each of the sqrt(p) processes containing part of the kth row 
        // sends it to the other sqrt(p)-1 processes in that same row. The same thing applies for columns.

        // Kth row and column that will be received from other processes
        std::vector<int> kthRow(n);
        std::vector<int> kthColumn(n);

        int rowTag = 1;
        int colTag = 2;

        int rankOfFirstProcessorInCol = taskId % sqrtP;
        int rankOfLastProcessorInCol = rankOfFirstProcessorInCol + sqrtP * (sqrtP - 1);

        if (k >= upperLeftCoordinates.first && k <= bottomRightCoordinates.first)
        {
            // The current process will broadcast its portion of the kth row to all processes with the same COLUMN id
            int rowIdx = k % n;

            // Fill the vector with the contents of the row to send
            kthRow = previousDkMatrix[rowIdx];

            // Send the row to processes just before and just after this process on the same row in the 2-d processor mapping
            int previousProcessor = taskId - sqrtP;
            int nextProcessor = taskId + sqrtP;

            if ( previousProcessor >= rankOfFirstProcessorInCol )
            {
                MPI_Send((void*)&kthRow[0], n, MPI_INT, previousProcessor, rowTag, MPI_COMM_WORLD);
            }
            
            if (nextProcessor <= rankOfLastProcessorInCol)
            {
                MPI_Send((void*)&kthRow[0], n, MPI_INT, nextProcessor, rowTag, MPI_COMM_WORLD);
            }
        }
        else
        {
            // Otherwise, the current process will wait to receive the needed row from another process
            MPI_Status receiveStatus;
            MPI_Recv((void*)&kthRow[0], n, MPI_INT, MPI_ANY_SOURCE, rowTag, MPI_COMM_WORLD, &receiveStatus);

            // Once the current process receives the row, it needs to forward it to the next processor in the column opposite from where the message came from
            int previousProcessorRank = taskId - sqrtP;
            int nextProcessorRank = taskId + sqrtP;
            int sourceProcessorRank = receiveStatus.MPI_SOURCE;

            int destinationRank = -1;

            if (previousProcessorRank == sourceProcessorRank) destinationRank = nextProcessorRank;
            else if (nextProcessorRank == sourceProcessorRank) destinationRank = previousProcessorRank;

            // assert(destinationRank != -1);

            if (destinationRank >= rankOfFirstProcessorInCol && destinationRank <= rankOfLastProcessorInCol)
            {
                MPI_Send((void*)&kthRow[0], n, MPI_INT, destinationRank, rowTag, MPI_COMM_WORLD);
            }
        }

        int rankOfFirstProcessorInRow = rowId * sqrtP;
        int rankOfLastProcessorInRow = rankOfFirstProcessorInRow + sqrtP - 1;

        if (k >= upperLeftCoordinates.second && k <= bottomRightCoordinates.second)
        {
            // The current process will broadcast its portion of the kth column to all processes with the same ROW id
            int colIdx = k % n;

            // Fill the vector with the contents of the column to send
            for (int i = 0; i < n; ++i)
            {
                kthColumn[i] = previousDkMatrix[i][colIdx];
            }

            // Send the column to the previous and next processors in the current row of processors
            int previousProcessorRank = taskId - 1;
            int nextProcessorRank = taskId + 1;

            if (previousProcessorRank >= rankOfFirstProcessorInRow)
            {
                MPI_Send((void*) &kthColumn[0], n, MPI_INT, previousProcessorRank, colTag, MPI_COMM_WORLD);
            }

            if (nextProcessorRank <= rankOfLastProcessorInRow)
            {
                MPI_Send((void*)&kthColumn[0], n, MPI_INT, nextProcessorRank, colTag, MPI_COMM_WORLD);
            }
        }
        else
        {
            // Otherwise, the current process will wait to receive the needed column from another process
            MPI_Status receiveStatus;
            MPI_Recv((void*)&kthColumn[0], n, MPI_INT, MPI_ANY_SOURCE, colTag, MPI_COMM_WORLD, &receiveStatus);

            // The current process needs to forward the received column in the direction oposite to where it came from
            int previousProcessorRank = taskId - 1;
            int nextProcessorRank = taskId + 1;
            int sourceProcessorRank = receiveStatus.MPI_SOURCE;

            int destinationRank = -1;

            if (previousProcessorRank == sourceProcessorRank) destinationRank = nextProcessorRank;
            else if (nextProcessorRank == sourceProcessorRank) destinationRank = previousProcessorRank;

            // assert(destinationRank != -1);

            if (destinationRank >= rankOfFirstProcessorInRow && destinationRank <= rankOfLastProcessorInRow)
            {
                MPI_Send((void*) &kthColumn[0], n, MPI_INT, destinationRank, colTag, MPI_COMM_WORLD);
            }
        }

        // At this point, every processor has all the necessary information for computing its D^k submatrix
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                int dIToK = kthRow[j];
                int dKToJ = kthColumn[i];
                graph[i][j] = std::min(previousDkMatrix[i][j], dIToK + dKToJ);
            }
        }
        previousDkMatrix = graph;
    }

    // printVectorContentsWithAssertions(graph, answer, taskId, upperLeftCoordinates.first, upperLeftCoordinates.second);

    // All processors will send their 2-d matrix to the master process
    if (taskId == MASTER)
    {
        int numTasks;
        MPI_Comm_size(MPI_COMM_WORLD, &numTasks);

        std::map<int, std::vector<std::vector<int>>> result;

        result[0] = graph; // put the submatrix of the master process

        // Gather the submatrices of all other processes
        for (int i = 0; i < n * (numTasks - 1); ++i)
        {
            MPI_Status status;
            std::vector<int> buffer(n);

            MPI_Recv((void*)&buffer[0], n, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);

            result[status.MPI_SOURCE].push_back(buffer);
        }

        // At this point we have all the submatrices. We just need to reorder them into one big matrix
        std::vector<std::vector<int>> finalMatrix(theGraph.size(), std::vector<int>(theGraph.size()));

        for (int processorRank = 0; processorRank < numTasks; ++processorRank)
        {
            // Fill the submatrix for the current processor
            int startingRowId = processorRank / sqrtP;
            int startingColumnId = processorRank % sqrtP;

            int graphStartingRowIndex = startingRowId * (theGraph.size() / sqrtP);
            int graphStartingColumnIndex = startingColumnId * (theGraph.size() / sqrtP);

            int tempColumnPosition = graphStartingColumnIndex;

            for (int m = 0; m < n; ++m)
            {
                for (int l = 0; l < n; ++l)
                {
                    finalMatrix[graphStartingRowIndex][tempColumnPosition++] = result[processorRank][m][l];
                }
                tempColumnPosition = graphStartingColumnIndex;
                ++graphStartingRowIndex;
            }
        }

        printVectorContentsWithAssertions(finalMatrix, answer, taskId, 0, 0);
    }
    else
    {
        for (std::vector<int> row : graph)
        {
            MPI_Send((void*)&row[0], row.size(), MPI_INT, MASTER, 0, MPI_COMM_WORLD);
        }
    }
}

