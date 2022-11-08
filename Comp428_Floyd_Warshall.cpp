#include <iostream>
#include <vector>
#include <limits.h>
#include <math.h>
#include <cassert>

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

void printVectorContentsWithAssertions(std::vector<std::vector<int>> matrix, int taskId, int graphStartRow, int graphStartColumn)
{
    std::cout << "Process #" << taskId << " matrix contents:\n";
    std::cout << "[\n";
    int j = graphStartRow;
    int k = graphStartColumn;
    for (std::vector<int> row : matrix)
    {
        for (int i : row)
        {
            assert(i == theGraph[j][k++]);
            std::cout << i << ", ";
        }
        k = graphStartColumn;
        ++j;
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

    int sqrtP = (int) sqrt(numTasks);
    int subMatrixSize = theGraph.size() / sqrtP;

    std::vector<std::vector<int>> subMatrix(subMatrixSize, std::vector<int>(subMatrixSize));

    // Calculate the starting row and column indices from the original graph
    int startingRowId = taskId / sqrtP;
    int startingColumnId = taskId % sqrtP;

    int graphStartingRowIndex = startingRowId * (theGraph.size() / sqrtP);
    int graphStartingColumnIndex = startingColumnId * (theGraph.size() / sqrtP);

    int graphEndingRowIndex = (startingRowId + 1) * (theGraph.size() / sqrtP) - 1;
    int graphEndingColumnIndex = (startingColumnId + 1) * (theGraph.size() / sqrtP) - 1;

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

            std::cout << "Process #" << otherProcessId << " start row idx: " << startingRowIndex << " start column idx: " << startingColumnIndex << "\n";

            // Send the submatrix by chunks to the other processor
            for (int i = 0; i < subMatrixSize; ++i)
            {
                MPI_Send( (void*) &theGraph[startingRowIndex + i][startingColumnIndex], subMatrixSize, MPI_INT, otherProcessId, 0, MPI_COMM_WORLD);
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
        // printVectorContentsWithAssertions(subMatrix, taskId, 0, 0);
    }
    else
    {
        // Every processor that is not the master process will wait to receive its chunk of the matrix
        MPI_Status receiveStatus;

        for (int i = 0; i < subMatrixSize; ++i)
        {
            // std::cout << "Process #" << taskId << " receiving submatrix row #" << i + 1 << std::endl;
            MPI_Recv(&subMatrix[i][0], subMatrixSize, MPI_INT, MASTER, 0, MPI_COMM_WORLD, &receiveStatus);
        }

        // printVectorContentsWithAssertions(subMatrix, taskId, graphStartingRowIndex, graphStartingColumnIndex);
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
    std::pair<int,int> upperLeftCoordinates, 
    std::pair<int, int> bottomRightCoordinates,
    int rowId,
    int columnId,
    int sqrtP
)
{
    int n = graph.size();
    std::vector<std::vector<int>> previousDkMatrix = graph; // Matrix that will store the costs of the matrix Dk-1

    for (int k = 1; k < n; ++k)
    {
        // k is the current vertex for which we are trying to find the shortest path to all other vertices

        // During the Kth iteration of the algorithm, each of the sqrt(p) processes containing part of the kth row 
        // sends it to the other sqrt(p)-1 processes in that same row. The same thing applies for columns.

        // First we need to determine the row and column ids during this kth iteration
        int kthRowAndColId = k % sqrtP;

        // Create row and column communicators
        //MPI_Comm rowComm;
        //MPI_Comm colComm;
        // 
        //// Split along the kth row and column
        //MPI_Comm_split( MPI_COMM_WORLD, rowId, 0, &colComm );
        //MPI_Comm_split( MPI_COMM_WORLD, columnId, 0, &rowComm );

        // Kth row and column that will be received from other processes
        std::vector<int> kthRow(n);
        std::vector<int> kthColumn(n);

        if (kthRowAndColId == rowId)
        {
            // The current process will broadcast its portion of the kth row to all processes with the same COLUMN id
            int rowIdx = k % (n / sqrtP);

            for (int i = 0; i < n; ++i)
            {
                kthRow[i] = previousDkMatrix[rowIdx][i];
            }

            for ( int i = 0, otherProcessId = taskId % sqrtP; i < sqrtP; ++i, otherProcessId += sqrtP )
            {
                std::cout << "Process #" << taskId << " sending row to process " << otherProcessId << "\n";
                // Send the row to all other processes with the same rowId except yourself
                if ( otherProcessId != taskId ) MPI_Send((void*) &kthRow[0], n, MPI_INT, otherProcessId, 0, MPI_COMM_WORLD);
            }
        }
        else
        {
            // std::cout << "Process #" << taskId << " receiving row\n";

            // Otherwise, the current process will wait to receive the needed row from another process
            MPI_Status receiveStatus;
            MPI_Recv((void*) &kthRow[0], n, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &receiveStatus);
        }

        if (kthRowAndColId == columnId)
        {
            // The current process will broadcast its portion of the kth column to all processes with the same ROW id
            int colIdx = k % (n / sqrtP);

            for (int i = 0; i < n; ++i)
            {
                kthColumn[i] = previousDkMatrix[i][colIdx];
            }

            for (int i = 0, otherProcessId = rowId * sqrtP; i < sqrtP; ++i, ++otherProcessId)
            {
                std::cout << "Process #" << taskId << " sending column to process " << otherProcessId << "\n";
                if (otherProcessId != taskId) MPI_Send((void*) &kthColumn[0], n, MPI_INT, otherProcessId, 0, MPI_COMM_WORLD);
            }
        }
        else
        {
            // std::cout << "Process #" << taskId << " receiving column\n";

            // Otherwise, the current process will wait to receive the needed column from another process
            MPI_Status receiveStatus;
            MPI_Recv((void*) &kthColumn[0], n, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &receiveStatus);
        }

        // Free communicators after use
        /*MPI_Comm_free(&rowComm);
        MPI_Comm_free(&colComm);*/

        std::cout << "Process #" << taskId << " waiting at barrier!\n";
        MPI_Barrier(MPI_COMM_WORLD); // Wait for all processes to reach this point

        //for (int i = 0; i < n; ++i)
        //{
        //    for (int j = 0; j < n; ++j)
        //    {
        //        // We now need to obtain two values of the previous submatrix from other processes that have them

        //    }
        //}
    }
}

