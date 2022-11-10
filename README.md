# Parallel Floyd Warshall Algorithm (All-pairs shortest path)

## Compiling & running the program that implements the broadcasting strategy (must be done on an MPI cluster)

    mpicxx FloydWarshallWithBroadcasts.cpp -o broadcast -std=c++14
    mpirun -np 16 broadcast
    
## Compiling & running the program that implements the pipelining strategy (must be done on an MPI cluster)  

    mpicxx FloydWarshallPipelining.cpp -o pipelining -std=c++14
    mpirun -np 16 pipelining
