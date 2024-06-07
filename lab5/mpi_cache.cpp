#include <iostream>
#include <vector>
#include <cstdlib>
#include <mpi.h>
#include <omp.h>
#include <cmath>

using namespace std;

const int N = 1000; // 矩阵大小
float mat[N][N];    // 矩阵数据

void initialize_matrix(int rank)
{
    srand(time(NULL) * (rank + 1)); // Seed based on rank
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            mat[i][j] = rand() % 100; // Fill with random values
        }
    }
}

void pipeline_gaussian_elimination(int rank, int size)
{
    MPI_Status status;
    for (int k = 0; k < N; ++k)
    {
        int owner = k % size;
        if (rank == owner)
        {
#pragma omp simd
            for (int j = k + 1; j < N; ++j)
            {
                mat[k][j] /= mat[k][k];
            }
            mat[k][k] = 1.0;
        }
        MPI_Bcast(mat[k], N, MPI_FLOAT, owner, MPI_COMM_WORLD);

#pragma omp parallel for
        for (int i = k + 1; i < N; ++i)
        {
            if (i % size == rank)
            {
                float factor = mat[i][k];
#pragma omp simd
                for (int j = k + 1; j < N; ++j)
                {
                    mat[i][j] -= factor * mat[k][j];
                }
                mat[i][k] = 0.0;
            }
        }
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    initialize_matrix(rank);
    pipeline_gaussian_elimination(rank, size);

    MPI_Finalize();
    return 0;
}
