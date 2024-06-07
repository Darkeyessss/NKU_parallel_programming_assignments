#include <iostream>
#include <vector>
#include <cstdlib>
#include <mpi.h>
#include <omp.h>

using namespace std;

vector<vector<float>> mat;
int dim;

void setup(int rank, int size, int n)
{
    dim = n;
    mat.resize(dim, vector<float>(dim, 0.0));

    if (rank == 0)
    {
        for (int i = 0; i < dim; i++)
        {
            for (int j = 0; j < dim; j++)
            {
                mat[i][j] = float(rand()) / RAND_MAX * 100;
            }
        }
    }
}

void parallelGaussianElimination(int rank, int size)
{
    for (int k = 0; k < dim; k++)
    {
        int owner = k % size;
        if (rank == owner)
        {
#pragma omp parallel for simd
            for (int j = k + 1; j < dim; j++)
            {
                mat[k][j] /= mat[k][k];
            }
            mat[k][k] = 1.0;

            for (int i = k + 1; i < dim; i++)
            {
#pragma omp parallel for simd
                for (int j = k + 1; j < dim; j++)
                {
                    mat[i][j] -= mat[i][k] * mat[k][j];
                }
                mat[i][k] = 0.0;
            }
        }

        for (int i = 0; i < size; i++)
        {
            if (i != owner)
            {
                MPI_Send(mat[k].data(), dim, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
            }
        }

        if (rank != owner)
        {
            MPI_Recv(mat[k].data(), dim, MPI_FLOAT, owner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
}

int main(int argc, char **argv)
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 1000; // Default matrix size
    if (argc > 1)
        n = atoi(argv[1]);

    setup(rank, size, n);
    parallelGaussianElimination(rank, size);

    MPI_Finalize();
    return 0;
}
