#include <iostream>
#include <vector>
#include <cstdlib>
#include <mpi.h>

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

void cyclicDecompose(int rank, int size)
{
    for (int k = 0; k < dim; k++)
    {
        int owner = k % size;
        if (rank == owner)
        {
            for (int j = k + 1; j < dim; j++)
            {
                mat[k][j] /= mat[k][k];
            }
            mat[k][k] = 1.0;
            for (int p = 0; p < size; p++)
            {
                if (p != rank)
                {
                    MPI_Send(mat[k].data(), dim, MPI_FLOAT, p, 0, MPI_COMM_WORLD);
                }
            }
        }
        else
        {
            MPI_Recv(mat[k].data(), dim, MPI_FLOAT, owner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        for (int i = k + 1; i < dim; i++)
        {
            if (i % size == rank)
            {
                for (int j = k + 1; j < dim; j++)
                {
                    mat[i][j] -= mat[i][k] * mat[k][j];
                }
                mat[i][k] = 0;
            }
        }
    }
}

int main(int argc, char **argv)
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 1000;
    if (argc > 1)
        n = atoi(argv[1]);

    setup(rank, size, n);
    cyclicDecompose(rank, size);

    MPI_Finalize();
    return 0;
}
