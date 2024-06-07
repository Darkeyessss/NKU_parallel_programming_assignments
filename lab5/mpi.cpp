#include <iostream>
#include <vector>
#include <cstdlib>
#include <mpi.h>

using namespace std;

vector<vector<float>> a;
vector<vector<float>> L;
vector<vector<float>> U;
int N;

void init(int rank, int size, int n)
{
    N = n;
    a.resize(N, vector<float>(N, 0.0));
    L.resize(N, vector<float>(N, 0.0));
    U.resize(N, vector<float>(N, 0.0));

    if (rank == 0)
    {
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                a[i][j] = float(rand()) / RAND_MAX * 100;
                if (i == j)
                    L[i][j] = 1;
            }
        }
        for (int p = 1; p < size; p++)
        {
            int start_row = p * N / size;
            int end_row = (p + 1) * N / size;
            for (int i = start_row; i < end_row; i++)
            {
                MPI_Send(a[i].data(), N, MPI_FLOAT, p, 0, MPI_COMM_WORLD);
            }
        }
    }
    else
    {
        int start_row = rank * N / size;
        int end_row = (rank + 1) * N / size;
        for (int i = start_row; i < end_row; i++)
        {
            MPI_Recv(a[i].data(), N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
}

void LUDecomposition(int rank, int size)
{
    for (int k = 0; k < N; k++)
    {
        int owner = k * size / N;
        if (rank == owner)
        {
            for (int j = k; j < N; j++)
            {
                float sum = 0.0;
                for (int p = 0; p < k; p++)
                {
                    sum += L[k][p] * U[p][j];
                }
                U[k][j] = a[k][j] - sum;
            }
            MPI_Bcast(U[k].data(), N, MPI_FLOAT, owner, MPI_COMM_WORLD);
            for (int i = k + 1; i < N; i++)
            {
                float sum = 0.0;
                for (int p = 0; p < k; p++)
                {
                    sum += L[i][p] * U[p][k];
                }
                L[i][k] = (a[i][k] - sum) / U[k][k];
            }
        }
        else
        {
            MPI_Bcast(U[k].data(), N, MPI_FLOAT, owner, MPI_COMM_WORLD);
        }
        int start_row = rank * N / size;
        int end_row = (rank + 1) * N / size;
        for (int i = max(k + 1, start_row); i < end_row; i++)
        {
            for (int j = k + 1; j < N; j++)
            {
                float sum = 0.0;
                for (int p = 0; p <= k; p++)
                {
                    sum += L[i][p] * U[p][j];
                }
                a[i][j] = a[i][j] - sum;
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

    init(rank, size, n);
    LUDecomposition(rank, size);

    MPI_Finalize();
    return 0;
}
