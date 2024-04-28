#include <iostream>
#include <vector>
#include <cmath>
#include <windows.h>

using namespace std;

void GaussianEliminationWithPivoting(vector<vector<float>> &A, vector<float> &b)
{
    int N = A.size(); // 获取矩阵大小

    for (int k = 0; k < N; k++)
    {
        // Step 1: Pivoting
        int maxRow = k;
        for (int i = k + 1; i < N; i++)
        {
            if (fabs(A[i][k]) > fabs(A[maxRow][k]))
            {
                maxRow = i;
            }
        }
        // Swap rows if needed
        if (maxRow != k)
        {
            swap(A[k], A[maxRow]);
            swap(b[k], b[maxRow]);
        }

        // Step 2: Forward Elimination
        for (int i = k + 1; i < N; i++)
        {
            float factor = A[i][k] / A[k][k];
            for (int j = k; j < N; j++)
            {
                A[i][j] -= A[k][j] * factor;
            }
            b[i] -= b[k] * factor;
        }
    }

    // Step 3: Back Substitution
    vector<float> x(N, 0);
    for (int i = N - 1; i >= 0; i--)
    {
        x[i] = b[i];
        for (int j = i + 1; j < N; j++)
        {
            x[i] -= A[i][j] * x[j];
        }
        x[i] /= A[i][i];
    }
}

void testGaussianElimination(int N)
{
    vector<vector<float>> A(N, vector<float>(N));
    vector<float> b(N);

    // Generate random matrix A and vector b
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            A[i][j] = float(rand()) / RAND_MAX * 100.0;
        }
        b[i] = float(rand()) / RAND_MAX * 100.0;
    }

    LARGE_INTEGER frequency, start, end;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start);

    GaussianEliminationWithPivoting(A, b);

    QueryPerformanceCounter(&end);
    double time_spent = double(end.QuadPart - start.QuadPart) / frequency.QuadPart;

    cout << "Test for N = " << N << " took " << time_spent * 1000 << " ms" << endl;
}

int main()
{
    vector<int> sizes = {10, 100, 200, 500, 1000}; // Sizes to test
    for (int size : sizes)
    {
        testGaussianElimination(size);
    }
    return 0;
}
