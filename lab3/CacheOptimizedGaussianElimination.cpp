#include <iostream>
#include <vector>
#include <cstdlib>
#include <windows.h>

using namespace std;

vector<vector<float>> a;
vector<float> b;
int N = 1000;        // 矩阵大小，默认值
int BLOCK_SIZE = 32; // 块大小，根据实际的缓存大小调整

void init(int n)
{
    N = n;
    a.resize(N, vector<float>(N));
    b.resize(N);
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            a[i][j] = float(rand()) / RAND_MAX * 100;
        }
        b[i] = float(rand()) / RAND_MAX * 100;
    }
}

void cleanup()
{
    a.clear();
    b.clear();
}

void CacheOptimizedGaussianElimination()
{
    for (int kk = 0; kk < N; kk += BLOCK_SIZE)
    {
        int K = min(kk + BLOCK_SIZE, N);
        for (int k = kk; k < K; k++)
        {
            for (int j = k + 1; j < N; j++)
                a[k][j] /= a[k][k];
            b[k] /= a[k][k];
            a[k][k] = 1.0;

            for (int i = k + 1; i < N; i++)
            {
                for (int j = k + 1; j < N; j++)
                    a[i][j] -= a[i][k] * a[k][j];
                b[i] -= a[i][k] * b[k];
                a[i][k] = 0;
            }
        }

        for (int i = K; i < N; i++)
        {
            for (int j = K; j < N; j++)
            {
                for (int k = kk; k < K; k++)
                {
                    a[i][j] -= a[i][k] * a[k][j];
                }
            }
        }
    }
}

int main()
{

    int size = 2000;

    LARGE_INTEGER frequency;
    LARGE_INTEGER start;
    LARGE_INTEGER end;
    QueryPerformanceFrequency(&frequency);

    init(size);
    QueryPerformanceCounter(&start);
    CacheOptimizedGaussianElimination();
    QueryPerformanceCounter(&end);
    cleanup();

    double interval = static_cast<double>(end.QuadPart - start.QuadPart) / frequency.QuadPart;
    cout << "Execution Time: " << interval << " seconds" << endl;

    return 0;
}
