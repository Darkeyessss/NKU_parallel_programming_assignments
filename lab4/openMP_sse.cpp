#include <iostream>
#include <vector>
#include <cstdlib>
#include <omp.h>
#include <windows.h>
#include <xmmintrin.h> // SSE

using namespace std;

vector<vector<float>> a;
vector<float> b;
vector<float> x; // 解向量
int N = 1000;    // 默认矩阵大小

void init(int n)
{
    N = n;
    a.resize(N, vector<float>(N));
    b.resize(N);
    x.resize(N);
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
    x.clear();
}

void ParallelGaussianElimination()
{
    // 设置线程数
    omp_set_num_threads(8);

// Forward Elimination
#pragma omp parallel private(i, j, k)
    for (int k = 0; k < N; k++)
    {
#pragma omp single
        {
            __m128 akk = _mm_set1_ps(a[k][k]);
            int j;
            for (j = k + 1; j <= N - 4; j += 4)
            {
                __m128 a_kj = _mm_loadu_ps(&a[k][j]);
                a_kj = _mm_div_ps(a_kj, akk);
                _mm_storeu_ps(&a[k][j], a_kj);
            }
            for (; j < N; j++)
            {
                a[k][j] /= a[k][k];
            }
            b[k] /= a[k][k];
            a[k][k] = 1.0;
        }

#pragma omp for schedule(static)
        for (int i = k + 1; i < N; i++)
        {
            __m128 aik = _mm_set1_ps(a[i][k]);
            int j;
            for (j = k + 1; j <= N - 4; j += 4)
            {
                __m128 a_kj = _mm_loadu_ps(&a[k][j]);
                __m128 a_ij = _mm_loadu_ps(&a[i][j]);
                a_ij = _mm_sub_ps(a_ij, _mm_mul_ps(aik, a_kj));
                _mm_storeu_ps(&a[i][j], a_ij);
            }
            for (; j < N; j++)
            {
                a[i][j] -= a[i][k] * a[k][j];
            }
            b[i] -= a[i][k] * b[k];
            a[i][k] = 0;
        }
    }

    // Back Substitution
    for (int i = N - 1; i >= 0; i--)
    {
        x[i] = b[i];
        for (int j = i + 1; j < N; j++)
        {
            x[i] -= a[i][j] * x[j];
        }
    }
}

int main()
{
    int size = 1000;

    LARGE_INTEGER frequency;
    LARGE_INTEGER start;
    LARGE_INTEGER end;
    QueryPerformanceFrequency(&frequency);

    init(size);
    QueryPerformanceCounter(&start);
    ParallelGaussianElimination();
    QueryPerformanceCounter(&end);
    cleanup();

    double interval = static_cast<double>(end.QuadPart - start.QuadPart) / frequency.QuadPart;
    cout << "Execution Time: " << interval << " seconds" << endl;

    return 0;
}
