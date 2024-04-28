#include <iostream>
#include <vector>
#include <cstdlib>
#include <immintrin.h>
#include <windows.h>

using namespace std;

vector<vector<float>> a;
vector<float> b;
int N = 1000; // 默认矩阵大小

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

void LU_avx()
{
    for (int k = 0; k < N; k++)
    {
        __m256 vk = _mm256_set1_ps(a[k][k]);
        for (int j = k + 1; j + 7 < N; j += 8)
        {
            __m256 va = _mm256_loadu_ps(&a[k][j]);
            va = _mm256_div_ps(va, vk);
            _mm256_storeu_ps(&a[k][j], va);
        }
        a[k][k] = 1.0;

        for (int i = k + 1; i < N; i++)
        {
            __m256 vi = _mm256_set1_ps(a[i][k]);
            for (int j = k + 1; j + 7 < N; j += 8)
            {
                __m256 vak = _mm256_loadu_ps(&a[k][j]);
                __m256 vai = _mm256_loadu_ps(&a[i][j]);
                __m256 vx = _mm256_mul_ps(vi, vak);
                vai = _mm256_sub_ps(vai, vx);
                _mm256_storeu_ps(&a[i][j], vai);
            }
            a[i][k] = 0.0;
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
    LU_avx();
    QueryPerformanceCounter(&end);
    cleanup();

    double interval = static_cast<double>(end.QuadPart - start.QuadPart) / frequency.QuadPart;
    cout << "Execution Time: " << interval << " seconds" << endl;

    return 0;
}
