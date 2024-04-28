#include <iostream>
#include <cstdlib>
#include <immintrin.h>
#include <windows.h>

using namespace std;

alignas(32) float a[1000][1000];
alignas(32) float b[1000];
int N = 1000; // 矩阵大小

void init()
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            a[i][j] = float(rand()) / RAND_MAX * 100;
        }
        b[i] = float(rand()) / RAND_MAX * 100;
    }
}

void LU_avx()
{
    for (int k = 0; k < N; k++)
    {
        __m256 vk = _mm256_set1_ps(a[k][k]);
        for (int j = k + 1; j < N; j += 8)
        {
            __m256 va = _mm256_load_ps(&a[k][j]);
            va = _mm256_div_ps(va, vk);
            _mm256_store_ps(&a[k][j], va);
        }
        a[k][k] = 1.0;

        for (int i = k + 1; i < N; i++)
        {
            __m256 vi = _mm256_set1_ps(a[i][k]);
            for (int j = k + 1; j < N; j += 8)
            {
                __m256 vak = _mm256_load_ps(&a[k][j]);
                __m256 vai = _mm256_load_ps(&a[i][j]);
                __m256 vx = _mm256_mul_ps(vi, vak);
                vai = _mm256_sub_ps(vai, vx);
                _mm256_store_ps(&a[i][j], vai);
            }
            a[i][k] = 0.0;
        }
    }
}

int main()
{
    init();
    LARGE_INTEGER frequency, start, end;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start);
    LU_avx();
    QueryPerformanceCounter(&end);

    double interval = static_cast<double>(end.QuadPart - start.QuadPart) / frequency.QuadPart;
    cout << "Execution Time: " << interval << " seconds" << endl;

    return 0;
}
