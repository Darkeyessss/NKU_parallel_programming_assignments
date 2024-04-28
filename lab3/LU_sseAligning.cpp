#include <iostream>
#include <cstdlib>
#include <xmmintrin.h>
#include <windows.h>

using namespace std;

alignas(16) float a[1000][1000];
alignas(16) float b[1000];
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

void LU_sse()
{
    for (int k = 0; k < N; k++)
    {
        __m128 vk = _mm_set1_ps(a[k][k]);
        for (int j = k + 1; j < N; j += 4)
        {
            __m128 va = _mm_load_ps(&a[k][j]);
            va = _mm_div_ps(va, vk);
            _mm_store_ps(&a[k][j], va);
        }


            for (int i = k + 1; i < N; i++)
        {
            __m128 vi = _mm_set1_ps(a[i][k]);
            for (int j = k + 1; j < N; j += 4)
            {
                __m128 vak = _mm_load_ps(&a[k][j]);
                __m128 vai = _mm_load_ps(&a[i][j]);
                __m128 vx = _mm_mul_ps(vi, vak);
                vai = _mm_sub_ps(vai, vx);
                _mm_store_ps(&a[i][j], vai);
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
    LU_sse();
    QueryPerformanceCounter(&end);

    double interval = static_cast<double>(end.QuadPart - start.QuadPart) / frequency.QuadPart;
    cout << "Execution Time: " << interval << " seconds" << endl;

    return 0;
}
