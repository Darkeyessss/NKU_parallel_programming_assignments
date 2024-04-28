#include <iostream>
#include <vector>
#include <cstdlib>
#include <arm_neon.h>
#include <windows.h>

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

void NEONGaussianElimination()
{
    // Forward Elimination using NEON
    for (int k = 0; k < N; k++)
    {
        float32x4_t vk = vdupq_n_f32(a[k][k]);
        for (int j = k + 1; j + 3 < N; j += 4)
        {
            float32x4_t va = vld1q_f32(&a[k][j]);
            va = vdivq_f32(va, vk);
            vst1q_f32(&a[k][j], va);
        }
        b[k] /= a[k][k];
        a[k][k] = 1.0;

        for (int i = k + 1; i < N; i++)
        {
            float32x4_t vi = vdupq_n_f32(a[i][k]);
            for (int j = k + 1; j + 3 < N; j += 4)
            {
                float32x4_t vak = vld1q_f32(&a[k][j]);
                float32x4_t vai = vld1q_f32(&a[i][j]);
                float32x4_t vx = vmulq_f32(vi, vak);
                vai = vsubq_f32(vai, vx);
                vst1q_f32(&a[i][j], vai);
            }
            b[i] -= a[i][k] * b[k];
            a[i][k] = 0.0;
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
    cout << "Enter the size of matrix N: ";
    int size;
    cin >> size;

    LARGE_INTEGER frequency;
    LARGE_INTEGER start;
    LARGE_INTEGER end;
    QueryPerformanceFrequency(&frequency);

    init(size);
    QueryPerformanceCounter(&start);
    NEONGaussianElimination();
    QueryPerformanceCounter(&end);
    cleanup();

    double interval = static_cast<double>(end.QuadPart - start.QuadPart) / frequency.QuadPart;
    cout << "Execution Time: " << interval << " seconds" << endl;

    return 0;
}
