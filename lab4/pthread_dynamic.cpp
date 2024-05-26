#include <iostream>
#include <vector>
#include <cstdlib>
#include <windows.h>

using namespace std;

vector<vector<float>> a;
vector<float> b;
vector<float> x;     // 解向量
int N = 3800;        // 默认矩阵大小
int num_threads = 8; // 线程数

struct ThreadParam
{
    int t_id;
    int k;
};

void init(int n)
{
    N = n;
    a.resize(N, vector<float>(N));
    b.resize(N);
    x.resize(N);

    // 生成上三角矩阵
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < i; j++)
        {
            a[i][j] = 0;
        }
        for (int j = i; j < N; j++)
        {
            a[i][j] = float(rand()) / RAND_MAX * 100;
        }
        b[i] = float(rand()) / RAND_MAX * 100;
    }

    // 进行若干次随机行交换和线性组合
    for (int k = 0; k < N; k++)
    {
        int i1 = rand() % N;
        int i2 = rand() % N;
        if (i1 != i2)
        {
            swap(a[i1], a[i2]);
            swap(b[i1], b[i2]);
        }
    }
}

void cleanup()
{
    a.clear();
    b.clear();
    x.clear();
}

DWORD WINAPI threadFunc(LPVOID param)
{
    ThreadParam *p = (ThreadParam *)param;
    int t_id = p->t_id;
    int k = p->k;

    // Elimination step
    for (int i = k + 1 + t_id; i < N; i += num_threads)
    {
        for (int j = k + 1; j < N; j++)
        {
            a[i][j] -= a[i][k] * a[k][j];
        }
        b[i] -= a[i][k] * b[k];
        a[i][k] = 0;
    }

    return 0;
}

void ParallelGaussianElimination()
{
    vector<HANDLE> handles(num_threads);
    vector<ThreadParam> params(num_threads);

    for (int k = 0; k < N; k++)
    {
        // Division step
        for (int j = k + 1; j < N; j++)
        {
            a[k][j] /= a[k][k];
        }
        b[k] /= a[k][k];
        a[k][k] = 1.0;

        for (int t_id = 0; t_id < num_threads; t_id++)
        {
            params[t_id].t_id = t_id;
            params[t_id].k = k;
            handles[t_id] = CreateThread(NULL, 0, threadFunc, (LPVOID)&params[t_id], 0, NULL);
        }

        WaitForMultipleObjects(num_threads, handles.data(), TRUE, INFINITE);

        for (HANDLE handle : handles)
        {
            CloseHandle(handle);
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
