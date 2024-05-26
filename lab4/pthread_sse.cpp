#include <iostream>
#include <vector>
#include <cstdlib>
#include <windows.h>
#include <emmintrin.h> // SSE2

using namespace std;

vector<vector<float>> a;
vector<float> b;
vector<float> x;     // 解向量
int N = 800;         // 默认矩阵大小
int num_threads = 8; // 线程数

HANDLE sem_leader;
vector<HANDLE> sem_division;
vector<HANDLE> sem_elimination;

struct ThreadParam
{
    int t_id;
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

    for (int k = 0; k < N; k++)
    {
        // Division step
        if (t_id == 0)
        {
            __m128 akk = _mm_set1_ps(a[k][k]);
            int j;
            for (j = k + 1; j <= N - 4; j += 4)
            {
                __m128 akj = _mm_loadu_ps(&a[k][j]);
                akj = _mm_div_ps(akj, akk);
                _mm_storeu_ps(&a[k][j], akj);
            }
            for (; j < N; j++)
            {
                a[k][j] /= a[k][k];
            }
            b[k] /= a[k][k];
            a[k][k] = 1.0;
            for (int i = 0; i < num_threads - 1; i++)
            {
                ReleaseSemaphore(sem_division[i], 1, NULL);
            }
        }
        else
        {
            WaitForSingleObject(sem_division[t_id - 1], INFINITE);
        }

        // Elimination step
        for (int i = k + 1 + t_id; i < N; i += num_threads)
        {
            __m128 aik = _mm_set1_ps(a[i][k]);
            int j;
            for (j = k + 1; j <= N - 4; j += 4)
            {
                __m128 akj = _mm_loadu_ps(&a[k][j]);
                __m128 aij = _mm_loadu_ps(&a[i][j]);
                __m128 mul = _mm_mul_ps(aik, akj);
                aij = _mm_sub_ps(aij, mul);
                _mm_storeu_ps(&a[i][j], aij);
            }
            for (; j < N; j++)
            {
                a[i][j] -= a[i][k] * a[k][j];
            }
            b[i] -= a[i][k] * b[k];
            a[i][k] = 0;
        }

        if (t_id == 0)
        {
            for (int i = 0; i < num_threads - 1; i++)
            {
                WaitForSingleObject(sem_leader, INFINITE);
            }
            for (int i = 0; i < num_threads - 1; i++)
            {
                ReleaseSemaphore(sem_elimination[i], 1, NULL);
            }
        }
        else
        {
            ReleaseSemaphore(sem_leader, 1, NULL);
            WaitForSingleObject(sem_elimination[t_id - 1], INFINITE);
        }
    }
    return 0;
}

void ParallelGaussianElimination()
{
    vector<HANDLE> handles(num_threads);
    vector<ThreadParam> params(num_threads);

    sem_leader = CreateSemaphore(NULL, 0, num_threads - 1, NULL);
    sem_division.resize(num_threads - 1);
    sem_elimination.resize(num_threads - 1);

    for (int i = 0; i < num_threads - 1; i++)
    {
        sem_division[i] = CreateSemaphore(NULL, 0, 1, NULL);
        sem_elimination[i] = CreateSemaphore(NULL, 0, 1, NULL);
    }

    for (int t_id = 0; t_id < num_threads; t_id++)
    {
        params[t_id].t_id = t_id;
        handles[t_id] = CreateThread(NULL, 0, threadFunc, (LPVOID)&params[t_id], 0, NULL);
    }

    WaitForMultipleObjects(num_threads, handles.data(), TRUE, INFINITE);

    for (int i = 0; i < num_threads - 1; i++)
    {
        CloseHandle(sem_division[i]);
        CloseHandle(sem_elimination[i]);
    }
    CloseHandle(sem_leader);
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
