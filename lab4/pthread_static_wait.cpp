#include <iostream>
#include <vector>
#include <cstdlib>
#include <pthread.h>
#include <sys/time.h>
#include <atomic>

using namespace std;

vector<vector<float>> a;
vector<float> b;
vector<float> x;     // 解向量
int N = 1000;        // 默认矩阵大小
int num_threads = 8; // 线程数

atomic<int> counter;
atomic<bool> division_done;
atomic<bool> elimination_done;

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

void *threadFunc(void *param)
{
    ThreadParam *p = (ThreadParam *)param;
    int t_id = p->t_id;

    for (int k = 0; k < N; k++)
    {
        if (t_id == 0)
        {
            for (int j = k + 1; j < N; j++)
            {
                a[k][j] /= a[k][k];
            }
            b[k] /= a[k][k];
            a[k][k] = 1.0;
            division_done.store(true);
        }
        else
        {
            while (!division_done.load())
            {
            }
        }

        for (int i = k + 1 + t_id; i < N; i += num_threads)
        {
            for (int j = k + 1; j < N; j++)
            {
                a[i][j] -= a[i][k] * a[k][j];
            }
            b[i] -= a[i][k] * b[k];
            a[i][k] = 0;
        }

        counter.fetch_add(1);
        if (counter.load() == num_threads)
        {
            counter.store(0);
            division_done.store(false);
        }
        else
        {
            while (counter.load() != 0)
            {
            }
        }
    }
    pthread_exit(NULL);
}

void ParallelGaussianElimination()
{
    pthread_t handles[num_threads];
    ThreadParam params[num_threads];

    for (int t_id = 0; t_id < num_threads; t_id++)
    {
        params[t_id].t_id = t_id;
        pthread_create(&handles[t_id], NULL, threadFunc, (void *)&params[t_id]);
    }

    for (int t_id = 0; t_id < num_threads; t_id++)
    {
        pthread_join(handles[t_id], NULL);
    }
}

int main()
{
    int size = 1000;

    struct timeval start, end;

    init(size);
    gettimeofday(&start, NULL);
    ParallelGaussianElimination();
    gettimeofday(&end, NULL);
    cleanup();

    double interval = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
    cout << "Execution Time: " << interval << " seconds" << endl;

    return 0;
}
