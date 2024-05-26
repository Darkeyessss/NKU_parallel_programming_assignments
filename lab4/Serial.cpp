#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>

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

void SerialGaussianElimination()
{
    // Forward Elimination
    for (int k = 0; k < N; k++)
    {
        for (int j = k + 1; j < N; j++)
        {
            a[k][j] /= a[k][k];
        }
        b[k] /= a[k][k];
        a[k][k] = 1.0;

        for (int i = k + 1; i < N; i++)
        {
            for (int j = k + 1; j < N; j++)
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

    init(size);

    auto start = chrono::high_resolution_clock::now();
    SerialGaussianElimination();
    auto end = chrono::high_resolution_clock::now();

    cleanup();

    chrono::duration<double> interval = end - start;
    cout << "Execution Time: " << interval.count() << " seconds" << endl;

    return 0;
}
