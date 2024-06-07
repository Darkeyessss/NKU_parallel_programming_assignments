#include <iostream>
#include <vector>
#include <cstdlib>
#include <windows.h>

using namespace std;

vector<vector<float>> a; 
vector<float> b;         
vector<vector<float>> L; 
vector<vector<float>> U; 
int N = 1000;            

void init(int n)
{
    N = n;
    a.resize(N, vector<float>(N));
    L.resize(N, vector<float>(N, 0.0)); 
    U.resize(N, vector<float>(N, 0.0)); 
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            a[i][j] = float(rand()) / RAND_MAX * 100;
            if (i == j)
                L[i][j] = 1; 
        }
    }
}

void cleanup()
{
    a.clear();
    L.clear();
    U.clear();
}

void LUDecomposition()
{
    for (int k = 0; k < N; k++)
    {
        for (int j = k; j < N; j++)
        {
            float sum = 0.0;
            for (int p = 0; p < k; p++)
            {
                sum += L[k][p] * U[p][j];
            }
            U[k][j] = a[k][j] - sum; 
        }

        for (int i = k + 1; i < N; i++)
        {
            float sum = 0.0;
            for (int p = 0; p < k; p++)
            {
                sum += L[i][p] * U[p][k];
            }
            L[i][k] = (a[i][k] - sum) / U[k][k]; 
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
    LUDecomposition();
    QueryPerformanceCounter(&end);
    cleanup();

    double interval = static_cast<double>(end.QuadPart - start.QuadPart) / frequency.QuadPart;
    cout << "Execution Time: " << interval << " seconds" << endl;

    return 0;
}
