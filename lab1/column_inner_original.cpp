#include <iostream>
#include <windows.h>
using namespace std;

void column_inner_original(long long n, double **matrix, double *vector, double *result)
{
    for (int i = 0; i < n; i++)
    {
        result[i] = 0.0;
        for (int j = 0; j < n; j++)
            result[i] += matrix[j][i] * vector[j];
    }
}

int main()
{
    long long n = 8192;
    LARGE_INTEGER start, end, frequency;
    double cpu_time_used;

    double **matrix = new double *[n];
    for (int i = 0; i < n; i++)
    {
        matrix[i] = new double[n];
    }
    double *vector = new double[n];
    double *result = new double[n];

    for (int i = 0; i < n; i++)
    {
        vector[i] = i + 1;
        for (int j = 0; j < n; j++)
        {
            matrix[i][j] = i + j + 1;
        }
    }

    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start);
    column_inner_original(n, matrix, vector, result);
    QueryPerformanceCounter(&end);
    cpu_time_used = static_cast<double>(end.QuadPart - start.QuadPart) / frequency.QuadPart;
    cout << "column_inner_original Time: " << cpu_time_used * 1000 << " ms\n";

    for (int i = 0; i < n; i++)
    {
        delete[] matrix[i];
    }
    delete[] matrix;
    delete[] vector;
    delete[] result;

    return 0;
}
