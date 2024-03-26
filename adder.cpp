#include <iostream>
#include <windows.h>
using namespace std;

void column_inner_original(long long n, double **matrix, double *vector, double *result)
{
    // 逐列访问矩阵元素
    for (int i = 0; i < n; i++)
    {
        result[i] = 0.0;
        for (int j = 0; j < n; j++)
            result[i] += matrix[j][i] * vector[j];
    }
}

void column_inner_cache(long long n, double **matrix, double *vector, double *result)
{
    // 改为逐行访问矩阵元素
    for (int i = 0; i < n; i++)
        result[i] = 0.0;
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++)
            result[i] += matrix[j][i] * vector[j];
}

double sum_chain(long long n, double *a)
{
    // 链式
    double sum = 0;
    for (int i = 0; i < n; i++)
        sum += a[i];
    return sum;
}

void sum_doubleloop(long long n, double *a)
{
    for (int m = n; m > 1; m /= 2)
        for (int i = 0; i < m / 2; i++)
            a[i] = a[i * 2] + a[i * 2 + 1];
    // a[0]是结果
}

int main()
{
    long long n = 4096;
    long long m = 3000;
    LARGE_INTEGER start, end, frequency;
    double cpu_time_used;

    double **matrix = new double *[n];
    for (int i = 0; i < n; i++)
    {
        matrix[i] = new double[n];
    }
    double *vector = new double[n];
    double *result_inner_product = new double[n];
    double *numbers = new double[n];
    double result_sum;

    for (int i = 0; i < n; i++)
    {
        vector[i] = 1;
        numbers[i] = 1;
        for (int j = 0; j < n; j++)
        {
            matrix[i][j] = 1;
        }
    }

    QueryPerformanceFrequency(&frequency);

    QueryPerformanceCounter(&start);
    column_inner_original(n, matrix, vector, result_inner_product);
    QueryPerformanceCounter(&end);
    cpu_time_used = static_cast<double>(end.QuadPart - start.QuadPart) / frequency.QuadPart;
    cout << "column_inner_original Time: " << cpu_time_used * 1000 << " ms\n";

    QueryPerformanceCounter(&start);
    column_inner_cache(n, matrix, vector, result_inner_product);
    QueryPerformanceCounter(&end);
    cpu_time_used = static_cast<double>(end.QuadPart - start.QuadPart) / frequency.QuadPart;
    cout << "column_inner_cache Time: " << cpu_time_used * 1000 << " ms\n";

    QueryPerformanceCounter(&start);
    for (int i = 0; i < m; i++)
    {
        result_sum = sum_chain(n, numbers);
    }
    QueryPerformanceCounter(&end);
    cpu_time_used = static_cast<double>(end.QuadPart - start.QuadPart) / frequency.QuadPart;
    cout << "sum_chain Time: " << cpu_time_used * 1000 << " ms\n";

    QueryPerformanceCounter(&start);
    for (int i = 0; i < m; i++)
    {
        sum_doubleloop(n, numbers);
    }

    QueryPerformanceCounter(&end);
    cpu_time_used = static_cast<double>(end.QuadPart - start.QuadPart) / frequency.QuadPart;
    cout << "sum_doubleloop Time: " << cpu_time_used * 1000 << " ms\n";

    for (int i = 0; i < n; i++)
    {
        delete[] matrix[i];
    }
    delete[] matrix;
    delete[] vector;
    delete[] result_inner_product;
    delete[] numbers;
    return 0;
}
