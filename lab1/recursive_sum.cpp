#include <iostream>
#include <windows.h>
using namespace std;

void recursive_sum(double *a, long long n)
{
    if (n == 1)
        return;
    else
    {
        for (int i = 0; i < n / 2; i++)
        {
            a[i] += a[n - i - 1];
        }
        recursive_sum(a, n / 2);
    }
}

double sum_recursive(long long n, double *a)
{
    recursive_sum(a, n);
    return a[0];
}

int main()
{
    long long n = 8192, m = 3000;
    LARGE_INTEGER start, end, frequency;
    double cpu_time_used, result_sum;
    double *numbers = new double[n];

    for (int i = 0; i < n; i++)
    {
        numbers[i] = i + 1;
    }

    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start);
    for (int i = 0; i < m; i++)
    {
        result_sum = sum_recursive(n, numbers);
    }
    QueryPerformanceCounter(&end);
    cpu_time_used = static_cast<double>(end.QuadPart - start.QuadPart) / frequency.QuadPart;
    cout << "sum_recursive Time: " << cpu_time_used * 1000 << " ms\n";

    delete[] numbers;
    return 0;
}
