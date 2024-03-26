#include <iostream>
#include <windows.h>
using namespace std;

double sum_multichain_unrolled(long long n, double *a)
{
    double sum = 0;
    long long i = 0;
    for (; i <= n - 4; i += 4)
    {
        sum += a[i] + a[i + 1] + a[i + 2] + a[i + 3];
    }
    for (; i < n; i++)
    {
        sum += a[i];
    }
    return sum;
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
        result_sum = sum_multichain_unrolled(n, numbers);
    }
    QueryPerformanceCounter(&end);
    cpu_time_used = static_cast<double>(end.QuadPart - start.QuadPart) / frequency.QuadPart;
    cout << "sum_multichain_unrolled Time: " << cpu_time_used * 1000 << " ms\n";

    delete[] numbers;
    return 0;
}
