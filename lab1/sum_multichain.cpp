#include <iostream>
#include <windows.h>
using namespace std;

double sum_multichain(long long n, double *a)
{
    double sum1 = 0, sum2 = 0;
    for (int i = 0; i < n; i += 2)
    {
        sum1 += a[i];
        if (i + 1 < n)
            sum2 += a[i + 1];
    }
    return sum1 + sum2;
}

int main()
{
    long long n = 65536, m = 3000;
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
        result_sum = sum_multichain(n, numbers);
    }
    QueryPerformanceCounter(&end);
    cpu_time_used = static_cast<double>(end.QuadPart - start.QuadPart) / frequency.QuadPart;
    cout << "sum_multichain Time: " << cpu_time_used * 1000 << " ms\n";

    delete[] numbers;
    return 0;
}
