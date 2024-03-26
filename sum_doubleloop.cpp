#include <iostream>
#include <windows.h>
using namespace std;

void sum_doubleloop(long long n, double *a)
{
    for (int m = n; m > 1; m /= 2)
        for (int i = 0; i < m / 2; i++)
            a[i] = a[i * 2] + a[i * 2 + 1];
}

int main()
{
    long long n = 8192, m = 3000;
    LARGE_INTEGER start, end, frequency;
    double cpu_time_used;
    double *numbers = new double[n];

    QueryPerformanceFrequency(&frequency);

    for (int i = 0; i < n; i++)
    {
        numbers[i] = i + 1;
    }

    QueryPerformanceCounter(&start);
    for (int i = 0; i < m; i++)
    {
        sum_doubleloop(n, numbers);
    }
    QueryPerformanceCounter(&end);
    cpu_time_used = static_cast<double>(end.QuadPart - start.QuadPart) / frequency.QuadPart;
    cout << "sum_doubleloop Time: " << cpu_time_used * 1000 << " ms\n";

    delete[] numbers;
    return 0;
}
