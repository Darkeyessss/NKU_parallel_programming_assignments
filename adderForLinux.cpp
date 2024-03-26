#include <iostream>
#include <sys/time.h> 
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
    for (int k = 1, round = 1; round <= 6; k *= 2, round++)
    {
        long long n = 1024 * k;
        long long m = 3000;
        timeval start, end;
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
            vector[i] = i + 1;
            numbers[i] = i + 1;
            for (int j = 0; j < n; j++)
            {
                matrix[i][j] = i + j + 1;
            }
        }

        gettimeofday(&start, NULL);
        column_inner_original(n, matrix, vector, result_inner_product);
        gettimeofday(&end, NULL);
        cpu_time_used = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_usec - start.tv_usec) / 1000.0;
        cout << "column_inner_original Time: " << cpu_time_used << " ms\n";

        gettimeofday(&start, NULL);
        column_inner_cache(n, matrix, vector, result_inner_product);
        gettimeofday(&end, NULL);
        cpu_time_used = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_usec - start.tv_usec) / 1000.0;
        cout << "column_inner_cache Time: " << cpu_time_used << " ms\n";

        gettimeofday(&start, NULL);
        for (int i = 0; i < m; i++)
        {
            result_sum = sum_chain(n, numbers);
        }
        gettimeofday(&end, NULL);
        cpu_time_used = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_usec - start.tv_usec) / 1000.0;
        cout << "sum_chain Time: " << cpu_time_used << " ms\n";

        gettimeofday(&start, NULL);
        for (int i = 0; i < m; i++)
        {
            sum_doubleloop(n, numbers);
        }
        gettimeofday(&end, NULL);
        cpu_time_used = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_usec - start.tv_usec) / 1000.0;
        cout << "sum_doubleloop Time: " << cpu_time_used << " ms\n";
        for (int i = 0; i < n; i++)
        {
            delete[] matrix[i];
        }
        delete[] matrix;
        delete[] vector;
        delete[] result_inner_product;
        delete[] numbers;
        cout << endl;
    }
    return 0;
}
