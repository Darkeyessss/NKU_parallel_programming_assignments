#include <iostream>
#include <vector>
#include <cmath>
#include <pthread.h>
#include <chrono>

// 线程数据结构
struct ThreadData
{
    std::vector<double> *weights;
    const std::vector<std::vector<double>> *X;
    const std::vector<int> *y;
    double *bias;
    double learning_rate;
    double lambda;
    int start;
    int end;
};

// 计算点积的辅助函数
double dot_product(const std::vector<double> &a, const std::vector<double> &b)
{
    double result = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        result += a[i] * b[i];
    }
    return result;
}

// 线程函数
void *thread_function(void *arg)
{
    ThreadData *data = static_cast<ThreadData *>(arg);
    std::vector<double> &weights = *(data->weights);
    const std::vector<std::vector<double>> &X = *(data->X);
    const std::vector<int> &y = *(data->y);
    double &bias = *(data->bias);
    double learning_rate = data->learning_rate;
    double lambda = data->lambda;
    int start = data->start;
    int end = data->end;

    int n_features = weights.size();

    for (int i = start; i < end; ++i)
    {
        double condition = y[i] * (dot_product(weights, X[i]) + bias);
        if (condition >= 1)
        {
            for (int j = 0; j < n_features; ++j)
            {
                weights[j] -= learning_rate * (2 * lambda * weights[j]);
            }
        }
        else
        {
            for (int j = 0; j < n_features; ++j)
            {
                weights[j] -= learning_rate * (2 * lambda * weights[j] - X[i][j] * y[i]);
            }
            bias -= learning_rate * y[i];
        }
    }

    return nullptr;
}

// SVM 类
class SVM
{
public:
    SVM(double learning_rate, double lambda, int iterations, int num_threads)
        : learning_rate(learning_rate), lambda(lambda), iterations(iterations), num_threads(num_threads) {}

    void fit(const std::vector<std::vector<double>> &X, const std::vector<int> &y)
    {
        int n_samples = X.size();
        int n_features = X[0].size();
        weights.resize(n_features, 0.0);
        bias = 0.0;

        for (int it = 0; it < iterations; ++it)
        {
            std::vector<pthread_t> threads(num_threads);
            std::vector<ThreadData> thread_data(num_threads);

            int chunk_size = n_samples / num_threads;
            for (int t = 0; t < num_threads; ++t)
            {
                thread_data[t] = {&weights, &X, &y, &bias, learning_rate, lambda, t * chunk_size, (t + 1) * chunk_size};
                if (t == num_threads - 1)
                {
                    thread_data[t].end = n_samples; // 最后一个线程处理剩余的样本
                }
                pthread_create(&threads[t], nullptr, thread_function, &thread_data[t]);
            }

            for (pthread_t &thread : threads)
            {
                pthread_join(thread, nullptr);
            }
        }
    }

    int predict(const std::vector<double> &X) const
    {
        double linear_output = dot_product(weights, X) + bias;
        return (linear_output >= 0) ? 1 : -1;
    }

private:
    std::vector<double> weights;
    double bias;
    double learning_rate;
    double lambda;
    int iterations;
    int num_threads;

    static double dot_product(const std::vector<double> &a, const std::vector<double> &b)
    {
        double result = 0.0;
        for (size_t i = 0; i < a.size(); ++i)
        {
            result += a[i] * b[i];
        }
        return result;
    }
};

int main()
{
    // 生成较大的数据集
    const int n_samples = 10000;
    const int n_features = 20;
    std::vector<std::vector<double>> X(n_samples, std::vector<double>(n_features));
    std::vector<int> y(n_samples);

    // 初始化数据集
    for (int i = 0; i < n_samples; ++i)
    {
        for (int j = 0; j < n_features; ++j)
        {
            X[i][j] = static_cast<double>(rand()) / RAND_MAX;
        }
        y[i] = (rand() % 2) * 2 - 1; // 随机生成 -1 或 1
    }

    // 定义SVM模型参数
    double learning_rate = 0.01;
    double lambda = 0.01;
    int iterations = 1000;
    int num_threads = 4; // 使用4个线程

    // 训练SVM模型并测试训练时间
    SVM svm(learning_rate, lambda, iterations, num_threads);
    auto start_time = std::chrono::high_resolution_clock::now();
    svm.fit(X, y);
    auto end_time = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> training_time = end_time - start_time;
    std::cout << "Training time: " << training_time.count() << " seconds" << std::endl;

    // 预测新的数据点
    std::vector<double> new_data(n_features);
    for (int j = 0; j < n_features; ++j)
    {
        new_data[j] = static_cast<double>(rand()) / RAND_MAX;
    }
    int prediction = svm.predict(new_data);

    std::cout << "Prediction for new data: " << prediction << std::endl;

    return 0;
}
