#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

class SVM
{
public:
    SVM(double learning_rate, double lambda, int iterations)
        : learning_rate(learning_rate), lambda(lambda), iterations(iterations) {}

    void fit(const std::vector<std::vector<double>> &X, const std::vector<int> &y)
    {
        int n_samples = X.size();
        int n_features = X[0].size();
        weights.resize(n_features, 0.0);
        bias = 0.0;

        for (int it = 0; it < iterations; ++it)
        {
            for (int i = 0; i < n_samples; ++i)
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

    // 训练SVM模型并测试训练时间
    SVM svm(learning_rate, lambda, iterations);
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
