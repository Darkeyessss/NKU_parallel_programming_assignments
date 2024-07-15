#include <iostream>
#include <vector>
#include <cmath>
#include <mpi.h>
#include <numeric>

class SVM
{
public:
    SVM(double learning_rate, double lambda, int iterations)
        : learning_rate(learning_rate), lambda(lambda), iterations(iterations) {}

    void fit(const std::vector<std::vector<double>> &X, const std::vector<int> &y, int rank, int size)
    {
        int n_samples = X.size();
        int n_features = X[0].size();
        weights.resize(n_features, 0.0);
        bias = 0.0;

        int chunk_size = n_samples / size;
        int start = rank * chunk_size;
        int end = (rank == size - 1) ? n_samples : start + chunk_size;

        for (int it = 0; it < iterations; ++it)
        {
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

            // Reduce weights and bias across all processes
            MPI_Allreduce(MPI_IN_PLACE, weights.data(), n_features, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, &bias, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

            // Average weights and bias across all processes
            if (rank == 0)
            {
                for (double &weight : weights)
                {
                    weight /= size;
                }
                bias /= size;
            }

            // Broadcast updated weights and bias to all processes
            MPI_Bcast(weights.data(), n_features, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(&bias, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
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
        return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
    }
};

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 生成较大的数据集
    const int n_samples = 10000;
    const int n_features = 20;
    std::vector<std::vector<double>> X(n_samples, std::vector<double>(n_features));
    std::vector<int> y(n_samples);

    // 初始化数据集
    if (rank == 0)
    {
        for (int i = 0; i < n_samples; ++i)
        {
            for (int j = 0; j < n_features; ++j)
            {
                X[i][j] = static_cast<double>(rand()) / RAND_MAX;
            }
            y[i] = (rand() % 2) * 2 - 1; // 随机生成 -1 或 1
        }
    }

    // 广播数据集
    for (int i = 0; i < n_samples; ++i)
    {
        MPI_Bcast(X[i].data(), n_features, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    MPI_Bcast(y.data(), n_samples, MPI_INT, 0, MPI_COMM_WORLD);

    // 定义SVM模型参数
    double learning_rate = 0.01;
    double lambda = 0.01;
    int iterations = 1000;

    // 训练SVM模型并测试训练时间
    SVM svm(learning_rate, lambda, iterations);
    auto start_time = std::chrono::high_resolution_clock::now();
    svm.fit(X, y, rank, size);
    auto end_time = std::chrono::high_resolution_clock::now();

    if (rank == 0)
    {
        std::chrono::duration<double> training_time = end_time - start_time;
        std::cout << "Training time: " << training_time.count() << " seconds" << std::endl;
    }

    // 预测新的数据点
    if (rank == 0)
    {
        std::vector<double> new_data(n_features);
        for (int j = 0; j < n_features; ++j)
        {
            new_data[j] = static_cast<double>(rand()) / RAND_MAX;
        }
        int prediction = svm.predict(new_data);
        std::cout << "Prediction for new data: " << prediction << std::endl;
    }

    MPI_Finalize();
    return 0;
}
