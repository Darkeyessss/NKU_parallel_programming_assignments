#include <iostream>
#include <vector>
#include <cuda_runtime.h>

__global__ void compute_gradients(const double *X, const int *y, double *weights, double *d_weights, double *bias, double *d_bias, double learning_rate, double lambda, int n_samples, int n_features)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_samples)
    {
        double dot_product = 0.0;
        for (int j = 0; j < n_features; ++j)
        {
            dot_product += weights[j] * X[i * n_features + j];
        }
        double condition = y[i] * (dot_product + *bias);
        if (condition < 1)
        {
            for (int j = 0; j < n_features; ++j)
            {
                atomicAdd(&d_weights[j], y[i] * X[i * n_features + j]);
            }
            atomicAdd(d_bias, y[i]);
        }
    }
}

__global__ void update_weights(double *weights, double *d_weights, double *bias, double *d_bias, double learning_rate, double lambda, int n_features)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n_features)
    {
        weights[j] -= learning_rate * (d_weights[j] / lambda);
        d_weights[j] = 0.0;
    }
    if (j == 0)
    {
        *bias -= learning_rate * (*d_bias / lambda);
        *d_bias = 0.0;
    }
}

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

        double *d_X, *d_weights, *d_d_weights, *d_bias, *d_d_bias;
        int *d_y;

        cudaMalloc(&d_X, n_samples * n_features * sizeof(double));
        cudaMalloc(&d_y, n_samples * sizeof(int));
        cudaMalloc(&d_weights, n_features * sizeof(double));
        cudaMalloc(&d_d_weights, n_features * sizeof(double));
        cudaMalloc(&d_bias, sizeof(double));
        cudaMalloc(&d_d_bias, sizeof(double));

        std::vector<double> flat_X;
        for (const auto &row : X)
        {
            flat_X.insert(flat_X.end(), row.begin(), row.end());
        }

        cudaMemcpy(d_X, flat_X.data(), n_samples * n_features * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, y.data(), n_samples * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_weights, weights.data(), n_features * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_bias, &bias, sizeof(double), cudaMemcpyHostToDevice);
        cudaMemset(d_d_weights, 0, n_features * sizeof(double));
        cudaMemset(d_d_bias, 0, sizeof(double));

        int block_size = 256;
        int grid_size_samples = (n_samples + block_size - 1) / block_size;
        int grid_size_features = (n_features + block_size - 1) / block_size;

        for (int it = 0; it < iterations; ++it)
        {
            compute_gradients<<<grid_size_samples, block_size>>>(d_X, d_y, d_weights, d_d_weights, d_bias, d_d_bias, learning_rate, lambda, n_samples, n_features);
            update_weights<<<grid_size_features, block_size>>>(d_weights, d_d_weights, d_bias, d_d_bias, learning_rate, lambda, n_features);
        }

        cudaMemcpy(weights.data(), d_weights, n_features * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(&bias, d_bias, sizeof(double), cudaMemcpyDeviceToHost);

        cudaFree(d_X);
        cudaFree(d_y);
        cudaFree(d_weights);
        cudaFree(d_d_weights);
        cudaFree(d_bias);
        cudaFree(d_d_bias);
    }

    int predict(const std::vector<double> &X) const
    {
        double linear_output = 0.0;
        for (size_t i = 0; i < weights.size(); ++i)
        {
            linear_output += weights[i] * X[i];
        }
        linear_output += bias;
        return (linear_output >= 0) ? 1 : -1;
    }

private:
    std::vector<double> weights;
    double bias;
    double learning_rate;
    double lambda;
    int iterations;
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
