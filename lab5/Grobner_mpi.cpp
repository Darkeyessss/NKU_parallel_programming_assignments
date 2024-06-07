#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>

using namespace std;

const int COL_NUM = 130; // 矩阵列数
const int ROW_NUM = 130; // 矩阵行数

// 矩阵行初始化函数
void initialize_matrix(vector<vector<int>> &matrix)
{
    for (auto &row : matrix)
    {
        for (auto &element : row)
        {
            element = rand() % 100; // 填充随机值
        }
    }
}

// 打印矩阵
void print_matrix(const vector<vector<int>> &matrix)
{
    for (const auto &row : matrix)
    {
        for (const auto &element : row)
        {
            cout << element << " ";
        }
        cout << endl;
    }
}

// 简单的高斯消去函数
void gaussian_elimination(vector<vector<int>> &matrix)
{
    int size = matrix.size();
    for (int k = 0; k < size; ++k)
    {
        // 主元归一
        for (int j = k + 1; j < size; ++j)
        {
            matrix[k][j] /= matrix[k][k];
        }
        matrix[k][k] = 1;

        // 消元更新
        for (int i = k + 1; i < size; ++i)
        {
            for (int j = k + 1; j < size; ++j)
            {
                matrix[i][j] -= matrix[i][k] * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // 每个进程的矩阵部分
    vector<vector<int>> local_matrix(ROW_NUM / world_size, vector<int>(COL_NUM));

    // 初始化矩阵
    if (world_rank == 0)
    {
        vector<vector<int>> full_matrix(ROW_NUM, vector<int>(COL_NUM));
        initialize_matrix(full_matrix);
        print_matrix(full_matrix);

        // 分发矩阵的每一行到各个进程
        for (int i = 0; i < world_size; ++i)
        {
            MPI_Scatter(full_matrix.data() + i * (ROW_NUM / world_size), ROW_NUM / world_size * COL_NUM, MPI_INT,
                        local_matrix.data(), ROW_NUM / world_size * COL_NUM, MPI_INT, i, MPI_COMM_WORLD);
        }
    }
    else
    {
        MPI_Scatter(NULL, 0, MPI_INT, local_matrix.data(), ROW_NUM / world_size * COL_NUM, MPI_INT, 0, MPI_COMM_WORLD);
    }

    // 执行局部高斯消去
    gaussian_elimination(local_matrix);

    // 每个进程打印其处理部分的结果
    cout << "Process " << world_rank << " has the following matrix:" << endl;
    print_matrix(local_matrix);

    MPI_Finalize();
    return 0;
}
