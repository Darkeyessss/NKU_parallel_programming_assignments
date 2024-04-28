#include <iostream>
#include <fstream>
#include <sys/time.h>
#include <arm_neon.h>
#include <cmath>
#include <bitset>

using namespace std;

const int TIMES = 100;
const int COL_NUM = 130;
const int R_ROW_NUM = COL_NUM;
const int R_COL_NUM = ceil(COL_NUM / 32.0);
const int E_COL_NUM = R_COL_NUM;
const string DATA_PATH = "/home/s2110508/SIMD/data/Grobner/1_130_22_8/";

// Function to perform XOR operation using Neon intrinsics
void XorNEON(unsigned int *dest, const unsigned int *src, int numCols)
{
    int i = 0;
    for (; i + 4 <= numCols; i += 4)
    {
        uint32x4_t r = vld1q_u32(&dest[i]);
        uint32x4_t e = vld1q_u32(&src[i]);
        r = veorq_u32(r, e);
        vst1q_u32(&dest[i], r);
    }
    // Handle remaining elements
    for (; i < numCols; i++)
        dest[i] ^= src[i];
}

// Main elimination logic using Neon
void EliminateUsingNeon(unsigned int **R, unsigned int **E, int *firstElement, int numRows)
{
    for (int i = 0; i < numRows; i++)
    {
        while (firstElement[i] != -1)
        {
            int rowIndex = firstElement[i];
            if (R[rowIndex][0] == 0 && R[rowIndex][R_COL_NUM - 1] == 0)
            { // Simplified null check
                memcpy(R[rowIndex], E[i], R_COL_NUM * sizeof(unsigned int));
                break;
            }
            else
            {
                XorNEON(E[i], R[rowIndex], E_COL_NUM);
                // Reset first element - simplified
                for (int j = 0; j < E_COL_NUM; j++)
                {
                    if (E[i][j] != 0)
                    {
                        firstElement[i] = 32 * (E_COL_NUM - 1 - j) + __builtin_clz(E[i][j]);
                        break;
                    }
                }
            }
        }
    }
}
void save_result(unsigned int **E, int *First, int E_RowNum)
{
    ofstream outputFile(datapath + "res.txt");
    if (!outputFile)
    {
        cerr << "Failed to open the Result File！" << endl;
        return;
    }
    for (int i = 0; i < E_RowNum; i++)
    {
        if (First[i] == -1) // 对应行全为0
        {
            outputFile << endl;
            // cout << endl;
            continue;
        }
        for (int j = 0; j < E_ColNum; j++)
        {
            if (E[i][j] == 0)
            {
                continue;
            }
            int number = 0;
            bitset<32> Bit = E[i][j];
            for (int k = 31; k >= 0; k--)
            {
                if (Bit.test(k)) // 判断Bit的第 k 位是否为1
                {
                    number = 32 * (E_ColNum - j - 1) + k;
                    // cout << number << " ";
                    outputFile << number << " ";
                }
            }
        }
        // cout << endl;
        outputFile << endl;
    }
    outputFile.close();
}
int main()
{
    struct timeval begin, end;
    double timeuse1 = 0, timeuse2 = 0;
    // 计算被消元行矩阵的行数
    ifstream file(datapath + "2.txt");
    if (!file.is_open())
    {
        cerr << "Failed to open the file." << endl;
        return 1;
    }
    int E_RowNum = 0;
    string line;
    while (getline(file, line))
    {
        E_RowNum++;
    }
    file.close();
    // 定义被消元矩阵E和首项First、消元行矩阵R
    int *First = new int[E_RowNum];
    unsigned int **E = new unsigned int *[E_RowNum];
    for (int i = 0; i < E_RowNum; i++)
    {
        E[i] = new unsigned int[E_ColNum];
    }
    unsigned int **R = new unsigned int *[R_RowNum];
    for (int i = 0; i < R_RowNum; i++)
    {
        R[i] = new unsigned int[R_ColNum];
    }
    // 串行算法计时
    for (int i = 0; i < times; i++)
    {
        Init_Zero(E, E_RowNum, E_ColNum);
        Init_E(E, First);
        Init_Zero(R, R_RowNum, R_ColNum);
        Init_R(R);
        gettimeofday(&begin, NULL);
        serial(E_RowNum, R, E, First);
        gettimeofday(&end, NULL);
        timeuse1 += (end.tv_sec - begin.tv_sec) * 1000 + (double)(end.tv_usec - begin.tv_usec) / 1000.0;
    }
    cout << "col=" << ColNum << "  serial:" << timeuse1 / times << "ms" << endl;
    // Neon算法计时
    for (int i = 0; i < times; i++)
    {
        Init_Zero(E, E_RowNum, E_ColNum);
        Init_E(E, First);
        Init_Zero(R, R_RowNum, R_ColNum);
        Init_R(R);
        gettimeofday(&begin, NULL);
        serial(E_RowNum, R, E, First);
        gettimeofday(&end, NULL);
        timeuse2 += (end.tv_sec - begin.tv_sec) * 1000 + (double)(end.tv_usec - begin.tv_usec) / 1000.0;
    }
    cout << "col=" << ColNum << "  Neon:" << timeuse2 / times << "ms" << endl;
    save_result(E, First, E_RowNum);
    return 0;
}