#include <iostream>
#include <fstream>
#include <sys/time.h>
#include <arm_neon.h>
#include <cmath>
#include <bitset>
#include <omp.h>

using namespace std;

const int TIMES = 100;
const int COL_NUM = 130;
const int R_ROW_NUM = COL_NUM;
const int R_COL_NUM = ceil(COL_NUM / 32.0);
const int E_COL_NUM = R_COL_NUM;
const string DATA_PATH = "/home/s2110508/SIMD/data/Grobner/1_130_22_8/";

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
    for (; i < numCols; i++)
        dest[i] ^= src[i];
}

void EliminateUsingNeon(unsigned int **R, unsigned int **E, int *firstElement, int numRows)
{
#pragma omp parallel for
    for (int i = 0; i < numRows; i++)
    {
        while (firstElement[i] != -1)
        {
            int rowIndex = firstElement[i];
            if (R[rowIndex][0] == 0 && R[rowIndex][R_COL_NUM - 1] == 0)
            {
                memcpy(R[rowIndex], E[i], R_COL_NUM * sizeof(unsigned int));
                break;
            }
            else
            {
                XorNEON(E[i], R[rowIndex], E_COL_NUM);
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
    ofstream outputFile(DATA_PATH + "res.txt");
    if (!outputFile)
    {
        cerr << "Failed to open the Result File！" << endl;
        return;
    }
    for (int i = 0; i < E_RowNum; i++)
    {
        if (First[i] == -1)
        {
            outputFile << endl;
            continue;
        }
        for (int j = 0; j < E_COL_NUM; j++)
        {
            if (E[i][j] == 0)
            {
                continue;
            }
            int number = 0;
            bitset<32> Bit = E[i][j];
            for (int k = 31; k >= 0; k--)
            {
                if (Bit.test(k))
                {
                    number = 32 * (E_COL_NUM - j - 1) + k;
                    outputFile << number << " ";
                }
            }
        }
        outputFile << endl;
    }
    outputFile.close();
}

int main()
{
    struct timeval begin, end;
    double timeuse1 = 0, timeuse2 = 0;

    ifstream file(DATA_PATH + "2.txt");
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

    int *First = new int[E_RowNum];
    unsigned int **E = new unsigned int *[E_RowNum];
    for (int i = 0; i < E_RowNum; i++)
    {
        E[i] = new unsigned int[E_COL_NUM];
    }
    unsigned int **R = new unsigned int *[R_ROW_NUM];
    for (int i = 0; i < R_ROW_NUM; i++)
    {
        R[i] = new unsigned int[R_COL_NUM];
    }

    for (int i = 0; i < TIMES; i++)
    {
        Init_Zero(E, E_RowNum, E_COL_NUM);
        Init_E(E, First);
        Init_Zero(R, R_ROW_NUM, R_COL_NUM);
        Init_R(R);
        gettimeofday(&begin, NULL);
        serial(E_RowNum, R, E, First);
        gettimeofday(&end, NULL);
        timeuse1 += (end.tv_sec - begin.tv_sec) * 1000 + (double)(end.tv_usec - begin.tv_usec) / 1000.0;
    }
    cout << "col=" << COL_NUM << "  serial:" << timeuse1 / TIMES << "ms" << endl;

    for (int i = 0; i < TIMES; i++)
    {
        Init_Zero(E, E_RowNum, E_COL_NUM);
        Init_E(E, First);
        Init_Zero(R, R_ROW_NUM, R_COL_NUM);
        Init_R(R);
        gettimeofday(&begin, NULL);
        EliminateUsingNeon(R, E, First, E_RowNum);
        gettimeofday(&end, NULL);
        timeuse2 += (end.tv_sec - begin.tv_sec) * 1000 + (double)(end.tv_usec - begin.tv_usec) / 1000.0;
    }
    cout << "col=" << COL_NUM << "  Neon:" << timeuse2 / TIMES << "ms" << endl;

    save_result(E, First, E_RowNum);

    delete[] First;
    for (int i = 0; i < E_RowNum; i++)
    {
        delete[] E[i];
    }
    delete[] E;
    for (int i = 0; i < R_ROW_NUM; i++)
    {
        delete[] R[i];
    }
    delete[] R;

    return 0;
}
