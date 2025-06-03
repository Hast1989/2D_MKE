// LU.cpp : Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.
//

#include <iostream>
#include <chrono>
#include<omp.h>
#include<cmath>

void MatrPrint(double* A, int n, int m)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            std::cout << A[i * m + j] << ' ';
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}
double NormMatr1(double* A, int n, int m)
{
    double s, max;
    max = 0;
    for (int i = 0; i < n; i++)
    {
        s = 0;
        for (int j = 0; j < m; j++)
        {
            s += std::fabs(A[i * m + j]);
        }
        if (max < s)
        {
            max = s;
        }
    }
    return max;
}
void MatrZ(double* A, int n, int m)
{
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            A[i * m + j] = 0.;
}
void LU(double* A, int n)
{
    for (int i = 0; i < n - 1; i++)
    {
        for (int j = i + 1; j < n; j++)
            A[j * n + i] = A[j * n + i] / A[i * n + i];
        for (int j = i + 1; j < n; j++)
            for (int k = i + 1; k < n; k++)
                A[j * n + k] += -(A[j * n + i] * A[i * n + k]);
    }
}
void LUp(double* A, int n)
{
    for (int i = 0; i < n - 1; i++)
    {
#pragma omp parallel for
        for (int j = i + 1; j < n; j++)
            A[j * n + i] = A[j * n + i] / A[i * n + i];
#pragma omp parallel for
        for (int j = i + 1; j < n; j++)
            for (int k = i + 1; k < n; k++)
                A[j * n + k] += -(A[j * n + i] * A[i * n + k]);
    }
}
void A_12(double* A12, double* A11, int b, int nb)
{
    double s;
    for (int i = 0; i < nb; i++)
    {
        for (int j = 0; j < b; j++)
        {
            s = 0;
            for (int k = 0; k < j; k++)
            {
                s += A11[j * b + k] * A12[(k)*nb + i];
            }
            A12[j * nb + i] -= s;
        }
    }
}
void A_21(double* A21, double* A11, int nb, int b)
{
    double s;
    for (int i = 0; i < nb; i++)
    {
        for (int j = 0; j < b; j++)
        {
            s = 0;
            for (int k = 0; k < j; k++)
            {
                s += A11[j + k * b] * A21[k + i * b];
            }
            A21[j + i * b] = (A21[j + i * b] - s) / A11[j + j * b];
        }
    }
}
void newA22(double* A, double* A21, double* A12, int nb, int b, int it, int n)
{

    for (int i = 0; i < nb; i++)
        for (int k = 0; k < b; k++)
            for (int j = 0; j < nb; j++)
                A[(it + b + i) * n + it + b + j] -= A21[i * b + k] * A12[k * nb + j];
}
void newA22paral(double* A, double* A21, double* A12, int nb, int b, int it, int n)
{
#pragma omp parallel for
    for (int i = 0; i < nb; i++)
        for (int k = 0; k < b; k++)
            for (int j = 0; j < nb; j++)
                A[(it + b + i) * n + it + b + j] -= A21[i * b + k] * A12[k * nb + j];
}
void blockLU(double* A, int n, int b)
{
    int it;
    double* A12;
    double* A21;
    double* A11;
    A11 = new double[b * b];
    A12 = new double[b * (n - b)];
    A21 = new double[(n - b) * b];
    for (int it = 0; it < n; it += b)
    {
        for (int i = 0; i < b; i++)
            for (int j = 0; j < b; j++)
                A11[i * b + j] = A[(it + i) * n + it + j];
        for (int i = 0; i < n - it - b; i++)
            for (int j = 0; j < b; j++)
                A21[i * b + j] = A[(it + b + i) * n + it + j];
        for (int i = 0; i < b; i++)
            for (int j = 0; j < n - it - b; j++)
                A12[i * (n - it - b) + j] = A[(it + i) * n + it + b + j];
        LU(A11, b);
        A_21(A21, A11, n - it - b, b);
        A_12(A12, A11, b, n - it - b);
        newA22(A, A21, A12, n - it - b, b, it, n);
        for (int i = 0; i < b; i++)
            for (int j = 0; j < b; j++)
                A[(it + i) * n + it + j] = A11[i * b + j];
        for (int i = 0; i < n - it - b; i++)
            for (int j = 0; j < b; j++)
                A[(it + b + i) * n + it + j] = A21[i * b + j];
        for (int i = 0; i < b; i++)
            for (int j = 0; j < n - it - b; j++)
                A[(it + i) * n + it + b + j] = A12[i * (n - it - b) + j];
    }
    delete[] A12;
    delete[] A21;
    delete[] A11;
}
void blockLUparal(double* A, int n, int b)
{
    int it;
    double* A12;
    double* A21;
    double* A11;
    A11 = new double[b * b];
    A12 = new double[b * (n - b)];
    A21 = new double[(n - b) * b];
    for (int it = 0; it < n; it += b)
    {
        for (int i = 0; i < b; i++)
            for (int j = 0; j < b; j++)
                A11[i * b + j] = A[(it + i) * n + it + j];
        for (int i = 0; i < n - it - b; i++)
            for (int j = 0; j < b; j++)
                A21[i * b + j] = A[(it + b + i) * n + it + j];
        for (int i = 0; i < b; i++)
            for (int j = 0; j < n - it - b; j++)
                A12[i * (n - it - b) + j] = A[(it + i) * n + it + b + j];
        LUp(A11, b);
#pragma omp parallel num_threads(2)
        {
            if (omp_get_thread_num() == 0)
            {
                A_21(A21, A11, n - it - b, b);
            }
            else
            {
                A_12(A12, A11, b, n - it - b);
            }
        }
        newA22paral(A, A21, A12, n - it - b, b, it, n);
        for (int i = 0; i < b; i++)
            for (int j = 0; j < b; j++)
                A[(it + i) * n + it + j] = A11[i * b + j];
        for (int i = 0; i < n - it - b; i++)
            for (int j = 0; j < b; j++)
                A[(it + b + i) * n + it + j] = A21[i * b + j];
        for (int i = 0; i < b; i++)
            for (int j = 0; j < n - it - b; j++)
                A[(it + i) * n + it + b + j] = A12[i * (n - it - b) + j];
    }
    delete[] A12;
    delete[] A21;
    delete[] A11;
}
void test(int n, int b)
{
    double* A;
    double* C;
    double* B;
    A = new double[n * n];
    C = new double[n * n];
    B = new double[n * n];
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (i <= j)
            {
                A[i * n + j] = cos(i + j);
            }
            else
            {
                A[i * n + j] = 1. + log((1.33 * (i)) / (j + 2.44));
            }
        }
    }
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
        {
            C[i * n + j] = A[i * n + j];
            //B[i * n + j] = A[i * n + j];
        }
    //std::cout << "Matrix A:" << std::endl;
    //MatrPrint(C, n, n);
    auto begin = std::chrono::steady_clock::now();
    LU(C, n);
    auto end = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
    std::cout << "Time L&U: " << elapsed_ms.count() << " size: " << n << std::endl;
    //std::cout << "Matrix L&U:" << std::endl;
    //MatrPrint(C, n, n);
    //begin = std::chrono::steady_clock::now();
    //LU(A, n);
    ////blockLU(A, n, b);
    //end = std::chrono::steady_clock::now();
    //elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
    //std::cout << "Time L&U blok: " << elapsed_ms.count() << " size: " << n << " block: " << b << std::endl;
    /*std::cout << "Time L&U parallel: " << elapsed_ms.count() << " size: " << n << std::endl;*/
    //std::cout << "Matrix L&U blok:" << std::endl;
    //MatrPrint(A, n, n);
    begin = std::chrono::steady_clock::now();
    blockLU(A, n, b);
    end = std::chrono::steady_clock::now();
    elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
    std::cout << "Time L&U block: " << elapsed_ms.count() << " size: " << n << " block: " << b << std::endl;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
        {
            A[i * n + j] -= C[i * n + j];
            B[i * n + j] -= C[i * n + j];

        }
    //std::cout << "Matrix L&U blok - L&U:" << std::endl;
    //MatrPrint(C, n, n);
    std::cout << std::endl;
    std::cout << "Norm1Matrix (L&U parallel-L&U): " << NormMatr1(A, n, n) << std::endl;
    //std::cout << "Norm1Matrix (L&U blokparal-L&U): " << NormMatr1(B, n, n) << std::endl;
    delete[] A;
    //delete[] B;
    delete[] C;

}
void testparallel(int n, int b)
{
    double* A;
    A = new double[n * n];
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (i <= j)
            {
                A[i * n + j] = cos(i + j);
            }
            else
            {
                A[i * n + j] = 1. + log((1.33 * (i)) / (j + 2.44));
            }
        }
    }
    
    auto begin = std::chrono::steady_clock::now();
    LUp(A, n);
    auto end = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
    std::cout << "Time L&U parallel: " << elapsed_ms.count() << " size: " << n << " block: " << b << std::endl;
    delete[] A;
}
void testonlyblock(int n, int b)
{
    double* A;
    double* B;
    A = new double[n * n];
    B = new double[n * n];
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (i <= j)
            {
                A[i * n + j] = cos(i + j);
            }
            else
            {
                A[i * n + j] = 1. + log((1.33 * (i)) / (j + 2.44));
            }
        }
    }
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
        {
            B[i * n + j] = A[i * n + j];
        }
    auto begin = std::chrono::steady_clock::now();
    auto end = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
    begin = std::chrono::steady_clock::now();
    blockLU(A, n, b);
    end = std::chrono::steady_clock::now();
    elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
    std::cout << "Time L&U blok: " << elapsed_ms.count() << " size: " << n << " block: " << b << std::endl;
    begin = std::chrono::steady_clock::now();
    blockLUparal(B, n, b);
    end = std::chrono::steady_clock::now();
    elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
    std::cout << "Time L&U blokparal: " << elapsed_ms.count() << " size: " << n << " block: " << b << std::endl;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
        {
            B[i * n + j] -= A[i * n + j];
        }
    std::cout << std::endl;
    std::cout << "Norm1Matrix (L&U blokparal-L&U block): " << NormMatr1(B, n, n) << std::endl;
    delete[] A;
    delete[] B;
}
void testonlyblockparallel(int n, int b)
{
    double* A;
    //double* B;
    A = new double[n * n];
    //B = new double[n * n];
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (i <= j)
            {
                A[i * n + j] = cos(i + j);
            }
            else
            {
                A[i * n + j] = 1. + log((1.33 * (i)) / (j + 2.44));
            }
        }
    }
    /*for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
        {
            B[i * n + j] = A[i * n + j];
        }*/
    auto begin = std::chrono::steady_clock::now();
    auto end = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
    begin = std::chrono::steady_clock::now();
    blockLUparal(A, n, b);
    end = std::chrono::steady_clock::now();
    elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
    std::cout << "Time L&U blokparal: " << elapsed_ms.count() << " size: " << n << " block: " << b << std::endl;
   /* for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
        {
            B[i * n + j] -= A[i * n + j];
        }*/
    std::cout << std::endl;
    //std::cout << "Norm1Matrix (L&U blokparal-L&U block): " << NormMatr1(B, n, n) << std::endl;
    delete[] A;
    //delete[] B;
}
int main()
{
    int n, b;
    n = 2048;
    b = 64;
    
   // for (int j = 0; j < 5; j++)
   // {
       // omp_set_num_threads(18);
       // std::cout << "Threads: " << omp_get_max_threads() << std::endl;
      // testonlyblock(n, b);
        for (int i = 2; i < 19; i++)
        {
            omp_set_num_threads(i);
            std::cout << "Threads: " << omp_get_max_threads() << std::endl;
            testonlyblockparallel(n, b);
            std::cout << "---------------------------" << std::endl;
            std::cout << std::endl;
            std::cout << std::endl;
        }
      //  n = n * 2;
    //}
    std::cout <<n<< std::endl;
    std::cout << "Hello World!\n";
    return 0;
}

// Запуск программы: CTRL+F5 или меню "Отладка" > "Запуск без отладки"
// Отладка программы: F5 или меню "Отладка" > "Запустить отладку"

// Советы по началу работы 
//   1. В окне обозревателя решений можно добавлять файлы и управлять ими.
//   2. В окне Team Explorer можно подключиться к системе управления версиями.
//   3. В окне "Выходные данные" можно просматривать выходные данные сборки и другие сообщения.
//   4. В окне "Список ошибок" можно просматривать ошибки.
//   5. Последовательно выберите пункты меню "Проект" > "Добавить новый элемент", чтобы создать файлы кода, или "Проект" > "Добавить существующий элемент", чтобы добавить в проект существующие файлы кода.
//   6. Чтобы снова открыть этот проект позже, выберите пункты меню "Файл" > "Открыть" > "Проект" и выберите SLN-файл.
