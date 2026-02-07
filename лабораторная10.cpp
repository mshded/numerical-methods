#include <iostream>
#include <vector>
#include <cmath>
#include <limits>

using namespace std;

// Функция для выполнения операции Гаусса
vector<double> gaussElimination(vector<vector<double>>& A, vector<double>& b) {
    int n = A.size();

    for (int i = 0; i < n; ++i) {
        double maxEl = abs(A[i][i]);
        int maxRow = i;
        for (int k = i + 1; k < n; ++k) {
            if (abs(A[k][i]) > maxEl) {
                maxEl = abs(A[k][i]);
                maxRow = k;
            }
        }

        swap(A[i], A[maxRow]);
        swap(b[i], b[maxRow]);

        // Обнуление элементов ниже главной диагонали
        for (int j = i + 1; j < n; ++j) {
            double c = -A[j][i] / A[i][i];
            for (int k = i; k < n; ++k) {
                if (i == k) {
                    A[j][k] = 0;
                } else {
                    A[j][k] += c * A[i][k];
                }
            }
            b[j] += c * b[i];
        }
    }

    // Обратная подстановка
    vector<double> x(n);
    for (int i = n - 1; i >= 0; --i) {
        x[i] = b[i] / A[i][i];
        for (int j = i - 1; j >= 0; --j) {
            b[j] -= A[j][i] * x[i];
        }
    }

    return x;
}

// Функция для оценки числа обусловленности методу Гаусса
double conditionNumber(const vector<vector<double>>& A) {
    double maxEigenvalue = 0;
    double minEigenvalue = numeric_limits<double>::max();

    // Находим максимальное и минимальное собственные значения (для симметричных матриц)
    for (const auto& row : A) {
        double rowNorm = 0;
        for (double el : row) {
            rowNorm += el * el;
        }
        rowNorm = sqrt(rowNorm);
        maxEigenvalue = max(maxEigenvalue, rowNorm);
        
        double negRowNorm = 0;
        for (double el : row) {
            negRowNorm += (el / ((el != 0) ? el : 1)) * (el / ((el != 0) ? el : 1));
        }
        minEigenvalue = min(minEigenvalue, sqrt(negRowNorm));
    }

    return maxEigenvalue / minEigenvalue;
}

void printSolution(const vector<double>& solution) {
    cout << "solution:" << endl;
    for (size_t i = 0; i < solution.size(); ++i) {
        cout << "x" << i + 1 << " = " << solution[i] << endl;
    }
}

int main() {
    // Пример системы линейных уравнений Ax = b
    vector<vector<double>> A = {
        {10, -7, 0},
        {-3, 2, 6},
        {5, -1, 5}
    };
    vector<double> b = {7, 4, 6};

    cout << "A:" << endl;
    for (const auto& row : A) {
        for (const auto& el : row) {
            cout << el << "\t";
        }
        cout << endl;
    }
    
    cout << "b:" << endl;
    for (const auto& el : b) {
        cout << el << "\t";
    }
    cout << endl;

    vector<double> solution = gaussElimination(A, b);

    printSolution(solution);

    // Оценка числа обусловленности
    double condNum = conditionNumber(A);
    cout << "condition number: " << condNum << endl;

    return 0;
}
