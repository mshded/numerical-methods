#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

using namespace std;

// Функция для проверки симметричности матрицы
bool isSymmetric(const vector<vector<double>>& A) {
    int n = A.size();
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (A[i][j] != A[j][i]) {
                return false; // Если найдено несоответствие, не симметрична
            }
        }
    }
    return true;
}

// Функция для проверки положительной определенности
bool isPositiveDefinite(const vector<vector<double>>& A) {
    int n = A.size();
    vector<double> y(n, 0.0);

    // Проверяем через Cholesky разложение
    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        for (int j = 0; j < i; ++j) {
            sum += y[j] * y[j]; // Подсчёт суммы y_j^2
        }
        y[i] = A[i][i] - sum;
        if (y[i] <= 0) {
            return false; // Если находим не положительное значение
        }
    }
    return true;
}

// Функция для Cholesky разложения
void cholesky(const vector<vector<double>>& A, vector<vector<double>>& L) {
    int n = A.size();
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            double sum = 0.0;
            for (int k = 0; k < j; ++k) {
                sum += L[i][k] * L[j][k];
            }
            if (i == j) {
                L[i][j] = sqrt(A[i][i] - sum); // Диагональные элементы
            } else {
                L[i][j] = (A[i][j] - sum) / L[j][j]; // Недиагональные элементы
            }
        }
    }
}

// Функция для обращения матрицы
void invertMatrix(const vector<vector<double>>& L, vector<vector<double>>& A_inv) {
    int n = L.size();
    vector<vector<double>> Y(n, vector<double>(n, 0.0));
    vector<vector<double>> X(n, vector<double>(n, 0.0));
    
    // Решаем LY = I
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            Y[i][j] = (i == j) ? 1.0 : 0.0; // Единичная матрица
            for (int k = 0; k < i; ++k) {
                Y[i][j] -= L[i][k] * Y[k][j];
            }
            Y[i][j] /= L[i][i];
        }
    }
    
    // Решаем L^TX = Y
    for (int i = n - 1; i >= 0; --i) {
        for (int j = 0; j < n; ++j) {
            X[i][j] = Y[i][j];
            for (int k = i + 1; k < n; ++k) {
                X[i][j] -= L[k][i] * X[k][j];
            }
            X[i][j] /= L[i][i];
        }
    }
    
    // X является обратной матрицей A_inv
    A_inv = X;
}

// Функция для вывода матрицы
void printMatrix(const vector<vector<double>>& mat) {
    for (const auto& row : mat) {
        for (double val : row) {
            cout << fixed << setprecision(2) << val << " ";
        }
        cout << endl;
    }
}

int main() {
    // Задание матрицы A
    vector<vector<double>> A = {
        {1, 1, 1},
        {1, 2, 2},
        {1, 2, 3}
    };

    int n = A.size();
    
    // 1. Проверка симметричности
    if (!isSymmetric(A)) {
        cout << "matrix is not symmetric." << endl;
        return 1;
    }

    // 2. Проверка положительной определенности
    if (!isPositiveDefinite(A)) {
        
        cout << "matrix is not >." << endl;
        return 1;
    }
    
    // 3. Cholesky разложение
    vector<vector<double>> L(n, vector<double>(n, 0.0));
    cholesky(A, L);
    cout << "matrix L:" << endl;
    printMatrix(L);

    // 4. Обратная матрица
    vector<vector<double>> A_inv(n, vector<double>(n, 0.0));
    invertMatrix(L, A_inv);
    cout << "reverse matrix A_inv:" << endl;
    printMatrix(A_inv);
    system("pause");
    return 0;
}
