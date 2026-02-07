#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

using namespace std;

// Функция для печати матрицы
void printMatrix(const vector<vector<double>>& mat) {
    for (const auto& row : mat) {
        for (double val : row) {
            cout << fixed << setprecision(2) << val << "\t";
        }
        cout << endl;
    }
}

// Функция для транспонирования матрицы
vector<vector<double>> transpose(const vector<vector<double>>& mat) {
    int rows = mat.size();
    int cols = mat[0].size();
    vector<vector<double>> trans(cols, vector<double>(rows));
    
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            trans[j][i] = mat[i][j];
    
    return trans;
}

// Функция для выполнения умножения матриц
vector<vector<double>> multiply(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    int rowsA = A.size();
    int colsA = A[0].size();
    int colsB = B[0].size();
    
    vector<vector<double>> result(rowsA, vector<double>(colsB, 0));
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            for (int k = 0; k < colsA; k++) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return result;
}

// Функция для вычисления SVD (ограниченная версия для 2x2 матриц)
void svd(const vector<vector<double>>& A, vector<vector<double>>& U, vector<vector<double>>& S, vector<vector<double>>& V) {
    // Упрощенная реализация для матриц 2x2
    double a = A[0][0], b = A[0][1];
    double c = A[1][0], d = A[1][1];
    
    // Нахождение сингулярных чисел
    double sigma1 = sqrt(pow(a, 2) + pow(c, 2)); // s1
    double sigma2 = sqrt(pow(b, 2) + pow(d, 2)); // s2

    // Примерные значения для матриц U, S, V
    U = {{1, 0}, {0, 1}};
    S = {{sigma1, 0}, {0, sigma2}};
    V = {{1, 0}, {0, 1}};
}

// Функция для вычисления вектора d = U^T * b
vector<double> calculateD(const vector<vector<double>>& U, const vector<double>& b) {
    vector<double> d(U.size(), 0.0);
    auto U_T = transpose(U);
    for (size_t i = 0; i < U_T.size(); ++i) {
        for (size_t j = 0; j < U_T[i].size(); ++j) {
            d[i] += U_T[i][j] * b[j];
        }
    }
    return d;
}

// Главная функция
int main() {
    int m, n;

    // Ввод размеров матрицы
    cout << "lines (m): ";
    cin >> m;
    cout << "columns (n): ";
    cin >> n;

    // Ввод матрицы A
    vector<vector<double>> A(m, vector<double>(n));
    cout << "A:" << endl;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            cin >> A[i][j];
        }
    }

    // Ввод вектора b
    vector<double> b(m);
    cout << "b:" << endl;
    for (int i = 0; i < m; ++i) {
        cin >> b[i];
    }

    // 1. Вычисление SVD
    vector<vector<double>> U(m, vector<double>(m));
    vector<vector<double>> S(m, vector<double>(n));
    vector<vector<double>> V(n, vector<double>(n));

    svd(A, U, S, V);

    // Печать результатов
    cout << "U:" << endl;
    printMatrix(U);
    cout << "S:" << endl;
    printMatrix(S);
    cout << "V:" << endl;
    printMatrix(V);

    // 2. Вычисление d = U^T * b
    vector<double> d = calculateD(U, b);
    cout << "d (U^T * b):" << endl;
    for (double value : d) {
        cout << fixed << setprecision(2) << value << " ";
    }
    cout << endl;

    system("pause");
    return 0;
}
