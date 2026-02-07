#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

using namespace std;

void print_matrix(const vector<vector<double>>& matrix) {
    for (const auto& row : matrix) {
        for (const auto& val : row) {
            cout << fixed << setprecision(4) << val << "\t";
        }
        cout << endl;
    }
}

void jacobi_method(const vector<vector<double>>& A, vector<vector<double>>& eigenvectors, vector<double>& eigenvalues) {
    int n = A.size();
    eigenvectors = vector<vector<double>>(n, vector<double>(n, 0));
    for (int i = 0; i < n; ++i) {
        eigenvectors[i][i] = 1.0; // Инициализация единичной матрицы
    }

    vector<vector<double>> B = A;
    // Jacobi rotation
    int max_iterations = 100;
    for (int iter = 0; iter < max_iterations; ++iter) {
        double max_off_diag = 0.0;
        int p = 0, q = 0;

        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                if (fabs(B[i][j]) > fabs(max_off_diag)) {
                    max_off_diag = B[i][j];
                    p = i;
                    q = j;
                }
            }
        }

        if (fabs(max_off_diag) < 1e-10) {
            break; // Достигнута сходимость
        }

        double theta = 0.5 * atan2(2 * B[p][q], B[q][q] - B[p][p]);
        double c = cos(theta);
        double s = sin(theta);

        // Rotate
        for (int i = 0; i < n; ++i) {
            if (i != p && i != q) {
                double temp = B[i][p];
                B[i][p] = c * temp - s * B[i][q];
                B[i][q] = s * temp + c * B[i][q];
            }
        }

        double app = c * c * B[p][p] - 2 * s * c * B[p][q] + s * s * B[q][q];
        double aqq = s * s * B[p][p] + 2 * s * c * B[p][q] + c * c * B[q][q];
        double apq = 0.0;

        B[p][p] = app;
        B[q][q] = aqq;
        B[p][q] = apq;
        B[q][p] = apq;

        // Update eigenvectors
        vector<double> temp_p(n), temp_q(n);
        for (int i = 0; i < n; ++i) {
            temp_p[i] = eigenvectors[i][p];
            temp_q[i] = eigenvectors[i][q];
        }

        for (int i = 0; i < n; ++i) {
            eigenvectors[i][p] = c * temp_p[i] - s * temp_q[i];
            eigenvectors[i][q] = s * temp_p[i] + c * temp_q[i];
        }
    }

    // Сохраняем собственные значения
    for (int i = 0; i < n; ++i) {
        eigenvalues[i] = B[i][i];
    }
}

void svd(const vector<vector<double>>& A, vector<vector<double>>& U, vector<double>& S, vector<vector<double>>& V) {
    int m = A.size();
    int n = A[0].size();

    vector<vector<double>> AT(n, vector<double>(m));
    // Вычисление A^T
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            
            AT[j][i] = A[i][j];
        }
    }

    // Вычисление A^T * A
    vector<vector<double>> ATA(n, vector<double>(n, 0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < m; ++k) {
                ATA[i][j] += AT[i][k] * A[k][j];
            }
        }
    }

    // Находим собственные значения и векторы для A^T * A
    vector<double> eigenvalues(n);
    jacobi_method(ATA, V, eigenvalues);

    // Сингулярные значения
    S.resize(n);
    for (int i = 0; i < n; ++i) {
        S[i] = sqrt(max(0.0, eigenvalues[i])); // Извлекаем корень из собственных значений
    }

    // Теперь находим матрицу U
    U = vector<vector<double>>(m, vector<double>(n, 0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            for (int k = 0; k < n; ++k) {
                U[j][i] += A[j][k] * V[k][i];
            }
        }
        double norm = 0;
        for (int j = 0; j < m; ++j) {
            norm += U[j][i] * U[j][i];
        }
        norm = sqrt(norm);
        for (int j = 0; j < m; ++j) {
            U[j][i] /= norm; // Нормируем вектор
        }
    }
}

int main() {
    vector<vector<double>> A = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9},
        {10, 11, 12}
    };

    vector<vector<double>> U;
    vector<double> S;
    vector<vector<double>> V;

    svd(A, U, S, V);

    cout << "Matrix U:" << endl;
    print_matrix(U);

    cout << "\nSingular values:" << endl;
    for (auto s : S) {
        cout << s << "\t";
    }
    cout << endl;

    cout << "\nMatrix V:" << endl;
    print_matrix(V);

    return 0;
}
