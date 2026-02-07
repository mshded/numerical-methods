#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>

using namespace std;

// Функция для решения системы линейных уравнений методом Гаусса
bool simq(vector<vector<double>>& A, vector<double>& b) {
    int n = A.size(); // Число уравнений

    // Формируем расширенную матрицу
    for (int i = 0; i < n; ++i) {
        A[i].push_back(b[i]);
    }

    // Прямой ход метода Гаусса
    for (int i = 0; i < n; ++i) {
        // Находим ведущий элемент
        double max_elem = abs(A[i][i]);
        int max_row = i;
        for (int k = i + 1; k < n; ++k) {
            if (abs(A[k][i]) > max_elem) {
                max_elem = abs(A[k][i]);
                max_row = k;
            }
        }

        // Меняем текущую строку с строкой, содержащей максимальный элемент
        if (max_row != i) {
            swap(A[i], A[max_row]);
        }

        // Приведение к ступенчатому виду
        for (int k = i + 1; k < n; ++k) {
            double factor = A[k][i] / A[i][i];
            for (int j = i; j <= n; ++j) {
                A[k][j] -= factor * A[i][j];
            }
        }
    }

    // Обратная подстановка
    vector<double> x(n);
    for (int i = n - 1; i >= 0; --i) {
        x[i] = A[i][n] / A[i][i];
        for (int k = i - 1; k >= 0; --k) {
            A[k][n] -= A[k][i] * x[i];
        }
    }

    cout << "solution:" << endl;
    for (int i = 0; i < n; i++) {
        cout << "x" << i + 1 << " = " <<  x[i] << endl;
    }

    return true;
}

int main() {
    int n;

    cout << "the number of equals: ";
    cin >> n;

    vector<vector<double>> A(n, vector<double>(n)); // Матрица коэффициентов
    vector<double> b(n); // Вектор свободных членов

    cout << "koefficients of the matrix(line)" << endl;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            cin >> A[i][j];
        }
    }

    cout << "free koufficients:" << endl;
    for (int i = 0; i < n; ++i) {
        cin >> b[i];
    }

    if (!simq(A, b)) {
        cout << "no solutions" << endl;
    }

    return 0;
}
