#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

using namespace std;

void printVector(const vector<double>& vec) {
    for (double val : vec) {
        cout << fixed << setprecision(2) << val << " ";
    }
    cout << endl;
}

// Функция для решения системы уравнений методом Зейделя
vector<double> seidel(const vector<vector<double>>& A, const vector<double>& b, double tolerance = 1e-6, int maxIterations = 1000) {
    int n = A.size();
    vector<double> x(n, 0.0);  // Начальное приближение (все 0)

    for (int iteration = 0; iteration < maxIterations; iteration++) {
        vector<double> xOld = x; // Сохранение старого значения для проверки сходимости

        for (int i = 0; i < n; i++) {
            double sum = b[i];

            for (int j = 0; j < n; j++) {
                if (i != j) {
                    sum -= A[i][j] * x[j];
                }
            }

            x[i] = sum / A[i][i];  // Обновление значения x[i]
        }

        // Проверка на сходимость
        double maxAbsError = 0.0;
        for (int i = 0; i < n; i++) {
            maxAbsError = max(maxAbsError, fabs(x[i] - xOld[i]));
        }

        if (maxAbsError < tolerance) {
            cout << iteration + 1 << " iterations" << endl;
            return x;  // Возвращаем результат
        }
    }

    cout << "max of iterations is reached." << endl;
    return x;  // Возвращаем результат даже если не сошлось
}

int main() {
    // Задаем матрицу A и вектор b для системы Ax = b
    vector<vector<double>> A = {
        {4, -1, 0, 0},
        {-1, 4, -1, 0},
        {0, -1, 4, -1},
        {0, 0, -1, 3}
    };

    vector<double> b = {15, 10, 10, 10};

    cout << "Solution:" << endl;

    vector<double> solution = seidel(A, b);
    printVector(solution);

    return 0;
}
