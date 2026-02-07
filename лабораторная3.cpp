#include <iostream>
#include <cmath>
#include <functional>
using namespace std;

// Пример функции f(x)
double f(double x) {
    return pow(x,4) - 13*x*x + 36 - 1/x;
}

// Производная функции f'(x)
double df(double x) {
    return 4*x*x*x - 26*x +2/x/x;
    // double h = 1e-6;
    // return (f(x+h)-f(x))/(x-h);
}

double newton(double x0, double tol, int max_iter) {
    double x_n = x0;
    double x_n1; // Следующее приближение

    for (int i = 0; i < max_iter; ++i) {
        double f_val = f(x_n);
        double df_val = df(x_n);

        // Проверка, чтобы не делить на ноль
        if (abs(df_val) < 1e-10) {
            cerr << "Derivative is too small." << endl;
            return x_n;
        }

        // Вычисляем следующее значение
        x_n1 = x_n - f_val / df_val;

        // Проверка на достижение заданной точности
        if (abs(x_n1 - x_n) < tol) {
            cout << "Num of iterations: " << i + 1 << endl;
            return x_n1;
        }

        x_n = x_n1; // Обновляем текущее значение
    }

    cerr << "The num of iterations has been exceeded." << endl;
    return x_n1; // Возвращаем последнее значение
}

int main() {
    double x0, tol;
    int max_iter;

    // Вводим начальное приближение, точность и максимальное количество итераций
    cout << "x0 = ";
    cin >> x0;
    cout << "tol = ";
    cin >> tol;
    cout << "Max num of iterations: ";
    cin >> max_iter;

    // Находим и выводим корень
    double root = newton(x0, tol, max_iter);
    cout << "root = " << root << endl;

    return 0;
}