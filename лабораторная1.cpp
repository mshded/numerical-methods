#include <iostream>
#include <cmath>
#include <functional>
using namespace std;

double f(double x) {
    return pow(x,4) - 13*x*x + 36 - 1/x;
}

double bisection(double a, double b, double tol) {
    if (f(a) * f(b) >= 0) {
        cerr << "Error: f(a) and f(b) must be different signs." << endl;
        return NAN; // возвращаем нечисловое значение, чтобы указать на ошибку
    }
    int k = 0;
    double c;
    while ((b - a) >= tol) {
        k+=1;
        // Находим середину интервала
        c = (a + b) / 2;

        // Проверяем, является ли c корнем
        if (f(c) == 0.0) {
            break;
        }

        // Определяем, в каком подинтервале будет находиться корень
        if (f(c) * f(a) < 0) {
            b = c; // Корень находится в [a, c]
        } else {
            a = c; // Корень находится в [c, b]
        }
    }
    cout << "Num of iterations: " << k << endl;
    return c; // Возвращаем приближенное значение корня
}

int main() {
    double a, b, tol;
    setlocale(LC_ALL, "Russian");
    // Вводим значения для интервала и точности
    cout << "a = ";
    cin >> a;
    cout << "b = ";
    cin >> b;
    cout << "tol = ";
    cin >> tol;

    // Находим и выводим корень
    double root = bisection(a, b, tol);
    if (!isnan(root)) {
        cout << "root = " << root << endl;
    }

    return 0;
}
