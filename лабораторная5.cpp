#include <iostream>
#include <cmath>
#include <iomanip>

using namespace std;

// Функция, определяющая g1
double g1(double y) {
    return (sin(y) - 1.6) / -2.0; // Изменили знак, чтобы получить x
}

// Функция, определяющая g2
double g2(double x) {
    return 0.8 - cos(x + 0.5);
}

int main() {
    double x, y;
    double x_next, y_next;
    double tol = 1e-6; // Заданная точность
    int max_iter = 100; // Максимальное число итераций
    int iter = 0;

    // Начальные приближения
    x = 0.0;
    y = 0.0;

    cout << "iterations\t x\t\t y" << endl;

    do {
        // Обновляем значения x и y
        x_next = g1(y);
        y_next = g2(x);

        // Выводим текущие значения
        cout << iter + 1 << "\t\t " << x_next << "\t " << y_next << endl;

        // Проверяем на сходимость
        if (abs(x_next - x) < tol && abs(y_next - y) < tol) {
            break;
        }

        x = x_next;
        y = y_next;

        iter++;
    } while (iter < max_iter);

    if (iter == max_iter) {
        cout << "limit of iterations was achieved" << endl;
    } else {
        cout << "x = " << x << ", y = " << y << endl;
    }

    return 0;
}
