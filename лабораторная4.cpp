#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

using namespace std;

// Функция для вычисления значения многочлена
double polynomial(double x, const vector<double>& coeff) {
    double result = 0.0;
    for (int i = 0; i < coeff.size(); ++i) {
        result += coeff[i] * pow(x, coeff.size() - 1 - i);
    }
    return result;
}

// Производная многочлена
double polynomial_derivative(double x, const vector<double>& coeff) {
    double result = 0.0;
    for (int i = 0; i < coeff.size() - 1; ++i) {
        result += coeff[i] * (coeff.size() - 1 - i) * pow(x, coeff.size() - 2 - i);
    }
    return result;
}

// Метод Ньютона для нахождения корней
double newton_method(double initial_guess, const vector<double>& coeff, double tol, int max_iterations) {
    double x0 = initial_guess;
    double x1;

    for (int i = 0; i < max_iterations; ++i) {
        double f_x0 = polynomial(x0, coeff);
        double f_prime_x0 = polynomial_derivative(x0, coeff);

        // Проверяем, не равна ли производная нулю
        if (fabs(f_prime_x0) < 1e-12) {
            cout << "f'(x) -> 0" << endl;
            return x0; // Возвращаем текущее значение
        }

        x1 = x0 - f_x0 / f_prime_x0; // Вычисляем новое значение

        // Проверяем на сходимость
        if (fabs(x1 - x0) < tol) {
            cout << "root = " << x1 << endl;
            return x1;
        }
        x0 = x1; 
    }

    cout << "can't find the root, limit of iterations is full" << endl;
    return x0; 
}

int main() {
    vector<double> coeff = {1, 1.2, -3, 2, -4, 7, 0.84}; 

    double x0;
    cout << "x0 = ";
    cin >> x0;

    double tol = 1e-6;
    int max_iterations = 100;

    // Вызываем метод Ньютона для нахождения корня
    newton_method(x0, coeff, tol, max_iterations);

    return 0;
}
