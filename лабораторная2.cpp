#include <iostream>
#include <cmath>
#include <iomanip>
#include <limits>

using namespace std;

double f(double x) {
    return x * x * x * x - 13 * x * x + 36 - 1 / x;
}

double g(double x) {
    return pow(1 / x + 13 * x * x - 36, 0.25);
}

// Метод простых итераций
double siter(double x0, double epsilon, int max_iterations) {
    double x_n = x0;
    for (int i = 0; i < max_iterations; i++) {
        double x_next = g(x_n);
        cout << "Iteration " << i + 1 << ": x = " << x_next << endl;

        // Проверка на сходимость
        if (fabs(x_next - x_n) < epsilon) {
            return x_next; 
        }
        x_n = x_next;
    }
    cout << "Maximum number of iterations reached." << endl;
    return x_n; 
}

int main() {
    double x0;
    double epsilon = 1e-6; 
    int max_iterations = 100; 

    cout << "x0 = ";
    cin >> x0;

    if (abs(x0) < numeric_limits<double>::epsilon()) {
        cout << "x0 != 0" << endl;
        return 1;
    }

    double root = siter(x0, epsilon, max_iterations);
    cout << "root = " << root << endl;

    return 0;
}
