#include <iostream>
#include <cmath>
#include <iomanip>

using namespace std;

double g1(double x2, double x3) {
    return (200 - 6 * x2 + 2 * x3) / 100;
}

double g2(double x1, double x3) {
    return (600 - 6 * x1 + 10 * x3) / 200;
}

double g3(double x1, double x2) {
    return (500 - x1 - 2 * x2) / 100;
}

int main() {
    double x1 = 0, x2 = 0, x3 = 0; 
    double tol = 1e-6; 
    int max_iterations = 100; 

    for (int k = 0; k < max_iterations; ++k) {
        double x1_new = g1(x2, x3);
        double x2_new = g2(x1, x3);
        double x3_new = g3(x1, x2);

        // Проверка на сходимость
        if (fabs(x1_new - x1) < tol && fabs(x2_new - x2) < tol && fabs(x3_new - x3) < tol) {
            x1 = x1_new;
            x2 = x2_new;
            x3 = x3_new;
            break; 
        }

        x1 = x1_new;
        x2 = x2_new;
        x3 = x3_new;
    }

    cout << "solution:" << endl;
    cout << "x1 = " << x1 << endl;
    cout << "x2 = " << x2 << endl;
    cout << "x3 = " << x3 << endl;

    return 0;
}
