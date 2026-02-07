#include <iostream>
#include <cmath>
#include <vector>
#include <iomanip>

using namespace std;

double f1(double x, double y) {
    return sin(x+y)-1.3*x;
}

double f2(double x, double y) {
    return x*x + y*y - 1;
}

// Частные производные
double df1_dx(double x, double y) {
    double eps=0.0001;
    return (f1(x+eps,y)-f1(x,y))/eps;
}

double df1_dy(double x, double y) {
    double eps=0.0001;
    return (f1(x,y+eps)-f1(x,y))/eps;
}

double df2_dx(double x, double y) {
    double eps=0.0001;
    return (f2(x+eps,y)-f2(x,y))/eps;
}

double df2_dy(double x, double y) {
    double eps=0.0001;
    return (f2(x,y+eps)-f2(x,y))/eps;
}

int main() {
    double x0 = 1.0; 
    double y0 = 1.0; 
    double tol = 1e-6; 
    int max_iter = 100; 

    cout << fixed << setprecision(6);
    cout << "Iterations\t x\t\t y" << endl;

    for (int iter = 0; iter < max_iter; iter++) {
        // Вычисляем функции в текущих приближениях
        double f1_val = f1(x0, y0);
        double f2_val = f2(x0, y0);

        // Вычисляем якобиан
        double J11 = df1_dx(x0, y0);
        double J12 = df1_dy(x0, y0);
        double J21 = df2_dx(x0, y0);
        double J22 = df2_dy(x0, y0);

        double det = J11 * J22 - J12 * J21; // Определитель якобиана

        if (fabs(det) < 1e-6) { 
            cout << "det = 0." << endl;
            return -1;
        }

        double delta_x = (J22 * (-f1_val) - J12 * (-f2_val)) / det;
        double delta_y = (J11 * (-f2_val) - J21 * (-f1_val)) / det;

        x0 += delta_x;
        y0 += delta_y;

        cout << iter + 1 << "\t\t " << x0 << "\t " << y0 << endl;

        // Проверка на сходимость
        if (fabs(delta_x) < tol && fabs(delta_y) < tol) {
            break;
        }
    }

    cout << "x = " << x0 << ", y = " << y0 << endl;
    return 0;
}
