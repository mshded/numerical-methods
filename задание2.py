import numpy as np
import math
import matplotlib.pyplot as plt

def f(x):
    return math.tan(0.5 * x + 0.2) - x ** 2

def get_nodes(a, b, n):
    return np.linspace(a, b, n)

def linear_spline(x_vals, y_vals, x):
    for i in range(len(x_vals) - 1):
        if x_vals[i] <= x <= x_vals[i + 1]:
            dx = x - x_vals[i]
            return y_vals[i] + (y_vals[i + 1] - y_vals[i]) / (x_vals[i + 1] - x_vals[i]) * dx
    return None


def quadratic_spline(x_vals, y_vals, x):
    n = len(x_vals) - 1
    if n < 1:
        return None
    if x < x_vals[0] or x > x_vals[-1]:
        return None
    interval = 0
    while interval < n and x > x_vals[interval + 1]:
        interval += 1
    a = y_vals[:-1]
    b = [0.0] * n
    c = [0.0] * n
    # Вычисляем первые производные
    h = [x_vals[i + 1] - x_vals[i] for i in range(n)]
    delta = [(y_vals[i + 1] - y_vals[i]) / h[i] for i in range(n)]
    # Естественные граничные условия (c[0] = 0), вычисление квадратичных коэффициентов
    c[0] = 0.0
    for i in range(1, n):
        c[i] = (delta[i] - delta[i - 1] - c[i - 1] * h[i - 1]) / h[i]
    # Вычисление линейных коэффициентов
    for i in range(n):
        b[i] = delta[i] - c[i] * h[i]
    dx = x - x_vals[interval]
    return a[interval] + b[interval] * dx + c[interval] * dx ** 2

def cubic_spline(x_vals, y_vals, x):
    n = len(x_vals) - 1
    a = y_vals[:]
    h = [x_vals[i + 1] - x_vals[i] for i in range(n)]
    alpha = [0] + [(3 / h[i]) * (a[i + 1] - a[i]) - (3 / h[i - 1]) * (a[i] - a[i - 1]) for i in range(1, n)]

    c = [0] * (n + 1)
    l = [1] + [0] * n
    mu = [0] * n
    z = [0] * (n + 1)

    for i in range(1, n):
        l[i] = 2 * (x_vals[i + 1] - x_vals[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]

    b = [0] * n
    d = [0] * n

    for j in range(n - 1, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]
        b[j] = (a[j + 1] - a[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3
        d[j] = (c[j + 1] - c[j]) / (3 * h[j])

    for i in range(n):
        if x_vals[i] <= x <= x_vals[i + 1]:
            dx = x - x_vals[i]
            return a[i] + b[i] * dx + c[i] * dx**2 + d[i] * dx**3

def max_error(x_vals, y_vals, spline_f, m):
    x_test = np.linspace(x_vals[0], x_vals[-1], m)
    y_true = np.array([f(x) for x in x_test])
    y_spline = np.array([spline_f(x_vals, y_vals, x) for x in x_test])
    return np.max(np.abs(y_true - y_spline))

def generate_deviation_table():
    a, b = -2, 2
    m = 1000  # Количество проверочных точек
    n_values = [5, 10, 30, 50, 100]

    print("\nТаблица максимальных отклонений:")
    print("| Количество узлов | Проверочные точки | Линейный сплайн | Квадратичный сплайн | Кубический сплайн |")
    print("|------------------|-------------------|------------------|----------------------|-------------------|")

    for n in n_values:
        x_vals = get_nodes(a, b, n)
        y_vals = [f(x) for x in x_vals]

        e_linear = max_error(x_vals, y_vals, linear_spline, m)
        e_quadratic = max_error(x_vals, y_vals, quadratic_spline, m)
        e_cubic = max_error(x_vals, y_vals, cubic_spline, m)

        print(f"| {n:16} | {m:17} | {e_linear:16.6f} | {e_quadratic:20.6f} | {e_cubic:17.6f} |")

generate_deviation_table()

a, b = -2, 2
n = int(input("\nВведите количество узлов n = "))
m = int(input("Введите количество точек для оценки отклонения m = "))

x_vals = get_nodes(a, b, n)
y_vals = [f(x) for x in x_vals]
x_fine = np.linspace(a, b, m)
y_true = [f(x) for x in x_fine]

y_linear = [linear_spline(x_vals, y_vals, x) for x in x_fine]
y_quadratic = [quadratic_spline(x_vals, y_vals, x) for x in x_fine]
y_cubic = [cubic_spline(x_vals, y_vals, x) for x in x_fine]

plt.figure(figsize=(10, 6))
plt.plot(x_fine, y_true, label="Исходная функция", color="black", linestyle="dashed")
plt.plot(x_fine, y_linear, label="Линейный сплайн", color="red")
plt.plot(x_fine, y_quadratic, label="Квадратичный сплайн", color="green")
plt.plot(x_fine, y_cubic, label="Кубический сплайн", color="blue")
plt.scatter(x_vals, y_vals, color="black", marker="o", label="Узлы интерполяции")

plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"Интерполяционные сплайны ({n} узлов)")
plt.grid(True)
plt.show()

# Абсолютная погрешность
error_linear = np.abs(np.array(y_true) - np.array(y_linear))
error_quadratic = np.abs(np.array(y_true) - np.array(y_quadratic))
error_cubic = np.abs(np.array(y_true) - np.array(y_cubic))

plt.figure(figsize=(10, 6))
plt.plot(x_fine, error_linear, label="Ошибка линейного сплайна", color="red")
plt.plot(x_fine, error_quadratic, label="Ошибка квадратичного сплайна", color="green")
plt.plot(x_fine, error_cubic, label="Ошибка кубического сплайна", color="blue")
plt.xlabel("x")
plt.ylabel("Абсолютная погрешность")
plt.title(f"График абсолютной погрешности ({n} узлов)")
plt.legend()
plt.grid(True)
plt.show()

