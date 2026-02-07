import math
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return math.tan(0.5 * x + 0.2) - x ** 2

def Ln(x,y,X):
    sum = 0
    for i in range(len(y)):
        w = 1
        wk = 1
        for k in range(len(y)):
            if k!=i:
                w *= (X-x[k])
                wk *= (x[i]-x[k])
        sum += y[i]*w/wk
    return sum

def divdif(x,y):
    n = len(x)
    coef = np.copy(y).astype(float)
    for j in range(1, n):
        for i in range(n-1, j-1, -1):
            coef[i] = (coef[i]-coef[i-1])/(x[i]-x[i-j])
    return coef

def Nn(x,y,X):
    coef = divdif(x,y)
    n = len(x)
    result = coef[-1]
    for j in range(n-2,-1,-1):
        result = result*(X-x[j])+coef[j]
    return result

def RLn(x,y,m,a,b):
    t = np.linspace(a,b,m)
    max_d = 0
    for ti in t:
        max_d = max(max_d,abs(f(ti)-Ln(x,y,ti)))
    return max_d

def RNn(x,y,m,a,b):
    t = np.linspace(a,b,m)
    max_d = 0
    for ti in t:
        max_d = max(max_d, abs(f(ti)-Nn(x,y,ti)))
    return max_d

# Линейный сплайн
def linear_spline(x, y, X):
    for i in range(1, len(x)):
        if X <= x[i]:
            return y[i - 1] + (y[i] - y[i - 1]) * (X - x[i - 1]) / (x[i] - x[i - 1])
    return y[-1]

# Квадратичный сплайн (используем более простой способ)
def quadratic_spline(x, y, X):
    n = len(x)
    for i in range(1, n):
        if x[i-1] <= X <= x[i]:
            h = x[i] - x[i-1]
            a = (y[i] - y[i-1]) / h
            return y[i-1] + a * (X - x[i-1]) + (y[i] - y[i-1]) * (X - x[i-1]) * (X - x[i]) / (2 * h)
    return y[-1]

# Кубический сплайн (без использования scipy)
def cubic_spline(x, y,X):
    n = len(x)
    h = np.diff(x)
    alpha = [0] * n
    l = [1] * n
    mu = [0] * n
    z = [0] * n
    c = [0] * n
    b = [0] * n
    d = [0] * n

    # Составление системы
    for i in range(1, n - 1):
        alpha[i] = (3 / h[i]) * (y[i + 1] - y[i]) - (3 / h[i - 1]) * (y[i] - y[i - 1])

    l[0] = 1
    z[0] = 0
    for i in range(1, n - 1):
        l[i] = 2 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]

    l[n - 1] = 1
    z[n - 1] = 0
    c[n - 1] = 0

    # Обратный ход для получения коэффициентов b, c, d
    for j in range(n - 2, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]
        b[j] = (y[j + 1] - y[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3
        d[j] = (c[j + 1] - c[j]) / (3 * h[j])

    a = y[:-1]

    return lambda X: a[int(X)] + b[int(X)] * (X - x[int(X)]) + c[int(X)] * (X - x[int(X)])**2 + d[int(X)] * (X - x[int(X)])**3


# Функция для вычисления максимального отклонения
def max_deviation(sp, x, y, m, a, b):
    t = np.linspace(a, b, m)
    max_d = 0
    for ti in t:
        max_d = max(max_d, abs(f(ti) - sp(x, y, ti)))
    return round(max_d, 4)

# Заполнение узлов интерполяции
a = -2
b = 2
n = int(input('Введите количество узлов n = ')) - 1
m = int(input('Введите количество точек для оценки отклонения m = '))
r = abs(b - a) / n

x = [round(a + i * r, 4) for i in range(n + 1)]
y = [round(f(xi), 4) for xi in x]

# Ввод значения для интерполяции
X = float(input('Введите значение, для которого нужно найти интерполяцию X = '))

# Вывод значений полиномов Лагранжа и Ньютона
print(f'Лагранж: {Ln(x, y, X)}, Ньютон: {Nn(x, y, X)}')

# Оценка отклонений
print(f'Максимальное отклонение для линейного сплайна: {max_deviation(linear_spline, x, y, m, a, b)}')
print(f'Максимальное отклонение для квадратичного сплайна: {max_deviation(quadratic_spline, x, y, m, a, b)}')
print(f'Максимальное отклонение для кубического сплайна: {max_deviation(cubic_spline(x,y,X), x, y, m, a, b)}')

# Графики
t = np.linspace(a, b, 500)
y_true = [f(ti) for ti in t]
y_L = [Ln(x, y, ti) for ti in t]
y_N = [Nn(x, y, ti) for ti in t]
y_linear = [linear_spline(x, y, ti) for ti in t]
y_quadratic = [quadratic_spline(x, y, ti) for ti in t]
y_cubic = [cubic_spline(x, y, ti) for ti in t]

plt.figure(figsize=(10, 6))
plt.plot(t, y_true, label="f(x)", color="black")
plt.plot(t, y_L, '--', label="Полином Лагранжа")
plt.plot(t, y_N, ':', label="Полином Ньютона")
plt.plot(t, y_linear, label="Линейный сплайн")
plt.plot(t, y_quadratic, label="Квадратичный сплайн")
plt.plot(t, y_cubic, label="Кубический сплайн")
plt.scatter(x, y, color='red', label="Узлы интерполяции")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Интерполяция функции f(x) сплайнами и полиномами")
plt.grid()
plt.show()