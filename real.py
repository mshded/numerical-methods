import math
import numpy as np
import matplotlib.pyplot as plt

# Исходная функция
def f(x):
    return math.tan(0.5 * x + 0.2) - x ** 2

# Полином Лагранжа
def Ln(x, y, X):
    sum = 0
    for i in range(len(y)):
        w = 1
        wk = 1
        for k in range(len(y)):
            if k != i:
                w *= (X - x[k])
                wk *= (x[i] - x[k])
        sum += y[i] * w / wk
    return sum

# Разделенные разности для полинома Ньютона
def divdif(x, y):
    n = len(x)
    coef = np.copy(y).astype(float)
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            coef[i] = (coef[i] - coef[i - 1]) / (x[i] - x[i - j])
    return coef

# Полином Ньютона
def Nn(x, y, X):
    coef = divdif(x, y)
    n = len(x)
    result = coef[-1]
    for j in range(n - 2, -1, -1):
        result = result * (X - x[j]) + coef[j]
    return result

# Оптимальные узлы (Чебышевские узлы)
def chebyshev_nodes(a, b, n):
    return [(a + b) / 2 + (b - a) / 2 * math.cos((2 * k + 1) * math.pi / (2 * (n + 1))) for k in range(n + 1)]

# Ввод данных
a, b = -2, 2
n = int(input("Введите количество узлов n = ")) - 1

# Узлы интерполяции (равноотстоящие)
x_uniform = np.linspace(a, b, n + 1)
y_uniform = [f(xi) for xi in x_uniform]

# Узлы интерполяции (Чебышевские)
x_chebyshev = chebyshev_nodes(a, b, n)
y_chebyshev = [f(xi) for xi in x_chebyshev]

# Графики
t = np.linspace(a, b, 500)
y_true = [f(ti) for ti in t]

# Равноотстоящие узлы
y_L_uniform = [Ln(x_uniform, y_uniform, ti) for ti in t]
y_N_uniform = [Nn(x_uniform, y_uniform, ti) for ti in t]

# Чебышевские узлы
y_L_chebyshev = [Ln(x_chebyshev, y_chebyshev, ti) for ti in t]
y_N_chebyshev = [Nn(x_chebyshev, y_chebyshev, ti) for ti in t]

plt.figure(figsize=(12, 12))

# График Лагранжа (равноотстоящие узлы)
plt.subplot(2, 2, 1)
plt.plot(t, y_true, label="f(x)", color="black")
plt.plot(t, y_L_uniform, '--', label="Лагранж (равноотстоящие)")
plt.scatter(x_uniform, y_uniform, color='red', label="Узлы интерполяции")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Лагранж: равноотстоящие узлы")
plt.legend()
plt.grid()

# График Лагранжа (Чебышевские узлы)
plt.subplot(2, 2, 2)
plt.plot(t, y_true, label="f(x)", color="black")
plt.plot(t, y_L_chebyshev, '--', label="Лагранж (Чебышевские)")
plt.scatter(x_chebyshev, y_chebyshev, color='blue', label="Чебышевские узлы")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Лагранж: Чебышевские узлы")
plt.legend()
plt.grid()

# График Ньютона (равноотстоящие узлы)
plt.subplot(2, 2, 3)
plt.plot(t, y_true, label="f(x)", color="black")
plt.plot(t, y_N_uniform, '--', label="Ньютон (равноотстоящие)")
plt.scatter(x_uniform, y_uniform, color='red', label="Узлы интерполяции")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Ньютон: равноотстоящие узлы")
plt.legend()
plt.grid()

# График Ньютона (Чебышевские узлы)
plt.subplot(2, 2, 4)
plt.plot(t, y_true, label="f(x)", color="black")
plt.plot(t, y_N_chebyshev, '--', label="Ньютон (Чебышевские)")
plt.scatter(x_chebyshev, y_chebyshev, color='blue', label="Чебышевские узлы")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Ньютон: Чебышевские узлы")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()