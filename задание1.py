import math
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return math.tan(0.5 * x + 0.2) - x ** 2

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

def divdif(x, y):
    n = len(x)
    coef = np.copy(y).astype(float)
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            coef[i] = (coef[i] - coef[i - 1]) / (x[i] - x[i - j])
    return coef

def Nn(x, y, X):
    coef = divdif(x, y)
    n = len(x)
    result = coef[-1]
    for j in range(n - 2, -1, -1):
        result = result * (X - x[j]) + coef[j]
    return result

def RLn(x, y, m, a, b):
    t = np.linspace(a, b, m)
    return max(abs(f(ti) - Ln(x, y, ti)) for ti in t)

def RNn(x, y, m, a, b):
    t = np.linspace(a, b, m)
    return max(abs(f(ti) - Nn(x, y, ti)) for ti in t)

def opt(a, b, n):
    return [(a + b) / 2 + (b - a) / 2 * math.cos((2 * k + 1) * math.pi / (2 * (n + 1))) for k in range(n + 1)]

def generate_deviation_table():
    a, b = -2, 2
    m = 1000
    n_values = [5, 10, 30, 50, 100]

    print("\nТаблица максимальных отклонений:")
    print(
        "| Количество узлов | Проверочные точки | Лагранж (равн.) | Лагранж (опт.) | Ньютон (равн.) | Ньютон (опт.) |")
    print(
        "|------------------|-------------------|------------------|-----------------|-----------------|---------------|")

    for n in n_values:
        x_ravn = np.linspace(a, b, n + 1)
        y_ravn = [f(xi) for xi in x_ravn]

        x_opt = opt(a, b, n)
        y_opt = [f(xi) for xi in x_opt]

        dif_L_ravn = RLn(x_ravn, y_ravn, m, a, b)
        dif_L_opt = RLn(x_opt, y_opt, m, a, b)
        dif_N_ravn = RNn(x_ravn, y_ravn, m, a, b)
        dif_N_opt = RNn(x_opt, y_opt, m, a, b)

        print(f"| {n:16} | {m:17} | {dif_L_ravn:16.6f} | {dif_L_opt:15.6f} | {dif_N_ravn:15.6f} | {dif_N_opt:13.6f} |")

generate_deviation_table()

a, b = -2, 2
n = int(input("\nВведите количество узлов n = ")) - 1
m = int(input("Введите количество точек для оценки отклонения m = "))

x_ravn = np.linspace(a, b, n + 1)
y_ravn = [f(xi) for xi in x_ravn]

x_opt = opt(a, b, n)
y_opt = [f(xi) for xi in x_opt]

t = np.linspace(a, b, 500)
y_true = [f(ti) for ti in t]

y_L_ravn = [Ln(x_ravn, y_ravn, ti) for ti in t]
y_N_ravn = [Nn(x_ravn, y_ravn, ti) for ti in t]

y_L_opt = [Ln(x_opt, y_opt, ti) for ti in t]
y_N_opt = [Nn(x_opt, y_opt, ti) for ti in t]

plt.figure(figsize=(12, 6))

# График Лагранжа (равноотстоящие узлы)
plt.subplot(2, 2, 1)
plt.plot(t, y_true, label="f(x)", color="black")
plt.plot(t, y_L_ravn, '--', label="Лагранж (равноотстоящие)")
plt.scatter(x_ravn, y_ravn, color='red', label="Узлы интерполяции")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Лагранж: равноотстоящие узлы")
plt.legend()
plt.grid()

# График Лагранжа (оптимальные узлы)
plt.subplot(2, 2, 2)
plt.plot(t, y_true, label="f(x)", color="black")
plt.plot(t, y_L_opt, '--', label="Лагранж (оптимальные)")
plt.scatter(x_opt, y_opt, color='blue', label="оптимальные узлы")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Лагранж: оптимальные узлы")
plt.legend()
plt.grid()

# График Ньютона (равноотстоящие узлы)
plt.subplot(2, 2, 3)
plt.plot(t, y_true, label="f(x)", color="black")
plt.plot(t, y_N_ravn, '--', label="Ньютон (равноотстоящие)")
plt.scatter(x_ravn, y_ravn, color='red', label="Узлы интерполяции")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Ньютон: равноотстоящие узлы")
plt.legend()
plt.grid()

# График Ньютона (оптимальные узлы)
plt.subplot(2, 2, 4)
plt.plot(t, y_true, label="f(x)", color="black")
plt.plot(t, y_N_opt, '--', label="Ньютон (оптимальные)")
plt.scatter(x_opt, y_opt, color='blue', label="оптимальные узлы")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Ньютон: оптимальные узлы")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# Абсолютные погрешности
error_L_ravn = np.abs(np.array(y_true) - np.array(y_L_ravn))
error_L_opt = np.abs(np.array(y_true) - np.array(y_L_opt))
error_N_ravn = np.abs(np.array(y_true) - np.array(y_N_ravn))
error_N_opt = np.abs(np.array(y_true) - np.array(y_N_opt))

plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.plot(t, error_L_ravn, color="red", label="Ошибка Лагранжа (равн.)")
plt.title("Абсолютная погрешность: Лагранж (равн.)")
plt.xlabel("x")
plt.ylabel("|f(x) - L(x)|")
plt.grid()
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(t, error_L_opt, color="blue", label="Ошибка Лагранжа (опт.)")
plt.title("Абсолютная погрешность: Лагранж (опт.)")
plt.xlabel("x")
plt.ylabel("|f(x) - L(x)|")
plt.grid()
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(t, error_N_ravn, color="green", label="Ошибка Ньютона (равн.)")
plt.title("Абсолютная погрешность: Ньютон (равн.)")
plt.xlabel("x")
plt.ylabel("|f(x) - N(x)|")
plt.grid()
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(t, error_N_opt, color="purple", label="Ошибка Ньютона (опт.)")
plt.title("Абсолютная погрешность: Ньютон (опт.)")
plt.xlabel("x")
plt.ylabel("|f(x) - N(x)|")
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()


