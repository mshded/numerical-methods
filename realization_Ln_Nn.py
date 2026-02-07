import math
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return math.tan(0.5*x+0.2)-x**2

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

a = -2
b = 2
n = int(input('Введите количество узлов n = '))-1
m = int(input('Введите количество точек для оценки отклонения m = '))
r = abs(b-a)/n

x = [round(a+i*r, 4) for i in range(n+1)]
y = [round(f(xi), 4) for xi in x]

for i in range(n+1):
    print(f'x[{i}] = {x[i]}')
    print(f'y[{i}] = {y[i]}')

X = float(input('Введите значение, для которого нужно найти интерполяцию X = '))
print(f'Лагранж: {Ln(x,y,X)}, Ньютон: {Nn(x,y,X)}')
print(f'Максимальное отклонение: Лагранж = {RLn(x,y,m,a,b)}, Ньютон = {RNn(x,y,m,a,b)}')

t = np.linspace(a, b, 500)
y_true = [f(ti) for ti in t]
y_L = [Ln(x,y,ti) for ti in t]
y_N = [Nn(x,y,ti) for ti in t]

plt.figure(figsize=(10, 6))
plt.plot(t, y_true, label="f(x)", color="black")
plt.plot(t, y_L, '--', label="Полином Лагранжа")
plt.plot(t, y_N, ':', label="Полином Ньютона")
plt.scatter(x, y, color='red', label="Узлы интерполяции")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Интерполяция функции f(x) полиномами Лагранжа и Ньютона")
plt.grid()
plt.show()

