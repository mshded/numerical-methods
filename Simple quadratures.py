import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve
from scipy.integrate import quad
import warnings

def f(x):
    return 3.7 * np.cos(1.5 * x) * np.exp(-4 * x / 3) + 2.4 * np.sin(4.5 * x) * np.exp(2 * x / 3) + 4

# Часть 1.1
def left_rectangle(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    return h * np.sum(f(x[:-1]))

def middle_rectangle(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a + h / 2, b - h / 2, n)
    return h * np.sum(f(x))

def trapezoidal(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    return h * (0.5 * f(a) + 0.5 * f(b) + np.sum(f(x[1:-1])))

def simpson(f, a, b, n):
    if n % 2 != 0:
        n += 1
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    c = np.ones(n + 1)
    c[1:n:2] = 4
    c[2:n-1:2] = 2
    return (h / 3) * np.sum(c * f(x))

# Часть 1.2:
def p(x, a, b, alpha, beta):
    x = np.asarray(x)
    r = np.ones_like(x, float)
    if alpha != 0:
        m = (x - a) > 1e-14
        r[m] *= (x[m] - a) ** (-alpha)
        r[~m] = 0.0
    if beta != 0:
        m = (b - x) > 1e-14
        r[m] *= (b - x[m]) ** (-beta)
        r[~m] = 0.0
    return r

def F(x, a, b, alpha, beta):
    return p(x, a, b, alpha, beta) * f(x)

def x_of_t(t, a, b):
    return 0.5*(b - a)*t + 0.5*(a + b)

def solve_cubic_cardano(a, b, c, d):
    if abs(a) < 1e-18:
        return np.roots([b, c, d])
    A = b / a
    B = c / a
    C = d / a
    Q = (3*B - A**2) / 9.0
    R = (9*A*B - 27*C - 2*A**3) / 54.0
    D = Q**3 + R**2
    if D >= 0:
        S = np.cbrt(R + np.sqrt(D))
        T = np.cbrt(R - np.sqrt(D))
        t1 = -A/3 + (S + T)
        return np.array([t1])
    else:
        theta = np.arccos(R / np.sqrt(-Q**3))
        r = 2 * np.sqrt(-Q)
        t1 = -A/3 + r * np.cos(theta/3)
        t2 = -A/3 + r * np.cos((theta + 2*np.pi)/3)
        t3 = -A/3 + r * np.cos((theta + 4*np.pi)/3)
        return np.array([t1, t2, t3])

def moments_gauss(a_i, b_i, a, b, alpha, beta, max_j=5):
    jac = 0.5 * (b_i - a_i)
    moments = np.zeros(max_j + 1)
    for j in range(max_j + 1):
        integrand = lambda t: ((x_of_t(t, a_i, b_i) - a) ** (-alpha) *
                               (b - x_of_t(t, a_i, b_i)) ** (-beta) *
                               (t ** j) * jac)
        moments[j], _ = quad(integrand, -1, 1, epsabs=1e-12, epsrel=1e-12, limit=200)
    return moments

def gauss_3point_single(a_i, b_i, a, b, alpha, beta):
    moments = moments_gauss(a_i, b_i, a, b, alpha, beta)

    # Кубический многочлен для узлов
    try:
        A = np.array([[moments[0], moments[1], moments[2]],
                      [moments[1], moments[2], moments[3]],
                      [moments[2], moments[3], moments[4]]])
        bvec = -np.array([moments[3], moments[4], moments[5]])
        coeffs = solve(A, bvec)
        poly = np.array([1.0, coeffs[2], coeffs[1], coeffs[0]])
        nodes_t = np.sort(np.real(solve_cubic_cardano(*poly)))
    except:
        nodes_t = np.array([-np.sqrt(3 / 5), 0.0, np.sqrt(3 / 5)])

    # Ограничиваем узлы в [-1, 1] для стабильности
    nodes_t = np.clip(nodes_t, -1.0, 1.0)

    # Веса через система Вандермонда
    V = np.vander(nodes_t, N=3, increasing=True).T
    try:
        w = solve(V, moments[:3])
    except:
        V_fallback = np.vander(nodes_t, N=3, increasing=True)
        w = solve(V_fallback.T, moments[:3])

    # Преобразуем узлы в [a_i, b_i] и считаем сумму
    I = 0.0
    for i in range(3):
        xi = x_of_t(nodes_t[i], a_i, b_i)
        I += w[i] * f(xi)
    return I

def gauss_3point_composite(a, b, alpha, beta, n_segments):
    h = (b - a) / n_segments
    I = 0.0
    for i in range(n_segments):
        a_i = a + i * h
        b_i = a_i + h
        I += gauss_3point_single(a_i, b_i, a, b, alpha, beta)
    return I

def moments_newton_cotes(a_i, b_i, a, b, alpha, beta):
    m_values = np.zeros(3)

    for j in range(3):
        integrand = lambda x: (x - a) ** (-alpha) * (b - x) ** (-beta) * x ** j
        m_values[j], _ = quad(integrand, a_i, b_i, limit=200, epsabs=1e-12, epsrel=1e-12)

    return m_values

def coefs_newton_cotes(a_i, b_i, a, b, alpha, beta):
    A = [0, 0, 0]
    m_values = moments_newton_cotes(a_i, b_i, a, b, alpha, beta)

    z_i = (a_i + b_i) / 2.0
    x0, x1, x2 = a_i, z_i, b_i

    # Решаем систему для нахождения коэффициентов
    A_matrix = np.array([
        [1, 1, 1],
        [x0, x1, x2],
        [x0**2, x1**2, x2 ** 2]
    ])

    b_vector = np.array([m_values[0], m_values[1], m_values[2]])

    A_coefs = np.linalg.solve(A_matrix, b_vector)
    return A_coefs

def newton_cotes_single(a_i, b_i):
    coefs = coefs_newton_cotes(a_i, b_i, a, b, alpha, beta)
    x_nodes = np.array([a_i, (a_i+b_i)/2, b_i])
    return sum(coefs[i] * f(x_nodes[i]) for i in range(3))

def newton_cotes_composite(a, b, alpha, beta, n_segments):
    h = (b - a) / n_segments
    I = 0.0
    for i in range(n_segments):
        a_i = a + i*h
        b_i = a_i + h
        I += newton_cotes_single(a_i, b_i)
    return I

def true_integral(a, b, alpha, beta):
    integrand = lambda x: F(x, a, b, alpha, beta)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            I_true, _ = quad(integrand, a, b, epsabs=1e-12, epsrel=1e-12, limit=1000)
        return I_true
    except Exception:
        x = np.linspace(a + 1e-12, b - 1e-12, 400001)
        y = integrand(x)
        return np.trapezoid(y, x)

def plot_errors_part1_1(a, b):
    methods = [
        ('Левые прямоугольники', left_rectangle),
        ('Средние прямоугольники', middle_rectangle),
        ('Трапеции', trapezoidal),
        ('Симпсон', simpson)
    ]
    n_values = np.array([4, 8, 16, 32, 64, 128, 256, 512])
    I_true = simpson(f, a, b, 4096)
    plt.figure(figsize=(10, 6))
    for name, method in methods:
        errors = [abs(method(f, a, b, n) - I_true) for n in n_values]
        plt.loglog(n_values, errors, 'o-', label=name, markersize=6)
    plt.xlabel('n')
    plt.ylabel('абсолютная погрешность')
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)
    plt.show()

def plot_errors_part1_2(a, b, alpha, beta):
    methods = [
        ('Ньютон-Котес (3 точки)', lambda n: newton_cotes_composite(a, b, alpha, beta, n)),
        ('Гаусс (3 точки)', lambda n: gauss_3point_composite(a, b, alpha, beta, n))
    ]
    n_values = np.array([3, 6, 12, 24, 48, 96])
    I_true = true_integral(a, b, alpha, beta)
    plt.figure(figsize=(10, 6))
    for name, method in methods:
        errors = [abs(method(n) - I_true) for n in n_values]
        plt.loglog(n_values, errors, 'o-', label=name, markersize=6)
    plt.xlabel('число сегментов')
    plt.ylabel('абсолютная погрешность')
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)
    plt.show()


a, b = 1.8, 2.3
alpha, beta = 0, 3/5

print("Часть 1.1:")
n_test = 8
print(f"Левые прямоугольники: {left_rectangle(f, a, b, n_test):.6f}")
print(f"Средние прямоугольники: {middle_rectangle(f, a, b, n_test):.6f}")
print(f"Трапеции: {trapezoidal(f, a, b, n_test):.6f}")
print(f"Симпсон: {simpson(f, a, b, n_test):.6f}")

print("\nЧасть 1.2:")
n_test = 6
print(f"Ньютон-Котес: {newton_cotes_composite(a, b, alpha, beta, n_test):.6f}")
print(f"Гаусс: {gauss_3point_composite(a, b, alpha, beta, n_test):.6f}")

plot_errors_part1_1(a, b)
plot_errors_part1_2(a, b, alpha, beta)
