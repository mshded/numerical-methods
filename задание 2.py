import numpy as np
from scipy.linalg import solve
from scipy.integrate import quad

def f(x):
    return (3.7 * np.cos(1.5 * x) * np.exp(-4 * x / 3)
            + 2.4 * np.sin(4.5 * x) * np.exp(2 * x / 3)
            + 4)

def p(x, a, b, alpha, beta):
    x = np.asarray(x)
    w = np.ones_like(x, float)

    if alpha != 0:
        mask = (x - a) > 1e-14
        w[mask] *= (x[mask] - a) ** (-alpha)
        w[~mask] = 0.0

    if beta != 0:
        mask = (b - x) > 1e-14
        w[mask] *= (b - x[mask]) ** (-beta)
        w[~mask] = 0.0

    return w

def F(x, a, b, alpha, beta):
    return p(x, a, b, alpha, beta) * f(x)

def x_of_t(t, ai, bi):
    return 0.5 * (bi - ai) * t + 0.5 * (ai + bi)

def solve_cubic_cardano(a3, a2, a1, a0):
    if abs(a3) < 1e-18:
        return np.roots([a2, a1, a0])

    A = a2 / a3
    B = a1 / a3
    C = a0 / a3

    Q = (3 * B - A**2) / 9
    R = (9 * A * B - 27 * C - 2 * A**3) / 54
    D = Q**3 + R**2

    if D >= 0:
        S = np.cbrt(R + np.sqrt(D))
        T = np.cbrt(R - np.sqrt(D))
        return np.array([-A / 3 + S + T])

    theta = np.arccos(R / np.sqrt(-Q**3))
    r = 2 * np.sqrt(-Q)
    return np.array([
        -A / 3 + r * np.cos(theta / 3),
        -A / 3 + r * np.cos((theta + 2*np.pi) / 3),
        -A / 3 + r * np.cos((theta + 4*np.pi) / 3)
    ])

def moments_gauss(ai, bi, a, b, alpha, beta, max_j=5):
    h = 0.5 * (bi - ai)
    moments = np.zeros(max_j + 1)

    for j in range(max_j + 1):
        def integrand(t):
            x = x_of_t(t, ai, bi)
            return (x - a)**(-alpha) * (b - x)**(-beta) * t**j * h

        moments[j], _ = quad(integrand, -1, 1, epsabs=1e-12, epsrel=1e-12, limit=200)

    return moments

def gauss_3point_single(ai, bi, a, b, alpha, beta):
    moments = moments_gauss(ai, bi, a, b, alpha, beta)

    # решаем систему для кубического многочлена
    try:
        A = np.array([
            [moments[0], moments[1], moments[2]],
            [moments[1], moments[2], moments[3]],
            [moments[2], moments[3], moments[4]]
        ])
        rhs = -np.array([moments[3], moments[4], moments[5]])

        c0, c1, c2 = solve(A, rhs)
        poly = np.array([1.0, c2, c1, c0])

        nodes_t = np.sort(np.real(solve_cubic_cardano(*poly)))
    except:
        nodes_t = np.array([-np.sqrt(3/5), 0.0, np.sqrt(3/5)])

    nodes_t = np.clip(nodes_t, -1, 1)

    # Веса
    V = np.vander(nodes_t, N=3, increasing=True).T
    w = solve(V, moments[:3])

    # Квадратура
    I = 0.0
    for t, weight in zip(nodes_t, w):
        I += weight * f(x_of_t(t, ai, bi))

    return I

def gauss_3point_composite(a, b, alpha, beta, n):
    h = (b - a) / n
    I = 0.0

    for k in range(n):
        ai = a + k * h
        bi = ai + h
        I += gauss_3point_single(ai, bi, a, b, alpha, beta)

    return I

def moments_newton_cotes(ai, bi, a, b, alpha, beta):
    res = np.zeros(3)
    for j in range(3):
        def integrand(x):
            return (x - a)**(-alpha) * (b - x)**(-beta) * x**j

        res[j], _ = quad(integrand, ai, bi, epsabs=1e-12, epsrel=1e-12, limit=200)

    return res

def coefs_newton_cotes(ai, bi, a, b, alpha, beta):
    m = moments_newton_cotes(ai, bi, a, b, alpha, beta)
    x0, x1, x2 = ai, 0.5*(ai + bi), bi

    A = np.array([[1, 1, 1],
                  [x0, x1, x2],
                  [x0**2, x1**2, x2**2]])

    return np.linalg.solve(A, m)

def newton_cotes_single(ai, bi, a, b, alpha, beta):
    c = coefs_newton_cotes(ai, bi, a, b, alpha, beta)
    x0, x1, x2 = ai, 0.5*(ai + bi), bi
    return c[0]*f(x0) + c[1]*f(x1) + c[2]*f(x2)

def newton_cotes_composite(a, b, alpha, beta, n):
    h = (b - a) / n
    I = 0.0

    for k in range(n):
        ai = a + k * h
        bi = ai + h
        I += newton_cotes_single(ai, bi, a, b, alpha, beta)

    return I

def richardson_method(S_list, h_list, m, n_terms=None):
    r = len(S_list)
    if n_terms is None:
        n_terms = r - 1

    A = np.ones((r, n_terms + 1))
    for i in range(r):
        for k in range(n_terms):
            A[i, k] = h_list[i] ** (m + k)

    sol = solve(A, np.array(S_list))
    J_refined = sol[-1]
    error = abs(J_refined - S_list[-1])

    return J_refined, error, sol[:-1]

def aitken_convergence(h1, h2, h3, S1, S2, S3):
    d1 = S2 - S1
    d2 = S3 - S2

    if abs(d1) < 1e-15 or abs(d2) < 1e-15:
        return -1

    ratio = abs(d1 / d2)
    if ratio <= 0 or ratio > 1e12:
        return -1

    L = h1 / h2
    if L <= 1:
        return -1

    return np.log(ratio) / np.log(L)

def adaptive_newton_cotes(a, b, alpha, beta, eps):
    print("Метод Ньютона-Котеса:")

    n = 4
    S_prev = newton_cotes_composite(a, b, alpha, beta, n)

    S_list = [S_prev]
    h_list = [(b - a) / n]

    while True:
        n *= 2
        S_curr = newton_cotes_composite(a, b, alpha, beta, n)
        h = (b - a) / n
        err = abs(S_curr - S_prev)

        print(f"n={n}, h={h:.6e}, S={S_curr:.12f}, ошибка={err:.2e}")

        S_list.append(S_curr)
        h_list.append(h)

        if err < eps:
            print(f"Точность достигнута при n={n}")
            break

        S_prev = S_curr

    if len(S_list) >= 3:
        m = aitken_convergence(h_list[-3], h_list[-2], h_list[-1],
                               S_list[-3], S_list[-2], S_list[-1])
        if m != -1:
            print(f"Порядок сходимости: m = {m:.4f}")

    return n, S_curr

def adaptive_gauss(a, b, alpha, beta, eps):
    print("Метод Гаусса:")

    n = 2
    S_prev = gauss_3point_composite(a, b, alpha, beta, n)

    S_list = [S_prev]
    h_list = [(b - a) / n]

    while True:
        n *= 2
        S_curr = gauss_3point_composite(a, b, alpha, beta, n)
        h = (b - a) / n
        err = abs(S_curr - S_prev)

        print(f"n={n}, h={h:.6e}, S={S_curr:.12f}, ошибка={err:.2e}")

        S_list.append(S_curr)
        h_list.append(h)

        if err < eps:
            print(f"Точность достигнута при n={n}")
            break

        S_prev = S_curr

    if len(S_list) >= 3:
        m = aitken_convergence(h_list[-3], h_list[-2], h_list[-1],
                               S_list[-3], S_list[-2], S_list[-1])
        if m != -1:
            print(f"Порядок сходимости: m = {m:.4f}")

    return n, S_curr

def optimal_step(a, b, alpha, beta, eps=1e-6):
    print("Оценка оптимального шага:")

    n_values = [4, 8, 16]
    S_list = []
    h_list = []

    for n in n_values:
        S = gauss_3point_composite(a, b, alpha, beta, n)
        h = (b - a) / n
        S_list.append(S)
        h_list.append(h)
        print(f"n={n}, h={h:.6e}, S={S:.12f}")

    m = aitken_convergence(h_list[0], h_list[1], h_list[2],
                           S_list[0], S_list[1], S_list[2])
    print(f"Порядок сходимости: m = {m:.4f}")

    J, err, _ = richardson_method(S_list, h_list, m)
    print(f"Уточненное значение: {J:.12f}")
    print(f"Оценка погрешности: {err:.2e}")

    h_curr = h_list[-1]
    h_opt = h_curr * (eps / err) ** (1 / m) if err > 1e-15 else h_curr / 10
    n_opt = max(2, int((b - a) / h_opt))
    h_opt_real = (b - a) / n_opt

    print(f"Оптимальный шаг: h = {h_opt_real:.6e} (n={n_opt})")

    return h_opt_real, n_opt, J, m


a = 1.8
b = 2.3
alpha, beta = 0, 3/5
epsilon = 1e-6

n_newton, S_newton = adaptive_newton_cotes(a, b, alpha, beta, epsilon)
n_gauss, S_gauss = adaptive_gauss(a, b, alpha, beta, epsilon)
h_opt, n_opt, J_refined, m_est = optimal_step(a, b, alpha, beta, epsilon)

h_gauss_final = (b - a) / n_gauss
h_newton_final = (b - a) / n_newton

print(f"h_opt = {h_opt:.3e}, n_opt = {n_opt}, "
      f"h_Gauss_final = {h_gauss_final:.3e}, "
      f"h_Newton_final = {h_newton_final:.3e}")
