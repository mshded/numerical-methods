import numpy as np
import matplotlib.pyplot as plt

A = 1/12
B = 1/20
xi = 1/12
c2 = xi
a21 = c2
b2 = 1/(2*c2)
b1 = 1 - b2

f_call_counter = 0

def f(x, y):
    global f_call_counter
    f_call_counter += 1
    y1, y2 = y
    return np.array([A * y2, -B * y1])


#  Аналитическое решение
omega = np.sqrt(A * B)

def y_true(x):
    C = B * np.pi
    D = (A**2 * np.pi) / omega
    xs = np.atleast_1d(x)
    y1 = C * np.cos(omega * xs) + D * np.sin(omega * xs)
    y1p = -C * omega * np.sin(omega * xs) + D * omega * np.cos(omega * xs)
    y2 = y1p / A
    return np.vstack([y1, y2]).T

def rk2_step(x, y, h):
    k1 = f(x, y)
    k2 = f(x + c2 * h, y + a21 * h * k1)
    return y + h * (b1 * k1 + b2 * k2)

def rk4_step(x, y, h):
    k1 = f(x, y)
    k2 = f(x + h/2, y + h/2 * k1)
    k3 = f(x + h/2, y + h/2 * k2)
    k4 = f(x + h, y + h * k3)
    return y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)

#  Встроенные шаги
def rk4_step_embedded(x, y, h):
    h2 = h / 2
    y_bar = rk4_step(x, y, h)
    y_mid = rk4_step(x, y, h2)
    y_tilde = rk4_step(x + h2, y_mid, h2)
    return y_bar, y_tilde

def rk2_step_embedded(x, y, h):
    h2 = h / 2
    y_bar = rk2_step(x, y, h)
    y_mid = rk2_step(x, y, h2)
    y_tilde = rk2_step(x + h2, y_mid, h2)
    return y_bar, y_tilde

def initial_step_general(x0, y0, x_end, eps, s, method_euler=False):
    f0 = f(x0, y0)
    denom = max(abs(x0), abs(x_end), 1e-12)
    delta = (1 / denom) ** (s + 1) + np.linalg.norm(f0) ** (s + 1)
    h = (eps / delta) ** (1 / (s + 1))

    if method_euler or np.allclose(f0, 0.0, atol=1e-12):
        y_e = y0 + h * f0
        x_new = x0 + h
        f1 = f(x_new, y_e)
        denom2 = max(abs(x_new), abs(x_end), 1e-12)
        delta2 = (1 / denom2) ** (s + 1) + np.linalg.norm(f1) ** (s + 1)
        h2 = (eps / delta2) ** (1 / (s + 1))
        h = min(h, h2)

    return min(h, x_end - x0)

#  Интегрирование с постоянным шагом
def integrate_const(x0, y0, x_end, h, step_func):
    x = x0
    y = y0.copy()
    while x < x_end - 1e-14:
        step = min(h, x_end - x)
        y = step_func(x, y, step)
        x += step
    return y

def h_runge(x0, y0, x_target, eps, h_initial, s, step_func, max_iter=30):
    global f_call_counter
    h = h_initial
    for i in range(1, max_iter + 1):

        f_call_counter = 0
        y_h = integrate_const(x0, y0, x_target, h, step_func)
        calls_h = f_call_counter

        f_call_counter = 0
        y_h2 = integrate_const(x0, y0, x_target, h/2, step_func)
        calls_h2 = f_call_counter

        est_vec = (y_h2 - y_h) / (1 - 2**(-s))
        err = np.max(np.abs(est_vec))
        n_approx = int(np.ceil((x_target - x0) / h))

        print(f"iter {i}: h = {h:.6e}, steps ≈ {n_approx}, error = {err:.2e}, calls f = {calls_h + calls_h2}")

        if err <= eps:
            return h, y_h2, est_vec

        h /= 2

#  Интегрирование с адаптивным шагом
def adaptive_integrate(step_func_embedded, s, x0, y0, x_end, eps, pr, max_steps=10000):
    x = x0
    y = y0.copy()
    h = initial_step_general(x0, y0, x_end, eps, s=s, method_euler=True)

    xs = [x]
    hs = []
    rhos = []
    true_over_est = []

    f_calls = 0
    steps = 0

    while x < x_end - 1e-14 and steps < max_steps:

        if x + h > x_end:
            h = x_end - x

        global f_call_counter
        before = f_call_counter

        y_bar, y_tilde = step_func_embedded(x, y, h)

        after = f_call_counter
        f_calls += (after - before)

        est_local = (y_tilde - y_bar) / (1 - 2**(-s))
        rho = np.max(np.abs(est_local))

        y_true_next = y_true(x + h)[0]
        true_local = np.max(np.abs(y_true_next - y_bar))

        est = np.max(np.abs(est_local))
        ratio = np.nan if est == 0 else true_local / est

        hs.append(h)
        rhos.append(rho)
        true_over_est.append(ratio)
        xs.append(x + h)

        eps_2s = eps * (2**s)
        eps_low = eps / (2**(s+1))

        if pr == 1:
            print(f"step {steps:1d}: x = {x:.6f}, h = {h:.6e}, rho = {rho:.2e}")

        if rho > eps_2s:
            h /= 2
            continue
        elif rho > eps:
            y = y_tilde
            x += h
            steps += 1
            h /= 2
        elif rho >= eps_low:
            y = y_bar
            x += h
            steps += 1
        else:
            y = y_bar
            x += h
            steps += 1
            h *= 2

    return np.array(xs), np.array(hs), np.array(rhos), np.array(true_over_est), f_calls

#  Интегрирование с сохранением точек
def integrate_with_points(x0, y0, x_end, h, step_func):
    xs = [x0]
    ys = [y0.copy()]
    x = x0
    y = y0.copy()

    while x < x_end - 1e-14:
        step = min(h, x_end - x)
        y = step_func(x, y, step)
        x += step
        xs.append(x)
        ys.append(y.copy())

    return np.array(xs), np.array(ys)

x0 = 0.0
x_end = np.pi
y0 = np.array([B * np.pi, A * np.pi])
eps = 1e-4

print("\n3.1. РК4 с постоянным шагом\n")

h0_rk4 = initial_step_general(x0, y0, x_end, eps, s=4, method_euler=True)
h_final_rk4, y_const_rk4, est_vec_rk4 = h_runge(x0, y0, x_end, eps, h0_rk4, s=4, step_func=rk4_step)
print(f"Начальный шаг h0 = {h0_rk4:.6e}")
print(f"Итоговый шаг для РК4: {h_final_rk4:.6e}")
print(f"Максимальная оценка погрешности: {np.max(np.abs(est_vec_rk4)):.2e}")
print(f"Решение: {y_const_rk4}")

print("\n3.1. РК4 с автоматическим шагом\n")

print(f"Начальный шаг h0 = {h0_rk4:.6e}")
xs_adapt_rk4, hs_adapt_rk4, rhos_adapt_rk4, ratio_adapt_rk4, f_calls_rk4 = adaptive_integrate(rk4_step_embedded, 4, x0, y0, x_end, eps, pr = 1)
steps_rk4 = len(hs_adapt_rk4)
y_adapt_rk4 = y_true(xs_adapt_rk4[-1])[0]
print(f"Число шагов: {steps_rk4}")
print(f"Число вызовов: {f_calls_rk4}")
print(f"Решение: {y_adapt_rk4}")

print("\n3.2. Определение шага ЯМРК2\n")

h0_rk2 = initial_step_general(x0, y0, x_end, eps, s=2, method_euler=True)
print(f"Начальный шаг h0 = {h0_rk2:.6e}")
h_final_rk2, y_const_rk2, est_vec_rk2 = h_runge(x0, y0, x_end, eps, h0_rk2, s=2, step_func=rk2_step)
print(f"Итоговый шаг РК2: {h_final_rk2:.6e}")
print(f"Максимальная оценка погрешности: {np.max(np.abs(est_vec_rk2)):.2e}")

xs_rk4, ys_rk4 = integrate_with_points(x0, y0, x_end, h_final_rk4, rk4_step)
xs_rk2, ys_rk2 = integrate_with_points(x0, y0, x_end, h_final_rk2, rk2_step)

err_rk4 = np.max(np.abs(ys_rk4 - y_true(xs_rk4)), axis=1)
err_rk2 = np.max(np.abs(ys_rk2 - y_true(xs_rk2)), axis=1)

plt.figure()
plt.plot(xs_rk4, err_rk4)
plt.xlabel("x")
plt.ylabel("истинная полная погрешность")
plt.title("3.2: Истинная глобальная погрешность — RK4")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(xs_rk2, err_rk2)
plt.xlabel("x")
plt.ylabel("истинная полная погрешность")
plt.title("3.2: Истинная глобальная погрешность — RK2")
plt.grid(True)
plt.show()

xs_rk2_adapt, hs_rk2_adapt, rhos_rk2_adapt, ratio_rk2_adapt, fcalls_rk2 = adaptive_integrate(rk2_step_embedded, 2, x0, y0, x_end, eps, pr = 0)
xs_rk4_adapt, hs_rk4_adapt, rhos_rk4_adapt, ratio_rk4_adapt, fcalls_rk4 = adaptive_integrate(rk4_step_embedded, 4, x0, y0, x_end, eps, pr = 0)

# 3.3.1 h(x)
plt.figure()
plt.plot(xs_rk4_adapt[:-1], hs_rk4_adapt)
plt.xlabel("x")
plt.ylabel("h")
plt.title("3.3.1: Адаптивный шаг h(x) — RK4")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(xs_rk2_adapt[:-1], hs_rk2_adapt)
plt.xlabel("x")
plt.ylabel("h")
plt.title("3.3.1: Адаптивный шаг h(x) — RK2")
plt.grid(True)
plt.show()

# 3.3.2 Отношение истинной к оценённой локальной ошибки
plt.figure()
plt.plot(xs_rk4_adapt[:-1], ratio_rk4_adapt)
plt.xlabel("x")
plt.ylabel("Отношение true / est")
plt.title("3.3.2: RK4 отношение истинной к оценённой локальной ошибке")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(xs_rk2_adapt[:-1], ratio_rk2_adapt)
plt.xlabel("x")
plt.ylabel("Отношение true / est")
plt.title("3.3.2: RK2 отношение истинной к оценённой локальной ошибке")
plt.grid(True)
plt.show()

# 3.3.3 Зависимость числа вызовов f от eps
eps_list = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
fcalls_rk4_list = []
fcalls_rk2_list = []

for eps in eps_list:
    _, _, _, _, fc4 = adaptive_integrate(rk4_step_embedded, 4, x0, y0, x_end, eps, pr = 0)
    fcalls_rk4_list.append(fc4)
    _, _, _, _, fc2 = adaptive_integrate(rk2_step_embedded, 2, x0, y0, x_end, eps, pr = 0)
    fcalls_rk2_list.append(fc2)

plt.figure()
plt.loglog(eps_list, fcalls_rk4_list, marker='o')
plt.xlabel("eps")
plt.ylabel("число вызовов f")
plt.title("3.3.3: RK4 — вызовы f vs eps")
plt.grid(True, which="both")
plt.show()

plt.figure()
plt.loglog(eps_list, fcalls_rk2_list, marker='o')
plt.xlabel("eps")
plt.ylabel("число вызовов f")
plt.title("3.3.3: RK2 — вызовы f vs eps")
plt.grid(True, which="both")
plt.show()
