import numpy as np

A = 1/12
B = 1/20
xi = 1/12
c2 = xi
a21 = c2
b2 = 1/(2*c2)
b1 = 1 - b2
s = 2

f_call_counter = 0

def f(x, y):
    global f_call_counter
    f_call_counter += 1
    y1, y2 = y
    return np.array([A * y2, -B * y1], dtype=float)

# 2-х этапный ЯМРК 2-го порядка
def rk2_step(x, y, h):
    k1 = f(x, y)
    k2 = f(x + c2 * h, y + a21 * h * k1)
    return y + h * (b1 * k1 + b2 * k2)

# Начальный шаг
def initial_step(x0, y0, xk, eps, s=2):

    f0 = f(x0, y0)
    denom = max(abs(x0), abs(xk), 1e-12)
    delta = (1/denom)**(s+1) + np.linalg.norm(f0)**(s+1)
    h0 = (eps/delta)**(1/(s+1))

    if np.allclose(f0, 0.0, atol=1e-12):
        y_e = y0 + h0 * f0
        f1 = f(x0 + h0, y_e)
        denom2 = max(abs(x0 + h), abs(xk), 1e-12)
        delta2 = (1/denom2)**(s+1) + np.linalg.norm(f1)**(s+1)
        h1 = (eps/delta2)**(1/(s+1))
        h0 = min(h0, h1)

    return min(h0, xk - x0)

# Интегрирование с постоянным шагом
def integrate_const(x0, y0, x_end, h):
    x = x0
    y = y0.copy()

    while x < x_end - 1e-14:
        step = min(h, x_end - x)
        y = rk2_step(x, y, step)
        x += step

    return y

# оценка глобальной погрешности по методу Рунге
def h_runge(x0, y0, x_target, eps, h_initial, s=2, max_iter=30):
    global f_call_counter

    h = h_initial
    for iter_count in range(1, max_iter + 1):
        f_call_counter = 0
        y_h = integrate_const(x0, y0, x_target, h)
        calls_h = f_call_counter

        f_call_counter = 0
        y_h2 = integrate_const(x0, y0, x_target, h / 2.0)
        calls_h2 = f_call_counter

        est_vec = (y_h2 - y_h) / (1.0 - 2.0 ** (-s))
        err = np.max(np.abs(est_vec))

        n_approx = int(np.ceil((x_target - x0) / h))
        print(f"iter {iter_count:1d}: h = {h:.6e}, n≈{n_approx:2d}, est_max = {err:.2e}, f_calls = {calls_h + calls_h2}")

        if err <= eps:
            return h, y_h2, est_vec

        h = h / 2.0

# оценка Локальная погрешности по Рунге
def single_step_estimations(x, y, h, s=2):
    y_bar = rk2_step(x, y, h)

    h2 = h / 2.0
    y_mid = rk2_step(x, y, h2)
    y_tilde = rk2_step(x + h2, y_mid, h2)

    denom = (1.0 - 2.0 ** (-s))
    est_vec_local = (y_tilde - y_bar) / denom

    return y_bar, y_tilde, est_vec_local

# Алгоритм с автоматическим выбором шага
def integrate_adaptive(x0, y0, x_target, h_initial, rho, s=2, max_steps=100000):
    global f_call_counter

    x = x0
    y = y0.copy()
    h = h_initial
    steps = 0

    while x < x_target - 1e-14 and steps < max_steps:

        if x + h > x_target:
            h = x_target - x

        y_bar, y_tilde, est_vec_local = single_step_estimations(x, y, h, s=s)
        rho_est = np.max(np.abs(est_vec_local))
        steps += 1

        print(f"step {steps:1d}: x = {x:.6f}, h = {h:.6e}, rho = {rho_est:.2e}")

        if rho_est > rho * (2 ** s):
            h = h / 2.0
            continue

        elif rho_est > rho and rho_est <= rho * (2 ** s):
            x += h
            y = y_tilde.copy()
            h = h / 2.0
            continue

        elif rho_est >= rho / (2 ** (s + 1)) and rho_est <= rho:
            x += h
            y = y_bar.copy()
            continue

        else:
            x += h
            y = y_bar.copy()
            h = 2.0 * h
            continue

    return y, steps


x0 = 0
x_end = np.pi
y0 = np.array([B*np.pi, A*np.pi])

print("\nЧАСТЬ №1\n")

h0 = initial_step(x0, y0, x_end, eps=1e-4)
print("h0 =", h0)

h_final, y_final, est_vec = h_runge(x0, y0, x_end, eps=1e-4, h_initial=h0)
print("\nИтоговый постоянный шаг:", h_final)
print("Решение:", y_final)
print("Полная погрешность:", est_vec)

print("\nЧАСТЬ №2\n")

f_call_counter = 0
y_adapt, steps_adapt = integrate_adaptive(x0, y0, x_end, h0, rho=1e-5)

print("\nРешение:", y_adapt)
print("Число шагов:", steps_adapt)
print("Вызовов правой части:", f_call_counter)
