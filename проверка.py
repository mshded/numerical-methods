import numpy as np

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

def rk2_step(x, y, h):
    k1 = f(x, y)
    k2 = f(x + c2 * h, y + a21 * h * k1)
    return y + h * (b1 * k1 + b2 * k2)

def initial_step_general(x0, y0, x_end, eps, s, method_euler=False):
    f0 = f(x0, y0)

    denom = max(abs(x0), abs(x_end), 1e-12)
    delta = (1 / denom) ** (s + 1) + np.linalg.norm(f0) ** (s + 1)

    h = (eps / delta) ** (1 / (s + 1))

    # f0 ≈ 0
    if method_euler or np.allclose(f0, 0.0, atol=1e-12):
        y_e = y0 + h * f0
        x_new = x0 + h

        f1 = f(x_new, y_e)
        denom2 = max(abs(x_new), abs(x_end), 1e-12)
        delta2 = (1 / denom2) ** (s + 1) + np.linalg.norm(f1) ** (s + 1)
        h2 = (eps / delta2) ** (1 / (s + 1))

        h = min(h, h2)

    return min(h, x_end - x0)

def integrate_const(x0, y0, x_end, h, step_func):
    x = x0
    y = y0.copy()
    while x < x_end - 1e-14:
        step = min(h, x_end - x)
        y = step_func(x, y, step)
        x += step
    return y

def h_runge(x0, y0, x_target, eps, h_initial, s=2, step_func=rk2_step, max_iter=30):
    global f_call_counter
    h = h_initial
    for iter_count in range(1, max_iter + 1):
        f_call_counter = 0
        y_h = integrate_const(x0, y0, x_target, h, step_func)
        calls_h = f_call_counter
        f_call_counter = 0
        y_h2 = integrate_const(x0, y0, x_target, h/2.0, step_func)
        calls_h2 = f_call_counter
        est_vec = (y_h2 - y_h) / (1.0 - 2.0**(-s))
        err = np.max(np.abs(est_vec))
        n_approx = int(np.ceil((x_target - x0) / h))
        print(f"iter {iter_count}: h = {h:.6e}, n≈{n_approx}, est_max = {err:.2e}, f_calls = {calls_h + calls_h2}")
        if err <= eps:
            return h, y_h2, est_vec
        h = h / 2.0

def rk4_step(x, y, h):
    k1 = f(x, y)
    k2 = f(x + h/2, y + h/2 * k1)
    k3 = f(x + h/2, y + h/2 * k2)
    k4 = f(x + h, y + h * k3)
    return y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)

def rk4_step_embedded(x, y, h):
    h2 = h / 2.0
    y_bar = rk4_step(x, y, h)
    y_mid = rk4_step(x, y, h2)
    y_tilde = rk4_step(x + h2, y_mid, h2)
    return y_bar, y_tilde

def rk4_adaptive(x0, y0, x_end, eps, h0, s=4, max_steps=10000):
    x = x0
    y = y0.copy()
    h = h0
    steps = 0
    f_calls_total = 0
    while x < x_end - 1e-14 and steps < max_steps:
        if x + h > x_end:
            h = x_end - x
        y_bar, y_tilde = rk4_step_embedded(x, y, h)
        f_calls_total += 8
        est_local = (y_tilde - y_bar) / (1.0 - 2.0**(-s))
        rho = np.max(np.abs(est_local))
        eps_2s = eps * (2**s)
        eps_low = eps / (2**(s+1))
        print(f"step {steps:1d}: x = {x:.6f}, h = {h:.6e}, rho = {rho:.2e}")

        if rho > eps_2s:
            h = h / 2.0
            continue
        elif rho > eps:
            y = y_tilde
            x += h
            steps += 1
            h = h / 2.0
        elif rho >= eps_low:
            y = y_bar
            x += h
            steps += 1
        else:
            y = y_bar
            x += h
            steps += 1
            h = h * 2.0
    return y, steps, f_calls_total
def rk4_step(x, y, h):
    k1 = f(x, y)
    k2 = f(x + h/2, y + h/2 * k1)
    k3 = f(x + h/2, y + h/2 * k2)
    k4 = f(x + h, y + h * k3)
    return y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)

def rk4_step_embedded(x, y, h):
    h2 = h / 2.0
    y_bar = rk4_step(x, y, h)
    y_mid = rk4_step(x, y, h2)
    y_tilde = rk4_step(x + h2, y_mid, h2)
    return y_bar, y_tilde


def rk4_adaptive(x0, y0, x_end, eps, h0, s=4, max_steps=10000):
    x = x0
    y = y0.copy()
    h = h0

    xs = [x]
    hs = []

    steps = 0
    f_calls_total = 0

    while x < x_end - 1e-14 and steps < max_steps:
        if x + h > x_end:
            h = x_end - x

        y_bar, y_tilde = rk4_step_embedded(x, y, h)
        f_calls_total += 8

        est_local = (y_tilde - y_bar) / (1.0 - 2.0 ** (-s))
        rho = np.max(np.abs(est_local))

        eps_2s = eps * (2 ** s)
        eps_low = eps / (2 ** (s + 1))

        print(f"step {steps:1d}: x = {x:.6f}, h = {h:.6e}, rho = {rho:.2e}")

        if rho > eps_2s:
            h = h / 2.0
            continue

        if rho > eps:
            y = y_tilde
            x += h
            xs.append(x)
            hs.append(h)
            steps += 1
            h = h / 2.0

        elif rho >= eps_low:
            y = y_bar
            x += h
            xs.append(x)
            hs.append(h)
            steps += 1

        else:
            y = y_bar
            x += h
            xs.append(x)
            hs.append(h)
            steps += 1
            h = h * 2.0

    return y, xs, hs, steps, f_calls_total


x0 = 0
x_end = np.pi
y0 = np.array([B * np.pi, A * np.pi])
eps = 1e-4

print("3.1. РК4 с постоянным шагом\n")

# Для РК4 (s=4) рассчитываем начальный шаг отдельно
h0_rk4 = initial_step_general(x0, y0, x_end, eps, s=4, method_euler=True)
print(f"h0 = {h0_rk4:.6e}")

# Для постоянного шага РК4 нужно подобрать h через правило Рунге
h_final_rk4, y_const_rk4, est_vec_rk4 = h_runge(x0, y0, x_end, eps, h0_rk4, s=4, step_func=rk4_step)

print(f"Итоговый постоянный шаг для РК4: {h_final_rk4:.6e}")
n_steps_rk4 = int(np.ceil((x_end - x0) / h_final_rk4))
print(f"Число шагов: {n_steps_rk4}")
print(f"Решение: {y_const_rk4}")
print(f"Оценка погрешности: {np.max(np.abs(est_vec_rk4)):.2e}")

print("\n3.1. РК4 с автоматическим выбором шага\n")

# Для адаптивного РК4 используем начальный шаг h0_rk4
print(f"h0 = {h0_rk4:.6e}")
y_adapt, xs_adapt, hs_adapt, steps_adapt, calls_adapt = rk4_adaptive(x0, y0, x_end, eps, h0_rk4, s=4)

print(f"Решение: {y_adapt}")
print(f"Число шагов: {steps_adapt}")
print(f"Вызовов: {calls_adapt}")

print("\n3.2. Определение шага h для ЯМРК2\n")
h0_rk2 = initial_step_general(x0, y0, x_end, eps, s=2, method_euler=True)
print(f"h0 = {h0_rk2:.6e}")

h_final_rk2, y_final_rk2, est_vec_rk2 = h_runge(x0, y0, x_end, eps, h0_rk2, s=2, step_func=rk2_step)

print(f"Итоговый шаг для ЯМРК2: {h_final_rk2:.6e}")
n_steps = int(np.ceil((x_end - x0) / h_final_rk2))
print(f"Число шагов: {n_steps}")
print(f"Решение: {y_final_rk2}")
print(f"Оценка погрешности: {np.max(np.abs(est_vec_rk2)):.2e}")

