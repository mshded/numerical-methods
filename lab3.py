import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.log(x**2 + 1e-10) + x**3

a, b = 0.1, 1
m = 100
n_max = 5
noise_std = 0.2  # Стандартное отклонение

np.random.seed(42)
x_points = np.linspace(a, b, m)
y_true = f(x_points)

# 3 измерения на точку
y_noisy = []
for x in x_points:
    y_noisy.extend([f(x) + np.random.normal(0, noise_std) for _ in range(3)])
x_expanded = np.repeat(x_points, 3)
y_expanded = np.array(y_noisy)

# МНК через нормальные уравнения
def MNK_normal(x, y, n):
    E = np.vander(x, n + 1, increasing=True)
    ETE = E.T @ E
    ETy = E.T @ y
    coeffs = np.linalg.solve(ETE, ETy)
    return coeffs

# МНК через ортогональные полиномы
def MNK_ortho(x, y, n):
    q = [np.ones_like(x)]  # q0(x) = 1
    alpha = []
    beta = [0]

    q1 = x - np.mean(x)  # q1(x) = x - alpha1
    q.append(q1)

    for j in range(1, n):
        alpha_j = np.sum(x * q[j]**2) / np.sum(q[j]**2)
        beta_j = np.sum(q[j]**2) / np.sum(q[j - 1]**2)

        alpha.append(alpha_j)
        beta.append(beta_j)

        q_next = (x - alpha_j) * q[j] - beta_j * q[j - 1]
        q.append(q_next)

    # c_j
    coeffs = [np.sum(y * qj) / np.sum(qj ** 2) for qj in q]
    return coeffs, q

# Функция для вычисления ортогонального полинома в новых точках
def ortho_to_poly(coeffs_ortho, q_ortho, x_eval):
    n = len(coeffs_ortho) - 1
    result = np.zeros_like(x_eval)

    # Вычисление ортогональных полиномов в точках x_eval
    q_eval = [np.ones_like(x_eval)]  # q0

    if n >= 1:
        alpha1 = np.mean(x_expanded)
        q_eval.append(x_eval - alpha1)  # q1

        for j in range(1, n):
            alpha_j = np.sum(x_expanded * q_ortho[j]**2) / np.sum(q_ortho[j]**2)
            beta_j = np.sum(q_ortho[j]**2) / np.sum(q_ortho[j - 1]**2)
            q_next = (x_eval - alpha_j) * q_eval[j] - beta_j * q_eval[j - 1]
            q_eval.append(q_next)

    for j in range(n + 1):
        result += coeffs_ortho[j] * q_eval[j]

    return result

results = []

plt.figure(figsize=(15, 10))
for n in range(1, n_max + 1):
    # МНК (нормальные уравнения)
    coeffs_normal = MNK_normal(x_expanded, y_expanded, n)
    y_pred_normal = np.polyval(coeffs_normal[::-1], x_points)
    error_normal = np.sum((y_pred_normal - y_true) ** 2)

    # МНК (ортогональные полиномы)
    coeffs_ortho, q_ortho = MNK_ortho(x_expanded, y_expanded, n)
    y_pred_ortho = ortho_to_poly(coeffs_ortho, q_ortho, x_points)
    error_ortho = np.sum((y_pred_ortho - y_true) ** 2)

    results.append((n, error_normal, error_ortho))

    plt.subplot(2, 3, n)
    plt.scatter(x_expanded, y_expanded, s=10, alpha=0.3, label='Данные с шумом')
    plt.plot(x_points, y_true, 'k--', label='Истинная функция')
    plt.plot(x_points, y_pred_normal, 'r-', label=f'МНК (n={n})')
    plt.plot(x_points, y_pred_ortho, 'b--', label='Ортогональный')
    plt.title(f'Степень {n}')
    plt.legend()

plt.tight_layout()
plt.show()

print(
    "| Степень полинома (n) | Сумма квадратов ошибок (нормальные уравнения) | Сумма квадратов ошибок (ортогональные полиномы) |")
print(
    "|----------------------|-----------------------------------------------|-------------------------------------------------|")
for n, en, eo in results:
    print(
        f"| {n}                    | {en:.4f}                                        | {eo:.4f}                                          |")