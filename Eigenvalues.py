import numpy as np

# Генерация матрицы А
def generate_matrix(n):
    # Диагональная матрица
    A_diag = np.diag(np.random.rand(n) * 10)

    # Генерация обратимой матрицы
    while True:
        C = np.random.rand(n, n)
        if np.linalg.det(C) != 0:
            break
    A = np.linalg.inv(C) @ A_diag @ C
    print('Матрица А:', A, sep='\n')
    print('Диагональная матрица A:', A_diag, sep='\n')
    return A, A_diag.diagonal()


# Степенной метод
def power_method(A, delta, rtol, max_iter=1000):
    n = A.shape[0]
    y = np.random.rand(n)
    z = y / np.linalg.norm(y)
    lambda_prev = 0
    history = []

    for k in range(1, max_iter + 1):
        y_new = A @ z
        z_new = y_new / np.linalg.norm(y_new)

        # Вычисление приближений для собственного числа
        I = np.where(np.abs(z) > delta)[0]
        lambda_i = y_new[I] / z[I]
        lambda_avg = np.mean(lambda_i)

        # Проверка сходимости
        if np.abs(lambda_avg - lambda_prev) < rtol * np.abs(lambda_avg):
            break
        lambda_prev = lambda_avg
        z = z_new
        history.append(lambda_avg)

    return lambda_avg, z_new, history


# Обратный степенной метод
def inverse_power_method(A, sigma, delta, rtol, max_iter=1000):
    n = A.shape[0]
    y = np.random.rand(n)
    z = y / np.linalg.norm(y)
    sigma_history = [sigma]
    v = z.copy()

    for k in range(1, max_iter + 1):
        try:
            y_new = np.linalg.solve(A - sigma * np.eye(n), z)
            z_new = y_new / np.linalg.norm(y_new)
        except np.linalg.LinAlgError:
            print(f"Ошибка при sigma = {sigma}: матрица вырождена")
            return None, None, None

        # Вычисление приближений
        I = np.where(np.abs(y_new) > delta)[0]
        if len(I) == 0:
            print(f"Все координаты близки к нулю при sigma = {sigma}")
            return None, None, None

        mu_i = z[I] / y_new[I]
        mu_avg = np.mean(mu_i)
        sigma_new = sigma + mu_avg

        # Проверка сходимости
        if np.abs(sigma_new - sigma) < rtol * np.abs(sigma_new):
            v = z_new / np.linalg.norm(z_new)
            break
        sigma = sigma_new
        z = z_new
        sigma_history.append(sigma)

    return sigma, v, sigma_history

def householder(v):
    n = len(v)
    H = np.eye(n) - 2 * np.outer(v, v) / np.dot(v, v)
    return H

def hessenberg_form(A):
    n = A.shape[0]
    H = A.copy().astype(float)

    for k in range(n - 2):
        x = H[k + 1:, k]
        e1 = np.zeros_like(x)
        e1[0] = 1

        alpha = -np.sign(x[0]) * np.linalg.norm(x)
        v = x - alpha * e1
        v = v / np.linalg.norm(v)

        Q_k = np.eye(n)
        Q_k[k + 1:, k + 1:] = householder(v)

        H = Q_k @ H @ Q_k.T

    return H


def qr_decomposition(A):
    n = A.shape[0]
    R = A.copy().astype(float)
    Q = np.eye(n)

    for k in range(n - 1):
        x = R[k:, k]
        e1 = np.zeros_like(x)
        e1[0] = 1

        alpha = -np.sign(x[0]) * np.linalg.norm(x)
        v = x - alpha * e1
        v = v / np.linalg.norm(v)

        H_k = np.eye(n)
        H_k[k:, k:] = householder(v)

        R = H_k @ R
        Q = Q @ H_k.T

    return Q, R


def qr_algorithm(A, eps, max_iter=1000):
    n = A.shape[0]
    H = hessenberg_form(A)
    eigenvalues = []

    for m in range(n, 0, -1):
        if m == 1:
            eigenvalues.append(H[0, 0])
            break

        for _ in range(max_iter):
            shift = H[m - 1, m - 1]
            H_shifted = H[:m, :m] - shift * np.eye(m)

            Q, R = qr_decomposition(H_shifted)
            H_new = R @ Q + shift * np.eye(m)
            H[:m, :m] = H_new

            if np.abs(H_new[m - 1, m - 2]) < eps:
                eigenvalues.append(H_new[m - 1, m - 1])
                break

    return sorted(eigenvalues, reverse=True)

n = 4
delta=1e-8
rtol=1e-6
eps=1e-8

# Генерация матрицы
A, true_eigenvalues = generate_matrix(n)
print(f"Истинные собственные значения: {np.sort(true_eigenvalues)}")

# 1. Степенной метод
lambda_max, v_max, _ = power_method(A, delta, rtol)
print(f"\nСтепенной метод:\nλ_max = {lambda_max:.8f}")

# 2. Обратный степенной метод для всех собственных значений
print("\nОбратный степенной метод:")
found_eigenvalues = []
found_eigenvectors = []
for sigma in true_eigenvalues:
    lambda_i, v_i, _ = inverse_power_method(A, sigma, delta, rtol)
    found_eigenvalues.append(lambda_i)
    found_eigenvalues.append(v_i)
    print(f"σ = {sigma:.2f} λ = {lambda_i:.8f}")
    print("v = ",np.array_str(v_i))

# 3. QR-алгоритм
qr_eigenvalues = qr_algorithm(A,eps)
print("\nQR-алгоритм:", np.sort(qr_eigenvalues))
