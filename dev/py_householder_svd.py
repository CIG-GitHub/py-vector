import numpy as np

def householder(A):
    m, n = A.shape
    V = np.eye(m)
    
    for i in range(min(m, n)):
        x = A[i:, i]
        e = np.zeros_like(x)
        e[0] = np.linalg.norm(x)
        if x[0] >= 0:
            v = x + e
        else:
            v = x - e
        v = v / (np.linalg.norm(v) + 1e-8)  # Avoid division by zero
        
        H = np.eye(m - i) - 2 * np.outer(v, v)

        A[i:, i:] = H @ A[i:, i:]
        A[i:, i+1:] = A[i:, i+1:] @ H[1:,1:].T

        V[i:, :] = V[i:, :] - 2 * np.outer(v, v @ V[i:, :])

    return A, V

def bidiag_to_svd(B):
    m, n = B.shape
    U = np.eye(m)
    V = np.eye(n)

    max_iter = 1000
    tol = 1e-10

    for _ in range(max_iter):
        off_diag_sum = np.sum(np.abs(np.diag(B, k=1)))
        if off_diag_sum < tol:
            break

        # Implicit QR step
        p = B[0, 0]**2 - B[0, 1]**2
        c = p / np.sqrt(p**2 + B[1, 0]**2 * p**2)
        s = B[1, 0] * B[0, 0] / np.sqrt(p**2 + B[1, 0]**2 * p**2)

        G = np.array([[c, -s], [s, c]])

        B[:2, :] = G.T @ B[:2, :]
        U[:, :2] = U[:, :2] @ G

        c = B[0, 0] / np.sqrt(B[0, 0]**2 + B[0, 1]**2)
        s = B[0, 1] / np.sqrt(B[0, 0]**2 + B[0, 1]**2)
        G = np.array([[c, s], [-s, c]])

        B[:, :2] = B[:, :2] @ G
        V[:, :2] = V[:, :2] @ G

    return U, B, V.T

def svd_householder(A):
    m, n = A.shape
    B, V = householder(A)
    U, S, Vt = bidiag_to_svd(B)
    return U, np.diag(S), Vt

if __name__ == '__main__':
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    U, S, Vt = svd_householder(A)
    print("U:\n", U)
    print("Singular values:\n", S)
    print("Vt:\n", Vt)
    print("Reconstructed A:\n", U @ np.diag(S) @ Vt)