import math

def transpose(matrix):
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

def multiply_matrices(matrix_a, matrix_b):
    rows_a = len(matrix_a)
    cols_a = len(matrix_a[0])
    cols_b = len(matrix_b[0])

    result_matrix = [[0 for _ in range(cols_b)] for _ in range(rows_a)]
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result_matrix[i][j] += matrix_a[i][k] * matrix_b[k][j]
    return result_matrix

def calculate_eigenvalues(matrix):
    a, b, c = -sum([matrix[i][i] for i in range(len(matrix))]), \
              matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0] + matrix[0][0] * matrix[2][2] - matrix[0][2] * matrix[2][0] + matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1], \
              - (matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]) + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]))
    p = b - a**2 / 3
    q = 2 * (a / 3)**3 - a * b / 3 + c
    delta = (q / 2)**2 + (p / 3)**3
    if delta >= 0:
      u = (-q / 2 + math.sqrt(delta))**(1/3)
      v = (-q / 2 - math.sqrt(delta))**(1/3)
      eigenvalue1 = u + v - a / 3
      eigenvalue2 = - (u + v) / 2 - a / 3 + (u - v) / 2 * 1j * math.sqrt(3)
      eigenvalue3 = - (u + v) / 2 - a / 3 - (u - v) / 2 * 1j * math.sqrt(3)
      return [eigenvalue1, eigenvalue2, eigenvalue3]
    else:
      return "Complex eigenvalues not handled"

def calculate_eigenvectors(matrix, eigenvalues):
    eigenvectors = []
    for eigenvalue in eigenvalues:
        m = [[matrix[i][j] - eigenvalue * (i == j) for j in range(len(matrix))] for i in range(len(matrix))]
        
        if m[0][0] != 0:
          v1 = 1
          v2 = -m[0][1] / m[0][0]
          v3 = -m[0][2] / m[0][0]
        elif m[1][1] != 0:
          v2 = 1
          v1 = -m[1][0] / m[1][1]
          v3 = -m[1][2] / m[1][1]
        elif m[2][2] != 0:
          v3 = 1
          v1 = -m[2][0] / m[2][2]
          v2 = -m[2][1] / m[2][2]
        else:
          return "No unique solution"

        norm = math.sqrt(v1**2 + v2**2 + v3**2)
        eigenvectors.append([v1/norm, v2/norm, v3/norm])
    return eigenvectors

def svd(matrix):
    matrix_t = transpose(matrix)
    
    ata = multiply_matrices(matrix_t, matrix)
    aat = multiply_matrices(matrix, matrix_t)

    eigenvalues_ata = calculate_eigenvalues(ata)
    eigenvectors_ata = calculate_eigenvectors(ata, eigenvalues_ata)
    
    eigenvalues_aat = calculate_eigenvalues(aat)
    eigenvectors_aat = calculate_eigenvectors(aat, eigenvalues_aat)

    singular_values = [math.sqrt(val) for val in eigenvalues_ata]

    v = transpose(eigenvectors_ata)
    u = eigenvectors_aat
    s = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for i in range(len(singular_values)):
      s[i][i] = singular_values[i]
    
    return u, s, v