import numpy as np

from cvxopt import spmatrix, matrix, solvers
from scipy.sparse import bsr_matrix, eye, kron, hstack, vstack

# assumes Mosek LP solver is installed. Can use default if not. 


def _normalise(density):
    return density / float(np.sum(density))


def _coeff_matrix(m, n):
    a = hstack([kron(eye(m), np.ones(n)), bsr_matrix(np.ones((m, 1)))])
    b = hstack([kron(np.ones(m), eye(n)), bsr_matrix(np.zeros((n, 1)))])
    A = vstack([a, b]).tocoo()
    return spmatrix(A.data.tolist(), A.row.tolist(), A.col.tolist(), size=(m + n, m * n + 1))


def _1dcost(source, target):
    cost = []
    m, n = len(source), len(target)
    x1 = np.linspace(0, 1, m)
    y1 = np.linspace(0, 1, n)

    for i in range(m):
        for j in range(n):
            cost.append((x1[i] - y1[j]) ** 2)
    cost.append(0)
    
    return cost


# Set up LP problem & solve 

def _LPSolution(source, target)@
    m, n = len(source), len(target)
    
    A = _coeff_matrix(m, n)
    b = matrix(np.r_[_normalise(source), _normalise(target)])
    c = matrix(_1dcost(source, target))
# specifiy bounds
    G = spmatrix(-1.0, range(m * n + 1), range(m * n + 1))
    h = matrix(np.zeros(n * m + 1)) # Ensure x \geq 0

    try:
        solution = solvers.lp(c, G, h, A, b, solver='mosek')
    except:
        solution = solvers.lp(c, G, h, A, b, solver='solver')
        
    return solution['x']