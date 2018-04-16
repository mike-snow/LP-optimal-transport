import numpy as np
from cvxopt import spmatrix, matrix, solvers

from scipy.sparse import bsr_matrix, eye, kron, hstack, vstack
from scipy.spatial.distance import cdist


def _normalise(density):
    return density / float(np.sum(density))


def _coeff_matrix(m, n):
    a = hstack([kron(eye(m), np.ones(n)), bsr_matrix(np.ones((m, 1)))])
    b = hstack([kron(np.ones(m), eye(n)), bsr_matrix(np.zeros((n, 1)))])
    A = vstack([a, b]).tocoo()
    return spmatrix(A.data.tolist(), A.row.tolist(), A.col.tolist(), size=(m + n, m * n + 1))


def _2dcost(source, target):
    m1, m2 = source.shape
    n1, n2 = target.shape
# create meshgrids on domain [0,1] x [0,1]
    X1, X2 = np.linspace(0, 1, m1), np.linspace(0, 1, m2)
    Y1, Y2 = np.linspace(0, 1, n1), np.linspace(0, 1, n2)
    x1, x2 = np.meshgrid(X1, X2)
    y1, y2 = np.meshgrid(Y1, Y2)
# coordinate array
    coords_x = np.c_[x1.flatten(), x2.flatten()]
    coords_y = np.c_[y1.flatten(), y2.flatten()]
# pairwise distance between coordinates
    cost = np.append(cdist(coords_x, coords_y, 'sqeuclidean').flatten(), 0)
    return cost


def _LPSolution(source, target):
    # Set up LP problem & solve 

    A = _coeff_matrix(M, N)
    b = matrix(np.r_[_normalise(source.flatten('F')), _normalise(target.flatten('F'))])
    c = matrix(_2dcost(source, target))

    # Ensure x \geq 0 or in cvxopt Gx \leq h
    G = spmatrix(-1.0, range(M * N + 1), range(M * N + 1)) 
    h = matrix(np.zeros(M * N + 1)) 

    try:
        solution = solvers.lp(c, G, h, A, b, solver='mosek')
    except:
        solution = solvers.lp(c, G, h, A, b, solver='solver')
    
    return solution['x']