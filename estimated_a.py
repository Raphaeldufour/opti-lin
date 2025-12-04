import numpy as np
from scipy.optimize import linprog
from scipy import sparse
import time
import matplotlib.pyplot as plt
from scipy.io import loadmat

X = loadmat('pedsX_il.mat')['X']
b_estime = np.load('b_estimated.npy')

p, t = X.shape

c = np.ones(p + 1)
c[-1] = 0

diag_neg = -sparse.eye(p, format='csc')
A_v = sparse.vstack([diag_neg, diag_neg])

vec_b_neg = -b_estime.reshape((p, 1))
vec_b_pos = b_estime.reshape((p, 1))
A_b = np.vstack([vec_b_neg, vec_b_pos])
A_b_sparse = sparse.csc_matrix(A_b)

A_ub = sparse.hstack([A_v, A_b_sparse], format='csc')

a_estime = np.zeros(t)

for j in range(t):
    image_vals = X[:, j]
    
    b_ub_vec = np.concatenate((-image_vals, image_vals))
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub_vec, bounds=(None, None), method='highs')
    
    if res.success:
        a_estime[j] = res.x[-1]
    else:
        a_estime[j] = 1.0

    if j % 10 == 0:
        print(f"Image {j}/{t} traitée...")

F = X - np.outer(b_estime, a_estime)

idx = 50
if t > idx:
    img_F = F[:, idx].reshape((152, 232), order='F')
    plt.imshow(img_F, cmap='gray')
    plt.title(f"Avant-plan extrait (F) - Image {idx}")
    plt.axis('off')
    plt.show()