import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
from scipy.io import loadmat
import time

h, w = 152, 232
p = h * w
t = 100

X = loadmat('pedsX_il.mat')['X']

n_vars = t + 1

c = np.ones(n_vars) # return an array of ones of length n_vars
c[-1] = 0 

A_ub = np.zeros((2 * t, n_vars))

A_ub[:t, :t] = -np.eye(t)
A_ub[t:, :t] = -np.eye(t)

A_ub[:t, -1] = -1
A_ub[t:, -1] = 1

b_estimated = np.zeros(p)

for i in range(p):
    pixel_values = X[i, :]
    d_ub = np.concatenate([-pixel_values, pixel_values])
    
    res = linprog(c, A_ub=A_ub, b_ub=d_ub, method='highs')
    
    if res.success:
        b_estimated[i] = res.x[-1]
        if i % 1000 == 0:
            print(f"Pixel {i}/{p} traité.")
    else:
        b_estimated[i] = 0

image_background = b_estimated.reshape((h, w), order='F')

plt.figure(figsize=(10, 5))
plt.imshow(image_background, cmap='gray')
plt.title("Arrière-plan estimé b (reconstruit)")
plt.savefig('estimated_background.png')

print("L'image ci-dessus montre le fond b reconstruit.")
np.save('b_estimated.npy', b_estimated)
print("b_estimated sauvegardé dans 'b_estimated.npy'")