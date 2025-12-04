import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io

OUTPUT_DIR = 'median_b_outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)
data = scipy.io.loadmat('pedsX_il.mat')
X = data['X']

p_rows, p_cols = 152, 232
t = X.shape[1]

b_est = np.median(X, axis=1)

a_est = np.zeros(t)
for j in range(t):
    mask = np.abs(b_est) > 1e-5
    if np.sum(mask) > 0:
        a_est[j] = np.median(X[mask, j] / b_est[mask])
    else:
        a_est[j] = 1.0

B_final = np.outer(b_est, a_est)
F_final = X - B_final

threshold = 0.1 * np.max(np.abs(F_final))
F_cleaned = np.where(np.abs(F_final) > threshold, F_final, 0)

for k in range(t):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    img_orig = X[:, k].reshape((p_rows, p_cols), order='F')
    axes[0].imshow(img_orig, cmap='gray')
    axes[0].set_title(f"Frame {k+1} - Originale")
    axes[0].axis('off')
    
    img_back = (b_est * a_est[k]).reshape((p_rows, p_cols), order='F')
    axes[1].imshow(img_back, cmap='gray')
    axes[1].set_title(f"Frame {k+1} - Arrière-plan (Reconstruit)")
    axes[1].axis('off')
    
    img_fore = F_cleaned[:, k].reshape((p_rows, p_cols), order='F')
    axes[2].imshow(img_fore, cmap='gray')
    axes[2].set_title(f"Frame {k+1} - Avant-plan (Objet mobile)")
    axes[2].axis('off')
    
    filename = os.path.join(OUTPUT_DIR, f"resultat_frame_{k+1:03d}.png")
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig) 

print("Sauvegarde terminée")