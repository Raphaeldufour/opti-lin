import os
import numpy as np
from scipy.io import loadmat
from scipy.optimize import linprog
from scipy import sparse
from PIL import Image
import shutil
import subprocess
from tqdm import tqdm


def ensure_output_dir(path):
    os.makedirs(path, exist_ok=True)


def compute_foreground(X, b_estime):
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

    for j in tqdm(range(t), desc="Processing frames", unit="frame"):
        image_vals = X[:, j]
        b_ub_vec = np.concatenate((-image_vals, image_vals))

        res = linprog(c, A_ub=A_ub, b_ub=b_ub_vec, bounds=(None, None), method='highs-ipm')

        if res.success:
            a_estime[j] = res.x[-1]
        else:
            a_estime[j] = 1.0

    F = X - np.outer(b_estime, a_estime)
    return F


def save_frames(F, out_dir, shape=(152, 232), order='F', max_frames=200, every=1):
    p, t = F.shape
    count = 0
    for j in range(0, t, every):
        if count >= max_frames:
            break
        frame = F[:, j].reshape(shape, order=order)

        mn = frame.min()
        mx = frame.max()
        if mx - mn > 0:
            norm = (frame - mn) / (mx - mn)
        else:
            norm = np.zeros_like(frame)

        img = (255 * norm).astype(np.uint8)
        im = Image.fromarray(img)

        fname = os.path.join(out_dir, f"frame_{count:04d}.png")
        im.save(fname)
        count += 1

    print(f"Saved {count} frames to {out_dir}")
    return count


def make_video_ffmpeg(out_dir, out_video, framerate=10):
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg is None:
        print('ffmpeg not found on PATH — skipping video creation.')
        return False

    cmd = [
        ffmpeg,
        '-y',
        '-framerate', str(framerate),
        '-i', os.path.join(out_dir, 'frame_%04d.png'),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        out_video,
    ]

    print('Running:', ' '.join(cmd))
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode == 0:
        print(f'Video written to {out_video}')
        return True
    else:
        print('ffmpeg failed:')
        print(res.stderr.decode('utf-8'))
        return False


def main():
    data = loadmat('pedsX_il.mat')
    X = data['X']
    b_estime = np.load('b_estimated.npy')

    out_dir = 'output'
    ensure_output_dir(out_dir)

    print('Computing foreground (this may take a while)')
    F = compute_foreground(X, b_estime)

    print('Saving frames as PNG...')
    saved = save_frames(F, out_dir, max_frames=200, every=1)

    video_path = os.path.join(out_dir, 'foreground.mp4')
    ok = make_video_ffmpeg(out_dir, video_path, framerate=10)
    if not ok:
        print('If ffmpeg is unavailable, assemble the video with:')
        print('ffmpeg -framerate 10 -i output/frame_%04d.png -c:v libx264 -pix_fmt yuv420p output/foreground.mp4')


if __name__ == '__main__':
    main()