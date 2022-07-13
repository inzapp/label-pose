import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm


g_generated_dir = './generated'
g_augment_values = [
    [-45, 0.75],
    [-30, 0.85],
    [-15, 0.95],
    [15, 0.95],
    [30, 0.85],
    [45, 0.75]
]


def main():
    global g_generated_dir, g_augment_values
    os.makedirs(g_generated_dir, exist_ok=True)
    image_paths = glob('*.jpg')
    for image_path in tqdm(image_paths):
        basename = os.path.basename(image_path)[:-4]
        raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img_h, img_w = raw.shape[:2]
        center_point = (raw.shape[1] // 2, raw.shape[0] // 2)
        label_path = f'{image_path[:-4]}.txt'
        with open(label_path, 'rt') as f:
            lines = f.readlines()
        for i, val in enumerate(g_augment_values):
            angle, scale = val
            m = np.asarray(cv2.getRotationMatrix2D(center_point, angle, scale))
            img = cv2.warpAffine(raw, m, (0, 0))
            label = ''
            for line in lines:
                confidence, x_pos, y_pos = list(map(float, line.split()))
                x_pos *= img_w
                y_pos *= img_h
                new_x = m[0][0] * x_pos + m[0][1] * y_pos + m[0][2]
                new_y = m[1][0] * x_pos + m[1][1] * y_pos + m[1][2]
                new_x /= img_w
                new_y /= img_h
                new_x, new_y = np.clip([new_x, new_y], 0.0, 1.0)
                label += f'{confidence:.6f} {new_x:.6f} {new_y:.6f}\n'
            cv2.imwrite(f'{g_generated_dir}/{basename}_{i}.jpg', img)
            with open(f'{g_generated_dir}/{basename}_{i}.txt', 'wt') as f:
                f.writelines(label)


if __name__ == '__main__':
    main()

