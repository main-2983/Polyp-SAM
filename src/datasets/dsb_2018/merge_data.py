import os
import shutil
import numpy as np
from tqdm import tqdm
from PIL import Image

import cv2

root_folder = "/home/nguyen.mai/Workplace/Dataset/data-science-bowl-2018/stage1_train"
save_folder = "/home/nguyen.mai/Workplace/Dataset/data-science-bowl-2018/processed_stage1_train"
for name in tqdm(os.listdir(root_folder)):
    data_path = os.path.join(root_folder, name)
    mask_path = os.path.join(data_path, "masks")
    mask_files = os.listdir(mask_path)
    mask_files.sort()
    mask = Image.open(os.path.join(mask_path, mask_files[0]))
    mask = np.asarray(mask.convert('L'))
    for i in range(1, len(mask_files)):
        m = Image.open(os.path.join(mask_path, mask_files[i]))
        m = np.asarray(m.convert('L'))
        mask = np.logical_or(mask, m)
    mask = mask.astype(np.float_) * 255
    cv2.imwrite(f"{save_folder}/masks/{name}.png", mask)
    shutil.copy(f"{data_path}/images/{name}.png", f"{save_folder}/images")
