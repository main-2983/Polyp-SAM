import shutil
from glob import glob
from tqdm import tqdm
import numpy as np
np.random.seed(42)


def fast_copy(files, path):
    for img_path, mask_path in tqdm(files):
        shutil.copy(img_path, f"{path}/images")
        shutil.copy(mask_path, f"{path}/masks")


image_paths = glob("/home/nguyen.mai/Workplace/Dataset/data-science-bowl-2018/processed_stage1_train/images/*")
image_paths.sort()
mask_paths = glob("/home/nguyen.mai/Workplace/Dataset/data-science-bowl-2018/processed_stage1_train/masks/*")
mask_paths.sort()

train_valid_test_split = (0.8, 0.1, 0.1)

test_count = int(train_valid_test_split[2] * len(image_paths))
valid_count = test_count
train_count = len(image_paths) - test_count * 2

print(train_count, valid_count, test_count)
assert(test_count + valid_count + train_count == len(image_paths))

all_files = np.array(list(zip(image_paths, mask_paths)))
np.random.shuffle(all_files)

train_paths = all_files[:train_count]
valid_paths = all_files[train_count : train_count + valid_count]
test_paths  = all_files[-test_count:]

fast_copy(train_paths, "/home/nguyen.mai/Workplace/Dataset/data-science-bowl-2018/TrainDataset")
fast_copy(valid_paths, "/home/nguyen.mai/Workplace/Dataset/data-science-bowl-2018/ValDataset")
fast_copy(test_paths, "/home/nguyen.mai/Workplace/Dataset/data-science-bowl-2018/TestDataset")
