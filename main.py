from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import glob

from dataset import PromptPolypDataset


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=100):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))


image_paths = glob.glob("/home/nguyen.mai/Workplace/sun-polyp/Dataset/TrainDataset/image/*")
mask_paths = glob.glob("/home/nguyen.mai/Workplace/sun-polyp/Dataset/TrainDataset/mask/*")

dataset = PromptPolypDataset(image_paths, mask_paths)
pprint(dataset.mask_paths.index('/home/nguyen.mai/Workplace/sun-polyp/Dataset/TrainDataset/mask/cju1cu1u2474n0878tt7v4tdr.png'))

index = 1145
mask = dataset.binary_loader(mask_paths[index])
image = dataset.rgb_loader(image_paths[index])

# bboxes = dataset.sample_box(mask)
# print(bboxes)
#
# for i, box in enumerate(bboxes):
#     plt.figure(i)
#     plt.imshow(image)
#     show_mask(mask/255, plt.gca())
#     show_box(box, plt.gca())
#     plt.show()

x_coords, y_coords = dataset.uniform_sample_points(mask, num_points=3)
points = np.hstack([x_coords, y_coords])
print(points)

for i, point in enumerate(points):
    plt.figure(i)
    plt.imshow(image)
    show_mask(mask/255, plt.gca())
    show_points(point, 1, plt.gca())
    plt.show()