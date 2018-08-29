import cv2, os
from constants import *
import pandas as pd

df_train = pd.read_csv(os.path.join(INPUT_PATH, "train.csv"), index_col="id", usecols=[0])
df_depths = pd.read_csv(os.path.join(INPUT_PATH, "depths.csv"), index_col="id")
df_train = df_train.join(df_depths)

img_names = df_train.index.values

npixels = []
for img_name in img_names:
    img = cv2.imread(os.path.join(INPUT_PATH, "train", "images", img_name + ".png"))
    mask = cv2.imread(os.path.join(INPUT_PATH, "train", "masks", img_name + ".png"))[..., 0]

    mask = mask < 128
    num_of_pixel_in_mask = mask.sum()
    npixels.append(num_of_pixel_in_mask)
npixels.sort()
for i in npixels:
    print(i)