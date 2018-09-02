# import numpy as np
# import pandas as pd
#
# from random import randint
#
# import matplotlib.pyplot as plt
# plt.style.use('seaborn-white')
# import seaborn as sns
# sns.set_style("white")
#
# from sklearn.model_selection import train_test_split
#
# from skimage.transform import resize
#
# from keras.preprocessing.image import load_img
# from keras import Model
# from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# from keras.models import load_model
# from keras.optimizers import Adam
# from keras.utils.vis_utils import plot_model
# from keras.preprocessing.image import ImageDataGenerator
# from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout
#
# from tqdm import tqdm_notebook
#
# img_size_ori = 101
# img_size_target = 128
#
#
# def upsample(img):
#     if img_size_ori == img_size_target:
#         return img
#     return resize(img, (img_size_target, img_size_target), mode='constant', preserve_range=True)
#     # res = np.zeros((img_size_target, img_size_target), dtype=img.dtype)
#     # res[:img_size_ori, :img_size_ori] = img
#     # return res
#
#
# def downsample(img):
#     if img_size_ori == img_size_target:
#         return img
#     return resize(img, (img_size_ori, img_size_ori), mode='constant', preserve_range=True)
#     # return img[:img_size_ori, :img_size_ori]
#
# train_df = pd.read_csv("../input/train.csv", index_col="id", usecols=[0])
# depths_df = pd.read_csv("../input/depths.csv", index_col="id")
# train_df = train_df.join(depths_df)
# test_df = depths_df[~depths_df.index.isin(train_df.index)]
#
#
# train_df["images"] = [np.array(load_img("../input/train/images/{}.png".format(idx), grayscale=True)) / 255 for idx in tqdm_notebook(train_df.index)]
#
#
# train_df["masks"] = [np.array(load_img("../input/train/masks/{}.png".format(idx), grayscale=True)) / 255 for idx in tqdm_notebook(train_df.index)]
#
# train_df["coverage"] = train_df.masks.map(np.sum) / pow(img_size_ori, 2)
#
#
# def cov_to_class(val):
#     for i in range(0, 11):
#         if val * 10 <= i:
#             return i
#
#
# train_df["coverage_class"] = train_df.coverage.map(cov_to_class)
#
#
# fig, axs = plt.subplots(1, 2, figsize=(15,5))
# sns.distplot(train_df.coverage, kde=False, ax=axs[0])
# sns.distplot(train_df.coverage_class, bins=10, kde=False, ax=axs[1])
# plt.suptitle("Salt coverage")
# axs[0].set_xlabel("Coverage")
# axs[1].set_xlabel("Coverage class")
#
#
# plt.scatter(train_df.coverage, train_df.coverage_class)
# plt.xlabel("Coverage")
# plt.ylabel("Coverage class")
#
# sns.distplot(train_df.z, label="Train")
# sns.distplot(test_df.z, label="Test")
# plt.legend()
# plt.title("Depth distribution")
#
#
# # show some example images
# max_images = 60
# grid_width = 15
# grid_height = int(max_images / grid_width)
# fig, axs = plt.subplots(grid_height, grid_width, figsize=(grid_width, grid_height))
# for i, idx in enumerate(train_df.index[:max_images]):
#     img = train_df.loc[idx].images
#     mask = train_df.loc[idx].masks
#     ax = axs[int(i / grid_width), i % grid_width]
#     ax.imshow(img, cmap="Greys")
#     ax.imshow(mask, alpha=0.3, cmap="Greens")
#     ax.text(1, img_size_ori-1, train_df.loc[idx].z, color="black")
#     ax.text(img_size_ori - 1, 1, round(train_df.loc[idx].coverage, 2), color="black", ha="right", va="top")
#     ax.text(1, 1, train_df.loc[idx].coverage_class, color="black", ha="left", va="top")
#     ax.set_yticklabels([])
#     ax.set_xticklabels([])
# plt.suptitle("Green: salt. Top-left: coverage class, top-right: salt coverage, bottom-left: depth")
#
#
#
# # stratified train/validation split by salt coverage
# ids_train, ids_valid, x_train, x_valid, y_train, y_valid, cov_train, cov_test, depth_train, depth_test = train_test_split(
#     train_df.index.values,
#     np.array(train_df.images.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1),
#     np.array(train_df.masks.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1),
#     train_df.coverage.values,
#     train_df.z.values,
#     test_size=0.2, stratify=train_df.coverage_class, random_state=1337)
#
# # build model
