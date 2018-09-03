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


## https://www.kaggle.com/dremovd/goto-pytorch-baseline
directory = '../input'

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
import torchvision


def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            ConvRelu(in_channels, middle_channels),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNet11(nn.Module):
    def __init__(self, num_filters=32):
        """
        :param num_classes:
        :param num_filters:
        """
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        # Convolutions are from VGG11
        self.encoder = models.vgg11().features

        # "relu" layer is taken from VGG probably for generality, but it's not clear
        self.relu = self.encoder[1]

        self.conv1 = self.encoder[0]
        self.conv2 = self.encoder[3]
        self.conv3s = self.encoder[6]
        self.conv3 = self.encoder[8]
        self.conv4s = self.encoder[11]
        self.conv4 = self.encoder[13]
        self.conv5s = self.encoder[16]
        self.conv5 = self.encoder[18]

        self.center = DecoderBlock(num_filters * 8 * 2, num_filters * 8 * 2, num_filters * 8)
        self.dec5 = DecoderBlock(num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 8)
        self.dec4 = DecoderBlock(num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 4)
        self.dec3 = DecoderBlock(num_filters * (8 + 4), num_filters * 4 * 2, num_filters * 2)
        self.dec2 = DecoderBlock(num_filters * (4 + 2), num_filters * 2 * 2, num_filters)
        self.dec1 = ConvRelu(num_filters * (2 + 1), num_filters)

        self.final = nn.Conv2d(num_filters, 1, kernel_size=1, )

    def forward(self, x):
        conv1 = self.relu(self.conv1(x))
        conv2 = self.relu(self.conv2(self.pool(conv1)))
        conv3s = self.relu(self.conv3s(self.pool(conv2)))
        conv3 = self.relu(self.conv3(conv3s))
        conv4s = self.relu(self.conv4s(self.pool(conv3)))
        conv4 = self.relu(self.conv4(conv4s))
        conv5s = self.relu(self.conv5s(self.pool(conv4)))
        conv5 = self.relu(self.conv5(conv5s))

        center = self.center(self.pool(conv5))

        # Deconvolutions with copies of VGG11 layers of corresponding size
        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))
        return F.sigmoid(self.final(dec1))


def unet11(**kwargs):
    model = UNet11(**kwargs)

    return model


def get_model():
    model = unet11()
    model.train()
    return model.to(device)


import cv2
from pathlib import Path
from torch.nn import functional as F


def load_image(path, mask=False):
    """
    Load image from a given path and pad it on the sides, so that eash side is divisible by 32 (newtwork requirement)

    if pad = True:
        returns image as numpy.array, tuple with padding in pixels as(x_min_pad, y_min_pad, x_max_pad, y_max_pad)
    else:
        returns image as numpy.array
    """
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    height, width, _ = img.shape

    # Padding in needed for UNet models because they need image size to be divisible by 32
    if height % 32 == 0:
        y_min_pad = 0
        y_max_pad = 0
    else:
        y_pad = 32 - height % 32
        y_min_pad = int(y_pad / 2)
        y_max_pad = y_pad - y_min_pad

    if width % 32 == 0:
        x_min_pad = 0
        x_max_pad = 0
    else:
        x_pad = 32 - width % 32
        x_min_pad = int(x_pad / 2)
        x_max_pad = x_pad - x_min_pad

    img = cv2.copyMakeBorder(img, y_min_pad, y_max_pad, x_min_pad, x_max_pad, cv2.BORDER_REFLECT_101)
    if mask:
        # Convert mask to 0 and 1 format
        img = img[:, :, 0:1] // 255
        return torch.from_numpy(img).float().permute([2, 0, 1])
    else:
        img = img / 255.0
        return torch.from_numpy(img).float().permute([2, 0, 1])


# Adapted from vizualization kernel
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

from torch.utils import data


class TGSSaltDataset(data.Dataset):
    def __init__(self, root_path, file_list, is_test=False):
        self.is_test = is_test
        self.root_path = root_path
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        if index not in range(0, len(self.file_list)):
            return self.__getitem__(np.random.randint(0, self.__len__()))

        file_id = self.file_list[index]

        image_folder = os.path.join(self.root_path, "images")
        image_path = os.path.join(image_folder, file_id + ".png")

        mask_folder = os.path.join(self.root_path, "masks")
        mask_path = os.path.join(mask_folder, file_id + ".png")

        image = load_image(image_path)

        if self.is_test:
            return (image,)
        else:
            mask = load_image(mask_path, mask=True)
            return image, mask


depths_df = pd.read_csv(os.path.join(directory, 'train.csv'))

train_path = os.path.join(directory, 'train')
file_list = list(depths_df['id'].values)

device = "cuda"

import tqdm

file_list_val = file_list[::10]
file_list_train = [f for f in file_list if f not in file_list_val]
dataset = TGSSaltDataset(train_path, file_list_train)
dataset_val = TGSSaltDataset(train_path, file_list_val)

model = get_model()
#

learning_rate = 1e-4
loss_fn = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for e in range(1):
    train_loss = []
    for image, mask in tqdm.tqdm(data.DataLoader(dataset, batch_size=1, shuffle=True)):
        image = image.type(torch.float).to(device)
        y_pred = model(image)
        loss = loss_fn(y_pred, mask.to(device))

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        train_loss.append(loss.item())

    val_loss = []
    for image, mask in data.DataLoader(dataset_val, batch_size=1, shuffle=False):
        image = image.to(device)
        y_pred = model(image)

        loss = loss_fn(y_pred, mask.to(device))
        val_loss.append(loss.item())

    print("Epoch: %d, Train: %.3f, Val: %.3f" % (e, np.mean(train_loss), np.mean(val_loss)))