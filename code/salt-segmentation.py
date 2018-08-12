# -*- coding: utf-8 -*-
__author__ = 'Zhenyuan Shen: https://kaggle.com/szywind'

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os, time, gc, imutils, cv2
from keras.preprocessing.image import load_img
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm, tqdm_notebook
from keras import optimizers
from keras.models import model_from_json

from constants import *
from helpers import *
import math
import glob
import random
from PIL import Image
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
import unet, pspnet, tiramisunet, resnet_101, resnet_152
K.set_image_dim_ordering('tf')

np.set_printoptions(threshold='nan')


class SaltSeg():
    def __init__(self, train = True, input_width=INPUT_WIDTH, input_height=INPUT_HEIGHT, batch_size=32, epochs=100, learn_rate=1e-2, nb_classes=2):
        
        self.input_width = input_width
        self.input_height = input_height
        self.batch_size = batch_size
        self.epochs = epochs
        self.learn_rate = learn_rate
        self.nb_classes = nb_classes
        self.threshold = 0.5

        if MODEL_TYPE == MODEL.UNET or MODEL_TYPE == MODEL.REFINED_UNET:
            self.model = unet.get_unet_128(input_shape=(self.input_height, self.input_width, 1))

        elif MODEL_TYPE == MODEL.TIRAMISUNET:
            self.model = tiramisunet.get_tiramisunet(input_shape=(self.input_height, self.input_width, 1))

        elif MODEL_TYPE == MODEL.PSPNET2:
            self.model = pspnet.pspnet2(input_shape=(self.input_height, self.input_width, 1))

        elif MODEL_TYPE == MODEL.RESNET101:
            # self.model = resnet_101.unet_resnet101(self.input_height, self.input_width, 3)
            self.model = resnet_152.unet_resnet152(self.input_height, self.input_width, 3)

        self.model.summary()
        if train:
            self.net_path = '../weights/model.json'
            self.model_path = '../weights/salt-segmentation-model.h5'
            with open(self.net_path, 'w') as json_file:
                json_file.write(self.model.to_json())
        else:
            # self.net_path = '../weights/{}/model.json'.format(MODEL_DIR)
            # self.model_path = '../weights/{}/salt-segmentation-model.h5'.format(MODEL_DIR)

            self.net_path = '../weights/model.json'
            self.model_path = '../weights/salt-segmentation-model.h5'

        self.load_data()
        self.direct_result = True
        self.train_with_all = False
        self.apply_crf = False

    # Load Data & Make Train/Validation Split
    def load_data(self):
        df_train = pd.read_csv(os.path.join(INPUT_PATH, "train.csv"), index_col="id", usecols=[0])
        df_depths = pd.read_csv(os.path.join(INPUT_PATH, "depths.csv"), index_col="id")
        df_train = df_train.join(df_depths)
        self.df_test = df_depths[~df_depths.index.isin(df_train.index)]

        # train_images = [np.array(load_img("../input/train/images/{}.png".format(idx), grayscale=True)) / 255 for
        #                       idx in tqdm_notebook(df_train.index)]

        df_train["masks"] = [np.array(load_img("../input/train/masks/{}.png".format(idx), grayscale=True)) / 255 for idx
                             in tqdm_notebook(df_train.index)]

        def cov_to_class(val):
            for i in range(0, 11):
                if val * 10 <= i:
                    return i

        df_train["coverage"] = df_train.masks.map(np.sum) / float(self.input_width * self.input_height)
        df_train["coverage_class"] = df_train.coverage.map(cov_to_class)

        # stratified train/validation split by salt coverage
        self.ids_train, self.ids_valid, self.coverage_train, self.coverage_test, self.depth_train, self.depth_test = train_test_split(
            df_train.index.values,
            df_train.coverage.values,
            df_train.z.values,
            test_size=0.2, stratify=df_train.coverage_class, random_state=37)


    def train(self):

        try:
            self.model.load_weights(self.model_path)
        except:
            pass
        nTrain = len(self.ids_train)
        nValid = len(self.ids_valid)
        print('Training on {} samples'.format(nTrain))
        print('Validating on {} samples'.format(nValid))

        ## Prepare Data
        def train_generator():
            while True:
                for start in range(0, nTrain, self.batch_size):
                    x_batch = []
                    y_batch = []
                    end = min(start + self.batch_size, nTrain)
                    ids_train_batch = self.ids_train[start:end]

                    for img_name in ids_train_batch:
                        img = cv2.imread(os.path.join(INPUT_PATH, "train", "images", img_name + ".png")) #[..., 0]
                        img = cv2.resize(img, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
                        mask = cv2.imread(os.path.join(INPUT_PATH, "train", "masks", img_name + ".png"))[..., 0]
                        mask = cv2.resize(mask, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
                        # img, mask = randomShiftScaleRotate(img, mask,
                        #                                    shift_limit=(-0.0625, 0.0625),
                        #                                    scale_limit=(-0.125, 0.125),
                        #                                    rotate_limit=(-0, 0))
                        # img, mask = randomHorizontalFlip(img, mask)
                        # img = randomGammaCorrection(img)

                        img, mask = randomRotationAndFlip(img, mask)
                        # draw(img, mask)

                        if img.ndim == 2:
                            img = img[..., np.newaxis]

                        if self.direct_result:
                            mask = np.expand_dims(mask, axis=2)
                            x_batch.append(img)
                            y_batch.append(mask)

                        else:
                            target = np.zeros((mask.shape[0], mask.shape[1], self.nb_classes))
                            for k in range(self.nb_classes):
                                target[:,:,k] = (mask == k)
                            x_batch.append(img)
                            y_batch.append(target)


                    x_batch = np.array(x_batch, np.float32) / 255.0
                    y_batch = np.array(y_batch, np.float32) / 255.0

                    if USE_REFINE_NET:
                        yield x_batch, [y_batch, y_batch]
                    else:
                        yield x_batch, y_batch

        def valid_generator():
            while True:
                for start in range(0, nValid, self.batch_size):
                    x_batch = []
                    y_batch = []
                    end = min(start + self.batch_size, nValid)
                    ids_valid_batch = self.ids_valid[start:end]
                    for img_name in ids_valid_batch:
                        img = cv2.imread(os.path.join(INPUT_PATH, "train", "images", img_name + ".png")) # [..., 0]
                        img = cv2.resize(img, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
                        mask = cv2.imread(os.path.join(INPUT_PATH, "train", "masks", img_name + ".png"))[..., 0]
                        mask = cv2.resize(mask, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)

                        if img.ndim == 2:
                            img = np.expand_dims(img, axis=-1) # equivalent to img[..., np.newaxis]
                        if self.direct_result:
                            mask = np.expand_dims(mask, axis=2)
                            x_batch.append(img)
                            y_batch.append(mask)
                        else:
                            target = np.zeros((mask.shape[0], mask.shape[1], self.nb_classes))
                            for k in range(self.nb_classes):
                                target[:,:,k] = (mask == k)
                            x_batch.append(img)
                            y_batch.append(target)

                    x_batch = np.array(x_batch, np.float32) / 255.0
                    y_batch = np.array(y_batch, np.float32) / 255.0
                    if USE_REFINE_NET:
                        yield x_batch, [y_batch, y_batch]
                    else:
                        yield x_batch, y_batch

        callbacks = [EarlyStopping(monitor='val_loss',
                                       patience=10,
                                       verbose=1,
                                       min_delta=1e-4),
                    ReduceLROnPlateau(monitor='val_loss',
                                           factor=0.1,
                                           patience=3,
                                           cooldown=2,
                                           verbose=1),
                    ModelCheckpoint(filepath=self.model_path,
                                         save_best_only=True,
                                         save_weights_only=True),
                    TensorBoard(log_dir='logs')]

        # Set Training Options
        # opt = optimizers.RMSprop(lr=0.0001)
        opt = optimizers.Adam()

        if USE_REFINE_NET:
            self.model.compile(optimizer=opt,
                               loss=bce_dice_loss,
                               loss_weights=[1, 1],
                               metrics=[dice_score]
                               )
        else:
            self.model.compile(optimizer=opt,
                               loss=bce_dice_loss,
                               metrics=[dice_score]
                               )

        self.model.fit_generator(
            generator=train_generator(),
            steps_per_epoch=math.ceil(nTrain / float(self.batch_size)),
            epochs=1,
            verbose=1,
            callbacks=callbacks,
            validation_data=valid_generator(),
            validation_steps=math.ceil(nValid / float(self.batch_size)))

        self.model.fit_generator(
            generator=train_generator(),
            steps_per_epoch=math.ceil(nTrain / float(self.batch_size)),
            epochs=self.epochs,
            verbose=2,
            callbacks=callbacks,
            validation_data=valid_generator(),
            validation_steps=math.ceil(nValid / float(self.batch_size)))



    def test(self):
        if not os.path.isfile(self.net_path) or not os.path.isfile(self.model_path):
            raise RuntimeError("No model found.")

        # json_file = open(self.net_path, 'r')
        # loaded_model_json = json_file.read()
        # self.model = model_from_json(loaded_model_json)
        self.model.load_weights(self.model_path)

        ## find best threshold
        nValid = len(self.ids_valid)

        def valid_generator():
            while True:
                for start in range(0, nValid, self.batch_size):
                    x_batch = []
                    y_batch = []
                    end = min(start + self.batch_size, nValid)
                    ids_valid_batch = self.ids_valid[start:end]
                    for img_name in ids_valid_batch:
                        img = cv2.imread(os.path.join(INPUT_PATH, "train", "images", img_name + ".png")) # [..., 0]
                        img = cv2.resize(img, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
                        mask = cv2.imread(os.path.join(INPUT_PATH, "train", "masks", img_name + ".png"))[..., 0]
                        mask = cv2.resize(mask, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)

                        if img.ndim == 2:
                            img = np.expand_dims(img, axis=-1) # equivalent to img[..., np.newaxis]
                        if self.direct_result:
                            mask = np.expand_dims(mask, axis=2)
                            x_batch.append(img)
                            y_batch.append(mask)
                        else:
                            target = np.zeros((mask.shape[0], mask.shape[1], self.nb_classes))
                            for k in range(self.nb_classes):
                                target[:,:,k] = (mask == k)
                            x_batch.append(img)
                            y_batch.append(target)

                    x_batch = np.array(x_batch, np.float32) / 255.0
                    y_batch = np.array(y_batch, np.float32) / 255.0
                    if USE_REFINE_NET:
                        yield x_batch, [y_batch, y_batch]
                    else:
                        yield x_batch, y_batch

        p_valid = self.model.predict_generator(generator=valid_generator(),
                                               steps=math.ceil(nValid / float(self.batch_size)))

        print("Foo: ", len(p_valid))
        if USE_REFINE_NET:
            p_valid = p_valid[-1]

        y_valid = []
        for img_name in self.ids_valid:
            # j = np.random.randint(self.nAug)
            mask = cv2.imread(os.path.join(INPUT_PATH, "train", "masks", img_name + ".png"))[..., 0]
            mask = cv2.resize(mask, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
            # img = transformations2(img, j)
            y_valid.append(mask)

        y_valid = np.array(y_valid, np.float32)

        # best_score, best_threshold = find_best_seg_thr(y_valid, p_valid)
        # print(best_score, best_threshold)

        best_score, best_threshold = find_best_threshold(y_valid, p_valid)
        print(best_score, best_threshold)

        self.best_threshold = best_threshold


        ## test
        ids_test = self.df_test.index.values[:]

        nTest = len(ids_test)
        print('Testing on {} samples'.format(nTest))

        if False:
            def test_generator():
                while True:
                    for start in range(0, nTest, self.batch_size):
                        x_batch = []

                        end = min(start + self.batch_size, nTest)
                        ids_test_batch = ids_test[start:end]
                        for img_name in ids_test_batch:

                            img = cv2.imread(os.path.join(INPUT_PATH, "test", "images", img_name + ".png")) # [..., 0]
                            img = cv2.resize(img, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)

                            if img.ndim == 2:
                                img = np.expand_dims(img, axis=-1)  # equivalent to img[..., np.newaxis]

                            x_batch.append(img)
                        x_batch = np.array(x_batch, np.float32) / 255.0
                        yield x_batch

            pred_test = self.model.predict_generator(generator=test_generator(),
                                                  steps=math.ceil(nTest / float(self.batch_size)))

            if USE_REFINE_NET:
                pred_test = pred_test[-1]
        else:
            nbatch = 0
            IoU = 0
            pred_test = np.zeros([nTest, INPUT_HEIGHT, INPUT_WIDTH, 1])
            for start in range(0, nTest, self.batch_size):
                print(nbatch)
                nbatch += 1
                x_batch = []

                end = min(start + self.batch_size, nTest)
                ids_test_batch = ids_test[start:end]
                for img_name in ids_test_batch:

                    img = cv2.imread(os.path.join(INPUT_PATH, "test", "images", img_name + ".png")) # [..., 0]
                    img = cv2.resize(img, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)

                    if img.ndim == 2:
                        img = np.expand_dims(img, axis=-1)  # equivalent to img[..., np.newaxis]

                    x_batch.append(img)


                x_batch = np.array(x_batch, np.float32) / 255.0

                p_test = self.model.predict(x_batch, batch_size=self.batch_size)

                if USE_REFINE_NET:
                    p_test = p_test[-1]
                pred_test[start: end] = p_test

        print(pred_test.shape)

        pred_dict = {idx: RLenc(np.round(cv2.resize(pred_test[i], (ORIG_WIDTH, ORIG_HEIGHT), interpolation=cv2.INTER_LINEAR) > self.best_threshold))
                     for i, idx in enumerate(tqdm_notebook(ids_test))}
        sub = pd.DataFrame.from_dict(pred_dict, orient='index')
        sub.index.names = ['id']
        sub.columns = ['rle_mask']
        sub.to_csv('submission.csv')


        # if self.direct_result:
        #     result, probs = get_final_mask(p_test, self.threshold, apply_crf=self.apply_crf, images=images)
        # else:
        #     avg_p_test = p_test[...,1] - p_test[...,0]
        #     result = get_result(avg_p_test, 0)
        # str.extend(map(run_length_encode, result))

        # for i in range(len(y_batch)):
        #     IoU += numpy_dice_score(y_batch[i], result[i]) / nTest



        # save predicted masks
        # if not os.path.exists(OUTPUT_PATH):
        #     os.mkdir(OUTPUT_PATH)
        #
        # for i in range(start, end):
        #     image_path, mask_path = ids_test[i]
        #     img_path = image_path[image_path.rfind('/')+1:]
        #     res_mask = (255 * result[i-start]).astype(np.uint8)
        #     res_mask = np.dstack((res_mask,)*3)
        #     cv2.imwrite(OUTPUT_PATH + '{}'.format(img_path), res_mask)

        # print('mean IoU: {}'.format(IoU))


if __name__ == "__main__":
    ccs = SaltSeg(input_width=INPUT_WIDTH, input_height=INPUT_HEIGHT, train=IS_TRAIN, nb_classes=NUM_CLASS)
    if IS_TRAIN:
        ccs.train()
    ccs.test()