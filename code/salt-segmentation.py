# -*- coding: utf-8 -*-
__author__ = 'Zhenyuan Shen: https://kaggle.com/szywind'

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os, cv2
from keras.preprocessing.image import load_img
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, TensorBoard
from tqdm import tqdm, tqdm_notebook
from keras import optimizers

import math
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
import unet, pspnet, tiramisunet, resnet_50, resnet_101, resnet_152, densenet169, densenet161, densenet121
from losses import lovasz_loss, my_iou_metric, my_iou_metric_2
from keras.losses import binary_crossentropy
from segmentation_models import Unet
from segmentation_models.backbones import get_preprocessing
from segmentation_models.utils import set_trainable
from segmentation_models.backbones import get_preprocessing

from transform import train_augment
from constants import *
from helpers import bce_dice_loss, dice_score, evaluate_ious, find_best_threshold, RLenc, focal_loss

import keras.backend as K

K.set_image_dim_ordering('tf')

np.set_printoptions(threshold='nan')


class SaltSeg():
    def __init__(self):
        self.load_data()

    def set_model(self):
        if MODEL_TYPE == MODEL.UNET or MODEL_TYPE == MODEL.REFINED_UNET:
            self.model = unet.get_unet_128(input_shape=(INPUT_HEIGHT, INPUT_WIDTH, 1))

        elif MODEL_TYPE == MODEL.TIRAMISUNET:
            self.model = tiramisunet.get_tiramisunet(input_shape=(INPUT_HEIGHT, INPUT_WIDTH, 1))

        elif MODEL_TYPE == MODEL.PSPNET2:
            self.model = pspnet.pspnet2(input_shape=(INPUT_HEIGHT, INPUT_WIDTH, 1))

        elif MODEL_TYPE == MODEL.RESNET:
            # self.model = resnet_101.unet_resnet101(self.input_height, self.input_width, 3)
            # self.model = resnet_152.unet_resnet152(self.input_height, self.input_width, 3)
            # self.model = resnet_50.unet_resnet50(self.input_height, self.input_width, 3)
            # self.model = UResNet34(input_shape=(INPUT_HEIGHT, INPUT_WIDTH, 3))
            self.model = Unet(backbone_name='resnet34', encoder_weights='imagenet', freeze_encoder=True)

        # self.model.summary()

        with open(NET_FILE, 'w') as json_file:
            json_file.write(self.model.to_json())

    # Load Data & Make Train/Validation Split
    def load_data(self):
        df_train = pd.read_csv(os.path.join(INPUT_PATH, "train.csv"), index_col="id", usecols=[0])
        df_depths = pd.read_csv(os.path.join(INPUT_PATH, "depths.csv"), index_col="id")
        self.df_train = df_train.join(df_depths)
        self.df_test = df_depths[~df_depths.index.isin(df_train.index)]

        # train_images = [np.array(load_img("../input/train/images/{}.png".format(idx), grayscale=True)) / 255 for
        #                       idx in tqdm_notebook(df_train.index)]

# TODO
        # def cov_to_class(val):
        #     for i in range(0, 11):
        #         if val * 10 <= i:
        #             return i
        # if STRATIFIED_BY_COVERAGE:
        #     df_train["masks"] = [np.array(load_img("../input/train/masks/{}.png".format(idx), grayscale=True)) / 255 for idx
        #                          in tqdm_notebook(df_train.index)]
        #     self.df_train["coverage"] = df_train.masks.map(np.sum) / float(ORIG_WIDTH * ORIG_HEIGHT)
        #     self.df_train["stratify_class"] = self.df_train.coverage.map(cov_to_class)
        # else:
        #     max_depth = df_depths.max(0).values[-1]
        #     min_depth = df_depths.min(0).values[-1]
        #     depth_range = float(max_depth - min_depth)
        #     print(depth_range)
        #
        #     self.df_train["depth"] = [(depth - min_depth + 1) / (depth_range + 1) for depth in self.df_train.z]
        #     self.df_train["stratify_class"] = self.df_train.depth.map(cov_to_class)


        self.ids_test = self.df_test.index.values[:]

        # stratified train/validation split by salt coverage
        # self.ids_train, self.ids_valid, self.depth_train, self.depth_test = train_test_split(
        #     self.df_train.index.values,
        #     # self.df_train.coverage.values,
        #     self.df_train.z.values,
        #     test_size=0.2, stratify=self.df_train.stratify_class, random_state=37)


    def train_cv(self):
        fold_id = 0
        ious = 0

        kf = KFold(n_splits=CV_FOLD, shuffle=True, random_state=37)
        for train_index, valid_index in kf.split(self.df_train.index.values):

        # skf = StratifiedKFold(n_splits=CV_FOLD, random_state=37, shuffle=True)
        # for train_index, valid_index in skf.split(self.df_train.index.values, self.df_train.stratify_class.values):
            if DEBUG and fold_id > 0:
                break;
            self.set_model()
            ids_train = self.df_train.index.values[train_index]
            # self.coverage_train = self.df_train.stratify_class.values[train_index]
            ids_valid = self.df_train.index.values[valid_index]
            # self.coverage_valid = self.df_train.stratify_class.values[valid_index]

            print("Train {}-th fold".format(fold_id))


            if not SNAPSHOT_ENSEMBLING:
                self.model_path = '../weights/salt-segmentation-model{}.h5'.format(fold_id)
                y_valid_i, p_valid_i = self.train(fold_id, ids_train, ids_valid)
                print("y_valid_i, p_valid_i:", y_valid_i.shape, p_valid_i.shape)
                ious_i = evaluate_ious(y_valid_i, p_valid_i)
                print(list(ious_i))

                ious += ious_i
                print("Sum: ", list(ious))

            else:
                for j in range(M+1):
                    if j == 0:
                        continue
                        self.model_path = '../weights/salt-segmentation-model{}.h5'.format(fold_id)
                    else: # snapshot ensembling
                        self.model_path = '../weights/salt-segmentation-model{}-{}.h5'.format(fold_id, j)
                    y_valid_i, p_valid_i = self.train(fold_id, ids_train, ids_valid)
                    print("y_valid_i, p_valid_i:", y_valid_i.shape, p_valid_i.shape)

                    ious_i = evaluate_ious(y_valid_i, p_valid_i)
                    print(list(ious_i))
                    ious += ious_i
                print("Sum: ", list(ious))
            fold_id += 1

        ## find best threshold
        # best_score, best_threshold = find_best_seg_thr(y_valid, p_valid)

        # best_score, best_threshold = find_best_threshold(y_valid, p_valid)

        best_score, best_threshold = find_best_threshold(ious)
        print(best_score, best_threshold)

        self.best_threshold = best_threshold



    def train(self, fold_id, ids_train, ids_valid):
        # self.model_path = '../weights/salt-segmentation-model{}.h5'.format(fold_id)
        try:
            self.model.load_weights(self.model_path)
        except:
            pass

        nTrain = len(ids_train)
        nValid = len(ids_valid)
        print('Training on {} samples'.format(nTrain))
        print('Validating on {} samples'.format(nValid))

        ## Prepare Data
        def train_generator():
            while True:
                for start in range(0, nTrain, BN_SIZE):
                    x_batch = []
                    y_batch = []
                    end = min(start + BN_SIZE, nTrain)
                    ids_train_batch = ids_train[start:end]

                    for img_name in ids_train_batch:
                        img = cv2.imread(os.path.join(INPUT_PATH, "train", "images", img_name + ".png")) / 255
                        mask = cv2.imread(os.path.join(INPUT_PATH, "train", "masks", img_name + ".png"), cv2.IMREAD_GRAYSCALE) / 255
                        if np.sum(mask < 0.5) < 10:
                            continue
                        if MIDDLE_HEIGHT != ORIG_HEIGHT or MIDDLE_WIDTH != ORIG_WIDTH:
                            img = cv2.resize(img, (MIDDLE_WIDTH, MIDDLE_HEIGHT), interpolation=cv2.INTER_LINEAR)
                            mask = cv2.resize(mask, (MIDDLE_WIDTH, MIDDLE_HEIGHT), interpolation=cv2.INTER_LINEAR)
                            mask = (mask > 0.5).astype(np.float32)

                        # img, mask = randomShiftScaleRotate(img, mask,
                        #                                    shift_limit=(-0.08, 0.08),
                        #                                    scale_limit=(0, 0.125),
                        #                                    rotate_limit=(-0, 0))
                        # img, mask = randomHorizontalFlip(img, mask)
                        # # img = randomGammaCorrection(img)
                        # img = randomIntensityAugmentation(img)
                        #
                        # # img, mask = randomRotationAndFlip(img, mask)
                        # # draw(img, mask)

                        # cv2.imwrite(os.path.join("../tmp", img_name + ".png"), (img*255).astype(np.uint8))
                        # A = (mask*255).astype(np.uint8)
                        # cv2.imwrite(os.path.join("../tmp", img_name + "_mask.png"), np.dstack([A,A,A]))

                        img, mask = train_augment(img, mask)
                        # preprocessing_fn = get_preprocessing('resnet34')
                        # x = preprocessing_fn(x)

                        # print("Train: ", img.shape, mask.shape, img.dtype, mask.dtype)
                        # cv2.imwrite(os.path.join("../tmp", img_name + "_aug.png"), (img*255).astype(np.uint8))
                        # A = (mask*255).astype(np.uint8)
                        # cv2.imwrite(os.path.join("../tmp", img_name + "_mask_aug.png"), np.dstack([A,A,A]))

                        if img.ndim == 2:
                            img = img[..., np.newaxis]

                        mask = np.expand_dims(mask, axis=-1)
                        x_batch.append(img)
                        y_batch.append(mask)

                    x_batch = np.array(x_batch, np.float32)
                    y_batch = np.array(y_batch, np.float32)

                    if USE_REFINE_NET:
                        yield x_batch, [y_batch, y_batch]
                    else:
                        yield x_batch, [y_batch]

        def valid_generator():
            while True:
                for start in range(0, nValid, BN_SIZE):
                    x_batch = []
                    y_batch = []
                    end = min(start + BN_SIZE, nValid)
                    ids_valid_batch = ids_valid[start:end]
                    for img_name in ids_valid_batch:
                        img = cv2.imread(os.path.join(INPUT_PATH, "train", "images", img_name + ".png")) / 255
                        mask = cv2.imread(os.path.join(INPUT_PATH, "train", "masks", img_name + ".png"), cv2.IMREAD_GRAYSCALE) / 255  # [..., 0]


                        img = cv2.resize(img, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
                        mask = cv2.resize(mask, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
                        mask = (mask > 0.5).astype(np.float32)

                        # img, mask = do_center_pad_to_factor2(img, mask, factor=32)
                        # print("Valid: ", img.shape, mask.shape, img.dtype, mask.dtype)
                        if img.ndim == 2:
                            img = np.expand_dims(img, axis=-1) # equivalent to img[..., np.newaxis]
                        mask = np.expand_dims(mask, axis=-1)
                        x_batch.append(img)
                        y_batch.append(mask)

                    x_batch = np.array(x_batch, np.float32)
                    y_batch = np.array(y_batch, np.float32)
                    if USE_REFINE_NET:
                        yield x_batch, [y_batch, y_batch]
                    else:
                        yield x_batch, [y_batch]

        if IS_TRAIN and fold_id >= START:
            self.model.compile('sgd', 'binary_crossentropy', ['binary_accuracy'])

            # pretrain model decoder
            self.model.fit_generator(
                generator=train_generator(),
                steps_per_epoch=math.ceil(nTrain / float(BN_SIZE)),
                epochs=5,
                verbose=2,
                validation_data=valid_generator(),
                validation_steps=math.ceil(nValid / float(BN_SIZE)))

            # release all layers for training
            set_trainable(self.model)  # set all layers trainable and recompile model

            ''' first stage training with bce loss '''

            # callbacks = [
            #             EarlyStopping(monitor='val_my_iou_metric', mode='max', # ''val_loss',
            #                                patience=16,
            #                                verbose=1),
            #             ReduceLROnPlateau(monitor='val_my_iou_metric', mode='max', factor=0.5, patience=5, min_lr=0.00001, verbose=1),
            #             ModelCheckpoint(filepath=self.model_path, monitor='val_my_iou_metric', mode='max', save_best_only=True, save_weights_only=True, verbose=1),
            #             TensorBoard(log_dir='logs')]

            callbacks = [
                        EarlyStopping(monitor='val_loss',
                                           patience=16,
                                           verbose=1),
                        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1),
                        ModelCheckpoint(filepath=self.model_path, monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1),
                        TensorBoard(log_dir='logs')]


            # Set Training Options
            # opt = optimizers.RMSprop(lr=0.0001)
            # opt = optimizers.Adam(lr=1e-4)

            opt = optimizers.SGD(lr=1e-2, momentum=0.9, decay=0.0001)

            if USE_REFINE_NET:
                self.model.compile(optimizer=opt,
                                   loss=bce_dice_loss,
                                   loss_weights=[1, 1],
                                   metrics=[dice_score]
                                   )
            else:
                self.model.compile(optimizer=opt,
                                   # loss=[focal_loss(0.75)],
                                   loss=[binary_crossentropy],
                                   metrics=[my_iou_metric] # 'acc'
                               )

            self.model.fit_generator(
                generator=train_generator(),
                steps_per_epoch=math.ceil(nTrain / float(BN_SIZE)),
                epochs=20,
                verbose=2,
                callbacks=callbacks,
                validation_data=valid_generator(),
                validation_steps=math.ceil(nValid / float(BN_SIZE)))


            ''' second stage training with Lovasz loss '''

            from snapshot import SnapshotCallbackBuilder
            ''' Snapshot major parameters '''

            nb_epoch = T = EPOCHS  # number of epochs
            alpha_zero = 1e-2  # initial learning rate

            snapshot = SnapshotCallbackBuilder(T, M, alpha_zero)
            snapshotcallbacks = snapshot.get_callbacks(model_prefix=self.model_path)  # Build snapshot callbacks]

            input_x = self.model.layers[0].input

            output_layer = self.model.layers[-1].input

            from keras.models import Model
            model2 = Model(input_x, output_layer)

            if USE_REFINE_NET:
                model2.compile(optimizer=opt,
                                   loss=bce_dice_loss,
                                   loss_weights=[1, 1],
                                   metrics=[dice_score]
                                   )
            else:
                model2.compile(optimizer=opt,
                                   loss=[lovasz_loss],
                                   metrics=[my_iou_metric_2]
                               )

            model2.fit_generator(
                generator=train_generator(),
                steps_per_epoch=math.ceil(nTrain / float(BN_SIZE)),
                epochs=EPOCHS,
                verbose=2,
                callbacks=snapshotcallbacks,
                validation_data=valid_generator(),
                validation_steps=math.ceil(nValid / float(BN_SIZE)))

        ## evaluate on validation set
        self.model.load_weights(self.model_path)
        p_valid = self.model.predict_generator(generator=valid_generator(),
                                               steps=math.ceil(nValid / float(BN_SIZE))) # [-1]
        # p_valid = sigmoid(p_valid)

        if USE_REFINE_NET:
            p_valid = p_valid[-1]

        y_valid = []
        for img_name in ids_valid:
            mask = cv2.imread(os.path.join(INPUT_PATH, "train", "masks", img_name + ".png"), cv2.IMREAD_GRAYSCALE) / 255

            mask = cv2.resize(mask, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
            mask = (mask > 0.5).astype(np.float32)

            # mask = do_center_pad_to_factor(mask, factor=32)

            y_valid.append(mask)

        y_valid = np.array(y_valid, np.float32)
        return y_valid, p_valid

    def test_cv(self):
        self.set_model()

        try:
            self.best_threshold
        except AttributeError:
            self.best_threshold = DEFAULT_THRESHOLD
        print("THESHOLD: ", self.best_threshold)
        def get_mask(pred):
            # TODO
            # pred = pred[PAD_FRONT:-PAD_END, PAD_FRONT:-PAD_END]
            # if MIDDLE_HEIGHT == ORIG_HEIGHT and MIDDLE_WIDTH == ORIG_WIDTH:
            #     mask = pred > self.best_threshold
            # else:
            #     mask = cv2.resize(pred, (ORIG_WIDTH, ORIG_HEIGHT), interpolation=cv2.INTER_LINEAR) > self.best_threshold

            mask = cv2.resize(pred, (ORIG_WIDTH, ORIG_HEIGHT), interpolation=cv2.INTER_LINEAR) > self.best_threshold

            if np.sum(mask) < 10:
                mask = np.zeros(mask.shape)
            return mask

        if DEBUG:
            fold_id = 0
            if not SNAPSHOT_ENSEMBLING:
                self.model_path = '../weights/salt-segmentation-model{}.h5'.format(fold_id)
                pred_test = self.test(fold_id)
            else:
                pred_test = 0
                for j in range(M + 1):
                    if j == 0:
                        continue
                        print("1111111111111111111111111111111111111111111111111111111")
                        self.model_path = '../weights/salt-segmentation-model{}.h5'.format(fold_id)
                    else:
                        self.model_path = '../weights/salt-segmentation-model{}-{}.h5'.format(fold_id, j)
                    pred_test_i = self.test(0)
                    pred_test += pred_test_i
                pred_test /= float(M)

        else:
            pred_test = 0
            if not SNAPSHOT_ENSEMBLING:
                for fold_id in range(CV_FOLD):
                    self.model_path = '../weights/salt-segmentation-model{}.h5'.format(fold_id)
                    pred_test_i = self.test(fold_id)
                    pred_test += pred_test_i
                pred_test /= float(CV_FOLD)
            else:
                for fold_id in range(CV_FOLD):
                    for j in range(M+1):
                        if j == 0:
                            self.model_path = '../weights/salt-segmentation-model{}.h5'.format(fold_id)
                        else:
                            self.model_path = '../weights/salt-segmentation-model{}-{}.h5'.format(fold_id, j)
                        pred_test_i = self.test(fold_id)
                        pred_test += pred_test_i
                pred_test /= float(CV_FOLD*(M+1))

        pred_dict = {idx: RLenc(np.round(get_mask(pred_test[i])))
                     for i, idx in enumerate(tqdm_notebook(self.ids_test))}
        sub = pd.DataFrame.from_dict(pred_dict, orient='index')
        sub.index.names = ['id']
        sub.columns = ['rle_mask']
        sub.to_csv('submission.csv')

    def test(self, fold_id=''):
        # self.model_path = '../weights/salt-segmentation-model{}.h5'.format(fold_id)

        if not os.path.isfile(NET_FILE) or not os.path.isfile(self.model_path):
            raise RuntimeError("No model found.")

        # json_file = open(self.net_path, 'r')
        # loaded_model_json = json_file.read()
        # self.model = model_from_json(loaded_model_json)
        self.model.load_weights(self.model_path)

        ## test
        nTest = len(self.ids_test)
        print('Testing on {} samples'.format(nTest))

        if False:
            def test_generator():
                while True:
                    for start in range(0, nTest, BN_SIZE):
                        x_batch = []

                        end = min(start + BN_SIZE, nTest)
                        ids_test_batch = self.ids_test[start:end]
                        for img_name in ids_test_batch:

                            img = cv2.imread(os.path.join(INPUT_PATH, "test", "images", img_name + ".png")) / 255 # [..., 0]

                            img = cv2.resize(img, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)

                            # img = do_center_pad_to_factor(img, factor=32)
                            if img.ndim == 2:
                                img = np.expand_dims(img, axis=-1)  # equivalent to img[..., np.newaxis]

                            x_batch.append(img)
                        x_batch = np.array(x_batch, np.float32)
                        yield x_batch

            pred_test = self.model.predict_generator(generator=test_generator(),
                                                  steps=math.ceil(nTest / float(BN_SIZE))) # [-1]
            # pred_test = sigmoid(pred_test)

            if USE_REFINE_NET:
                pred_test = pred_test[-1]
        else:
            nbatch = 0
            pred_test = np.zeros([nTest, INPUT_HEIGHT, INPUT_WIDTH, 1])
            for start in range(0, nTest, BN_SIZE):
                nbatch += 1
                x_batch = []

                end = min(start + BN_SIZE, nTest)
                ids_test_batch = self.ids_test[start:end]
                for img_name in ids_test_batch:

                    img = cv2.imread(os.path.join(INPUT_PATH, "test", "images", img_name + ".png")) / 255 # [..., 0]
                    if MIDDLE_HEIGHT == ORIG_HEIGHT and MIDDLE_WIDTH == ORIG_WIDTH:
                        pass
                    else:
                        img = cv2.resize(img, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)

                    # img = do_center_pad_to_factor(img, factor=32)

                    if img.ndim == 2:
                        img = np.expand_dims(img, axis=-1)  # equivalent to img[..., np.newaxis]

                    x_batch.append(img)

                x_batch = np.array(x_batch, np.float32) # / 255.0

                p_test = self.model.predict(x_batch, batch_size=BN_SIZE) # [-1]
                # p_test = sigmoid(p_test)

                if USE_REFINE_NET:
                    p_test = p_test[-1]

                if N_TTA > 0:
                    p_test_flip = self.model.predict(x_batch[:, :, ::-1, :], batch_size=BN_SIZE) # [-1]
                    # p_test_flip = sigmoid(p_test_flip)

                    if USE_REFINE_NET:
                        p_test_flip = p_test_flip[-1]

                    p_test = (p_test + p_test_flip[:, :, ::-1, :]) / float(N_TTA)

                pred_test[start: end] = p_test

        print(pred_test.shape)
        return pred_test


if __name__ == "__main__":
    ccs = SaltSeg()
    ccs.train_cv()
    ccs.test_cv()