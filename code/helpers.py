from sklearn.metrics import fbeta_score
import numpy as np
import cv2
import imutils
import keras.backend as K
# import matplotlib.pylab as plt
import scipy.stats as stats

import pydensecrf.densecrf as dcrf

from pydensecrf.utils import compute_unary, create_pairwise_bilateral, \
    create_pairwise_gaussian, softmax_to_unary, unary_from_softmax
import tensorflow as tf
from keras.losses import binary_crossentropy
from skimage.transform import resize
from constants import *
from imgaug import augmenters as iaa



def randomIntensityAugmentation(image):
    intensity_seq = iaa.Sequential([
        # iaa.Invert(0.3),
        iaa.Sometimes(0.3, iaa.ContrastNormalization((0.5, 1.5))),
        iaa.OneOf([
            iaa.Noop(),
            iaa.Sequential([
                iaa.OneOf([
                    iaa.Add((-10, 10)),
                    iaa.AddElementwise((-10, 10)),
                    iaa.Multiply((0.95, 1.05)),
                    iaa.MultiplyElementwise((0.95, 1.05)),
                ]),
            ]),
            iaa.OneOf([
                iaa.GaussianBlur(sigma=(0.0, 1.0)),
                iaa.AverageBlur(k=(2, 5)),
                iaa.MedianBlur(k=(3, 5))
            ])
        ])
    ], random_order=False)
    return intensity_seq.augment_images(image)

def random_crop(img, dstSize, center=False):
    import random
    srcH, srcW = img.shape[:2]
    dstH, dstW = dstSize
    if srcH <= dstH or srcW <= dstW:
        return img
    if center:
        y0 = int((srcH - dstH) / 2)
        x0 = int((srcW - dstW) / 2)
    else:
        y0 = random.randrange(0, srcH - dstH)
        x0 = random.randrange(0, srcW - dstW)
    return img[y0:y0+dstH, x0:x0+dstW]

def randomRotationAndFlip(image, mask):
    choice = np.random.randint(0, 8, 1)[0]
    mode = choice // 2
    # image = imutils.rotate(image, mode * 90)
    # mask = imutils.rotate(mask, mode * 90)

    if choice % 2 == 1:
        image = cv2.flip(image, flipCode=1)
        mask = cv2.flip(mask, flipCode=1)

    return image, mask

def comp_mean(imglist):
    mean = [0, 0, 0]
    for img in imglist:
        mean += np.mean(np.mean(img, axis=0), axis=0)
    return mean/len(imglist)

def find_f_measure_threshold2(probs, labels, num_iters=100, seed=0.21):
    _, num_classes = labels.shape[0:2]
    best_thresholds = [seed] * num_classes
    best_scores = [0] * num_classes
    for t in range(num_classes):

        thresholds = list(best_thresholds)  # [seed]*num_classes
        for i in range(num_iters):
            th = i / float(num_iters)
            thresholds[t] = th
            f2 = fbeta_score(labels, probs > thresholds, beta=2, average='samples')
            if f2 > best_scores[t]:
                best_scores[t] = f2
                best_thresholds[t] = th
        print('\t(t, best_thresholds[t], best_scores[t])=%2d, %0.3f, %f' % (t, best_thresholds[t], best_scores[t]))
    print('')
    return best_thresholds, best_scores


def normalize(img):
    img = img.astype(np.float16)

    img[:, :, 0] = (img[:, :, 0] - 103.94) * 0.017
    img[:, :, 1] = (img[:, :, 1] - 116.78) * 0.017
    img[:, :, 2] = (img[:, :, 2] - 123.68) * 0.017
    # img = np.expand_dims(img, axis=0)
    return img


def transformations(src, choice):
    if choice == 0:
        # Rotate 90
        src = cv2.rotate(src, rotateCode=cv2.ROTATE_90_CLOCKWISE)
    if choice == 1:
        # Rotate 90 and flip horizontally
        src = cv2.rotate(src, rotateCode=cv2.ROTATE_90_CLOCKWISE)
        src = cv2.flip(src, flipCode=1)
    if choice == 2:
        # Rotate 180
        src = cv2.rotate(src, rotateCode=cv2.ROTATE_180)
    if choice == 3:
        # Rotate 180 and flip horizontally
        src = cv2.rotate(src, rotateCode=cv2.ROTATE_180)
        src = cv2.flip(src, flipCode=1)
    if choice == 4:
        # Rotate 90 counter-clockwise
        src = cv2.rotate(src, rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE)
    if choice == 5:
        # Rotate 90 counter-clockwise and flip horizontally
        src = cv2.rotate(src, rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE)
        src = cv2.flip(src, flipCode=1)
    return src

def transformations2(src, choice):
    mode = choice // 2
    src = imutils.rotate(src, mode * 90)
    if choice % 2 == 1:
        src = cv2.flip(src, flipCode=1)
    return src

def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.uniform(hue_shift_limit[0], hue_shift_limit[1])
        h = cv2.add(h, hue_shift)
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image

def randomShiftScaleRotate(image, mask,
                           shift_limit=(-0.0625, 0.0625),
                           scale_limit=(-0.1, 0.1),
                           rotate_limit=(-45, 45), aspect_limit=(0, 0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5, factor=1):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])  # degree
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))

    return image, mask


def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask


def randomGammaCorrection(image, u=0.5):
    if np.random.random() < u:
        lower = 0.75
        upper = 1.25
        mu = 1
        sigma = 0.25
        alpha = stats.truncnorm((lower-mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
        image = (pow(image/255.0, alpha.rvs(1)[0]) * 255).astype(np.uint8)
    return image

def randomSymmetricPadding(image, mask, u=0.5):
    if np.random.random() < u:
        orig_width = image.shape[1]
        image = np.pad(image, ((0, 0), (image.shape[1], image.shape[1]), (0, 0)), 'symmetric')
        mask = np.pad(mask, ((0, 0), (mask.shape[1], mask.shape[1])), 'symmetric')

        start = np.random.randint(0, image.shape[1] - orig_width)
        image = image[:, start: start + orig_width, :]
        mask = mask[:, start: start + orig_width]

    return image, mask

def random_crop(img, dstSize=(INPUT_HEIGHT, INPUT_WIDTH), center=False):
    import random
    if img.ndim < 4:
        srcH, srcW = img.shape[:2]
    else:
        srcH, srcW = img.shape[1:3]

    dstH, dstW = dstSize

    if srcH <= dstH or srcW <= dstW:
        return img
    if center:
        y0 = int((srcH - dstH) / 2)
        x0 = int((srcW - dstW) / 2)
    else:
        y0 = random.randrange(0, srcH - dstH)
        x0 = random.randrange(0, srcW - dstW)

    if img.ndim < 4:
        return img[y0:y0+dstH, x0:x0+dstW]
    else:
        return img[:, y0:y0+dstH, x0:x0+dstW, :]

def transform(*pair, center=False, padding=20, dstSize = (INPUT_HEIGHT, INPUT_WIDTH)):
    def reflectivePaddingAndCrop():
        result = []
        for image in pair:
            if image.ndim == 3:
                image = np.pad(image, ((padding, padding), (padding, padding), (0, 0)), 'reflect')
            else:
                image = np.pad(image, ((padding, padding), (padding, padding)), 'reflect')
            result.append(random_crop(image, dstSize=dstSize, center=center))
        return result

    def zeroPaddingAndCrop():
        result = []
        for image in pair:
            if image.ndim == 3:
                image = np.pad(image, ((padding, padding), (padding, padding), (0, 0)), 'constant', constant_values=((0, 0), (0, 0), (0, 0)))
            else:
                image = np.pad(image, ((padding, padding), (padding, padding)), 'constant', constant_values=((0, 0), (0, 0)))
            result.append(random_crop(image, dstSize=dstSize, center=center))
        return result

    def resizeAndCrop():
        result = []
        for image in pair:
            image = cv2.resize(image, (dstSize[1] + padding, dstSize[0] + padding), interpolation=cv2.INTER_LINEAR)
            result.append(random_crop(image, dstSize=dstSize, center=center))
        return result

    def resize():
        result = []
        for image in pair:
            image = cv2.resize(image, (dstSize[1], dstSize[0]), interpolation=cv2.INTER_LINEAR)
            result.append(image)
        return result

    if MODE == PADCROPTYPE.ZERO:
        return zeroPaddingAndCrop()
    elif MODE == PADCROPTYPE.RECEPTIVE:
        return reflectivePaddingAndCrop()
    elif MODE == PADCROPTYPE.RESIZE:
        return resizeAndCrop()
    elif MODE == PADCROPTYPE.NONE:
        return resize()


def run_length_encode(mask):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    inds = mask.flatten()
    runs = np.where(inds[1:] != inds[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    rle = ' '.join([str(r) for r in runs])
    return rle


def dice_score(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    # return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)

def dice_loss(y_true, y_pred):
    return 1. - dice_score(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

def weightedBCE(y_true, y_pred):
    mask = np.zeros((INPUT_HEIGHT, INPUT_HEIGHT))
    srcH, srcW = INPUT_HEIGHT, INPUT_HEIGHT
    dstH, dstW = ORIG_HEIGHT, ORIG_WIDTH

    y0 = int((srcH - dstH) / 2)
    x0 = int((srcW - dstW) / 2)

    mask[y0:y0+dstH, x0:x0+dstW] = 1

    crop = K.binary_crossentropy(y_true, y_pred) * mask
    return K.mean(crop, axis=-1)

def weightedBCELoss2d(y_true, y_pred, weights):
    w = K.flatten(weights)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    loss = w * y_pred_f * (1-y_true_f) + w * K.log(1+K.exp(-y_pred_f))
    return K.sum(loss)/K.sum(weights)

def weightedSoftDiceLoss(y_true, y_pred, weights):
    smooth = 1.
    w = K.flatten(weights)
    w2 = w * w
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(w2 * y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (K.sum(w2*y_true_f) + K.sum(w2*y_pred_f) + smooth)

def weightedLoss(y_true, y_pred):
    a = K.pool2d(y_true, (11,11), strides=(1, 1), padding='same', data_format=None, pool_mode='avg')
    ind = K.cast(K.greater(a, 0.01), dtype='float32') * K.cast(K.less(a, 0.99), dtype='float32')

    weights = K.cast(K.greater_equal(a, 0), dtype='float32')
    w0 = K.sum(weights)
    # w0 = weights.sum()
    weights = weights + ind * 2
    w1 = K.sum(weights)
    # w1 = weights.sum()
    weights = weights / w1 * w0
    return weightedBCELoss2d(y_true, y_pred, weights) + weightedSoftDiceLoss(y_true, y_pred, weights)

def focal_loss_1(y_true, y_pred, gamma=2.):
    # y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

    epsilon = K.epsilon()

    y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)

    loss = K.pow(1.0 - y_pred, gamma)

    loss = - K.sum(loss * y_true * K.log(y_pred), axis=-1)
    return loss

## https://www.kaggle.com/c/carvana-image-masking-challenge/discussion/39951
def focal_loss(gamma=2, alpha=0.5):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()

        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)  # improve the stability of the focal loss and see issues 1 for more information
        # pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        # pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        # return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
        return -K.mean(
            alpha * y_true * K.pow(1. - y_pred, gamma) * K.log(y_pred) +
        (1-alpha) * (1 - y_true) * K.pow(y_pred, gamma) * K.log(1. - y_pred))

    return focal_loss_fixed

def focal_dice_loss(y_true, y_pred):
    return focal_loss()(y_true, y_pred) + dice_loss(y_true, y_pred)

def get_score(train_masks, avg_masks, thr):
    d = 0.0
    for i in range(train_masks.shape[0]):
        pred_mask = avg_masks[i]
        # pred_mask = avg_masks[i][:,:,1] - avg_masks[i][:,:,0]
        pred_mask[pred_mask > thr] = 1
        pred_mask[pred_mask <= thr] = 0
        d += dice_loss(train_masks[i], pred_mask)
    return d/train_masks.shape[0]


def get_result(imgs, thresh):
    result = []
    for img in imgs:
        img[img > thresh] = 1
        img[img <= thresh] = 0
        result.append(cv2.resize(img, (1918, 1280), interpolation=cv2.INTER_LINEAR))
    return result

def get_final_mask(preds, thresh=0.5, apply_crf=False, images=None):
    results = []
    probs = []
    for i in range(len(preds)):
        pred = preds[i]

        prob = cv2.resize(pred, (ORIG_WIDTH, ORIG_HEIGHT))
        probs.append(prob)
        if apply_crf and images is not None and len(images) > 0:
            image = images[i]
            prob = np.dstack((prob,) * 2)
            prob[..., 0] = 1 - prob[..., 1]
            mask, _ = denseCRF(image, prob)
        else:
            mask = prob > thresh
        results.append(mask)
    return results, probs

# def find_best_seg_thr(y_valid, pred_valid):
#     best_score = 0
#     best_thr = -1
#     thresholds = np.linspace(0.5, 0.95, 10)
#     ious = np.array(
#         [get_score(y_valid, pred_valid, threshold) for threshold in thresholds])
#     print(ious[0])
#     print("foo: ", ious.shape, type(ious))
#     threshold_best_index = np.argmax(ious) # np.argmax(ious[9:-10]) + 9
#     iou_best = ious[threshold_best_index]
#     threshold_best = thresholds[threshold_best_index]
#
#     print("IOU:\n", ious)
#     return iou_best, threshold_best

def find_best_seg_thr(masks_gt, masks_pred):
    best_score = 0
    best_thr = -1
    for thr in np.linspace(0.5, 0.95, 10):
        score = get_score(masks_gt, masks_pred, thr)
        print('THR: {:.3f} SCORE: {:.6f}'.format(thr, score))
        if score > best_score:
            best_score = score
            best_thr = thr

    print('Best score: {} Best thr: {}'.format(best_score, best_thr))
    return best_score, best_thr

def denseCRF(image, final_probabilities):

    # softmax = final_probabilities.squeeze()

    softmax = final_probabilities.transpose((2, 0, 1))

    # The input should be the negative of the logarithm of probability values
    # Look up the definition of the softmax_to_unary for more information
    unary = unary_from_softmax(softmax)

    # The inputs should be C-continious -- we are using Cython wrapper
    unary = np.ascontiguousarray(unary)

    d = dcrf.DenseCRF2D(image.shape[1], image.shape[0], 2)
    # d = dcrf.DenseCRF(image.shape[0] * image.shape[1], 2)

    d.setUnaryEnergy(unary)

    d.addPairwiseBilateral(sxy=50, srgb=13, rgbim=image, compat=3)
    Q = d.inference(5)

    res = np.argmax(Q, axis=0).reshape((image.shape[0], image.shape[1]))

    return res,Q


def numpy_dice_score(y_true, y_pred):
    smooth = 1.
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def numpy_dice_loss(y_true, y_pred):
    return 1. - numpy_dice_score(y_true, y_pred)


# src: https://www.kaggle.com/aglotero/another-iou-metric
def iou_metric(y_true_in, y_pred_in, print_table=False):
    labels = y_true_in
    y_pred = y_pred_in

    true_objects = 2
    pred_objects = 2

    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins=true_objects)[0]
    area_pred = np.histogram(y_pred, bins=pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:, 1:]
    union = union[1:, 1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1  # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)

    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)


def iou_metric_batch(y_true_in, y_pred_in):
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.mean(metric)


def find_best_threshold(ious, thresholds = np.linspace(0, 1, 50)):
    threshold_best_index = np.argmax(ious[9:-10]) + 9
    iou_best = ious[threshold_best_index]
    threshold_best = thresholds[threshold_best_index]

    return iou_best, threshold_best

def evaluate_ious(y_valid, pred_valid):
    thresholds = np.linspace(0, 1, 50)
    ious = np.array(
        [iou_metric_batch(y_valid, np.int32(pred_valid > threshold)) for threshold in thresholds])
    return ious


# Source https://www.kaggle.com/bguberfain/unet-with-depth
def RLenc(img, order='F', format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not

    returns run length as an array or string (if format is True)
    """
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []  ## list of run lengths
    r = 0  ## the current run length
    pos = 1  ## count starts from 1 per WK
    for c in bytes:
        if (c == 0):
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    # if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''

        for rr in runs:
            z += '{} {} '.format(rr[0], rr[1])
        return z[:-1]
    else:
        return runs

def upsample(img):
    if ORIG_HEIGHT == INPUT_HEIGHT and ORIG_WIDTH == INPUT_WIDTH:
        return img
    return resize(img, (INPUT_WIDTH, INPUT_HEIGHT), mode='constant', preserve_range=True)
    # res = np.zeros((img_size_target, img_size_target), dtype=img.dtype)
    # res[:img_size_ori, :img_size_ori] = img
    # return res

def downsample(img):
    if ORIG_HEIGHT == INPUT_HEIGHT and ORIG_WIDTH == INPUT_WIDTH:
        return img
    return resize(img, (ORIG_WIDTH, ORIG_HEIGHT), mode='constant', preserve_range=True)
    # return img[:img_size_ori, :img_size_ori]


# from keras.layers import ZeroPadding2D
#
# # Extending the ZeroPadding2D layer to do reflection padding instead.
# class ReflectionPadding2D(ZeroPadding2D):
#     def call(self, x, mask=None):
#         pattern = [[0, 0],
#                    [self.top_pad, self.bottom_pad],
#                    [self.left_pad, self.right_pad],
#                    [0, 0]]
#         return tf.pad(x, pattern, mode='REFLECT')
#
# class SymmetricPadding2D(ZeroPadding2D):
#     def call(self, x, mask=None):
#         pattern = [[0, 0],
#                    [self.top_pad, self.bottom_pad],
#                    [self.left_pad, self.right_pad],
#                    [0, 0]]
#         return tf.pad(x, pattern, mode='SYMMETRIC')