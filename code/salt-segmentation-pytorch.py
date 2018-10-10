# # -*- coding: utf-8 -*-
# __author__ = 'Zhenyuan Shen: https://kaggle.com/szywind'
#
# from subprocess import check_output
# print(check_output(["ls", "../input"]).decode("utf8"))
#
# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# import os, math, shutil, time
# from keras.preprocessing.image import load_img
# from tqdm import tqdm, tqdm_notebook
#
# # from constants import *
# from helpers import *
# from sklearn.model_selection import train_test_split, StratifiedKFold
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from torch.autograd import Variable
# import unet_models
#
#
# np.set_printoptions(threshold='nan')
#
# class TGSSaltDataset(Dataset):
#     def __init__(self, df, is_test=False, data_augmentation=False):
#         self.is_test = is_test
#         self.df = df
#         self.data_augmentation = data_augmentation
#
#     def __len__(self):
#         return len(self.df)
#
#     def __getitem__(self, index):
#         if index not in range(0, len(self.df)):
#             return self.__getitem__(np.random.randint(0, self.__len__()))
#
#         img_name = self.df[index]
#         if self.is_test:
#             img = cv2.imread(os.path.join(INPUT_PATH, "test", "images", img_name + ".png"))  # [..., 0]
#             img = cv2.resize(img, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
#
#             if img.ndim == 2:
#                 img = np.expand_dims(img, axis=-1)  # equivalent to img[..., np.newaxis]
#
#             img = torch.from_numpy(img).float().permute([2, 0, 1]) / 255.0
#             return img
#
#         else:
#             img = cv2.imread(os.path.join(INPUT_PATH, "train", "images", img_name + ".png"))  # [..., 0]
#             img = cv2.resize(img, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
#             mask = cv2.imread(os.path.join(INPUT_PATH, "train", "masks", img_name + ".png"))[..., 0]
#             mask = cv2.resize(mask, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
#
#             if self.data_augmentation:
#                 img, mask = randomShiftScaleRotate(img, mask,
#                                                    shift_limit=(-0.08, 0.08),
#                                                    scale_limit=(0, 0.125),
#                                                    rotate_limit=(-0, 0))
#                 img, mask = randomHorizontalFlip(img, mask)
#                 img = randomGammaCorrection(img)
#                 # img = randomIntensityAugmentation(img)
#
#                 # img, mask = randomRotationAndFlip(img, mask)
#                 # draw(img, mask)
#
#             if img.ndim == 2:
#                 img = img[..., np.newaxis]
#             if mask.ndim == 2:
#                 mask = mask[..., np.newaxis]
#
#             img = torch.from_numpy(img).float().permute([2, 0, 1]) / 255.0
#             mask = torch.from_numpy(mask).float().permute([2, 0, 1]) / 255.0
#             return img, mask
#
# def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
#     torch.save(state, filename)
#     if is_best:
#         shutil.copyfile(filename, 'model_best.pth.tar')
#
#
# class AverageMeter(object):
#     """Computes and stores the average and current value"""
#     def __init__(self):
#         self.reset()
#
#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0
#
#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count
#
#
# def adjust_learning_rate(optimizer, epoch, initial_lr):
#     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#     lr = initial_lr * (0.1 ** (epoch // 30))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
#
#
# def accuracy(output, target, topk=(1,)):
#     """Computes the precision@k for the specified values of k"""
#     with torch.no_grad():
#         maxk = max(topk)
#         batch_size = target.size(0)
#
#         _, pred = output.topk(maxk, 1, True, True)
#         pred = pred.t()
#         correct = pred.eq(target.view(1, -1).expand_as(pred))
#
#         res = []
#         for k in topk:
#             correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
#             res.append(correct_k.mul_(100.0 / batch_size))
#         return res
#
# #https://github.com/pytorch/pytorch/issues/1249
# def dice_loss(m1, m2, is_average=True):
#     num = m1.size(0)
#     m1  = m1.view(num,-1)
#     m2  = m2.view(num,-1)
#     intersection = (m1 * m2)
#     scores = 2. * (intersection.sum(1)+1) / (m1.sum(1) + m2.sum(1)+1)
#     if is_average:
#         score = scores.sum()/num
#         return 1 - score
#     else:
#         return 1 - scores
#
# def mixed_dice_bce_loss(output, target, dice_weight=0.5, bce_weight=0.5,
#                         smooth=0, dice_activation='sigmoid'):
#     num_classes = output.size(1)
#     target = target[:, :num_classes, :, :].float()
#     bce_loss = nn.BCELoss()
#     return dice_weight * dice_loss(output, target) + bce_weight * bce_loss(output, target)
#
#
# class SaltSeg():
#     def __init__(self, input_width=INPUT_WIDTH, input_height=INPUT_HEIGHT, batch_size=32, learn_rate=1e-4):
#
#         self.input_width = input_width
#         self.input_height = input_height
#         self.batch_size = batch_size
#         self.learn_rate = learn_rate
#         self.nfolds = 5
#         self.tta = 2
#
#         # if MODEL_TYPE == MODEL.WRN:
#         #     self.model =
#         # elif MODEL_TYPE == MODEL.DPN:
#         #     self.model =
#         # elif MODEL_TYPE == RESNEXT:
#         #     self.model =
#
#         # self.model = unet_models.UNetResNet(34, 1, num_filters=32, dropout_2d=0.2,
#         #          pretrained=True, is_deconv=False)
#
#         # self.model = self.model.to("cuda")
#         # if train:
#         #     self.net_path = '../weights/model.json'
#         #     self.model_path = '../weights/salt-segmentation-model.h5'
#         #     with open(self.net_path, 'w') as json_file:
#         #         json_file.write(self.model.to_json())
#         # else:
#         #     # self.net_path = '../weights/{}/model.json'.format(MODEL_DIR)
#         #     # self.model_path = '../weights/{}/salt-segmentation-model.h5'.format(MODEL_DIR)
#         #
#         #     self.net_path = '../weights/model.json'
#         #     self.model_path = '../weights/salt-segmentation-model.h5'
#
#         self.load_data()
#         self.direct_result = True
#         self.train_with_all = False
#         self.apply_crf = False
#
#     # Load Data & Make Train/Validation Split
#     def load_data(self):
#         df_train = pd.read_csv(os.path.join(INPUT_PATH, "train.csv"), index_col="id", usecols=[0])
#         df_depths = pd.read_csv(os.path.join(INPUT_PATH, "depths.csv"), index_col="id")
#         self.df_train = df_train.join(df_depths)
#         self.df_test = df_depths[~df_depths.index.isin(df_train.index)]
#
#
#         df_train["masks"] = [np.array(load_img("../input/train/masks/{}.png".format(idx), grayscale=True)) / 255 for idx
#                              in tqdm_notebook(df_train.index)]
#
#         def cov_to_class(val):
#             for i in range(0, 11):
#                 if val * 10 <= i:
#                     return i
#
#         self.df_train["coverage"] = df_train.masks.map(np.sum) / float(self.input_width * self.input_height)
#         self.df_train["coverage_class"] = self.df_train.coverage.map(cov_to_class)
#
#         self.ids_test = self.df_test.index.values[:]
#
#         # stratified train/validation split by salt coverage
#         self.ids_train, self.ids_valid, self.coverage_train, self.coverage_test, self.depth_train, self.depth_test = train_test_split(
#             self.df_train.index.values,
#             self.df_train.coverage.values,
#             self.df_train.z.values,
#             test_size=0.2, stratify=self.df_train.coverage_class, random_state=37)
#
#
#     def train_cv(self):
#         fold_id = 0
#
#         # thres = []
#         # y_valid = None
#         # p_valid = None
#         ious = 0
#
#         # kf = KFold(n_splits=self.nfolds, shuffle=True, random_state=1)
#         # for train_index, test_index in kf.split(self.train_imgs):
#
#         skf = StratifiedKFold(n_splits=self.nfolds, random_state=37, shuffle=True)
#         for train_index, valid_index in skf.split(self.df_train.index.values, self.df_train.coverage_class.values):
#             if DEBUG and fold_id > 0:
#                 break;
#             self.ids_train = self.df_train.index.values[train_index]
#             self.coverage_train = self.df_train.coverage_class.values[train_index]
#             self.ids_valid = self.df_train.index.values[valid_index]
#             self.coverage_valid = self.df_train.coverage_class.values[valid_index]
#
#             # print(len(self.ids_train))
#             # for i in range(11):
#             #     print("i: ", sum(self.coverage_train == i))
#
#             print("Train {}-th fold".format(fold_id))
#             y_valid_i, p_valid_i = self.train(fold_id)
#             print("y_valid_i, p_valid_i:", y_valid_i.shape, p_valid_i.shape)
#
#             ious_i = evaluate_ious(y_valid_i, p_valid_i)
#             for iou in ious_i:
#                 print(iou)
#
#             ious += ious_i
#
#             del y_valid_i
#             del p_valid_i
#
#             # if y_valid is None:
#             #     y_valid = y_valid_i
#             #     p_valid = p_valid_i
#             # else:
#             #     y_valid = np.vstack([y_valid, y_valid_i])
#             #     p_valid = np.vstack([p_valid, p_valid_i])
#
#             fold_id += 1
#
#         ## find best threshold
#         # best_score, best_threshold = find_best_seg_thr(y_valid, p_valid)
#
#         # best_score, best_threshold = find_best_threshold(y_valid, p_valid)
#
#         best_score, best_threshold = find_best_threshold(ious)
#         print(best_score, best_threshold)
#
#         self.best_threshold = best_threshold
#
#
#
#     def train(self, fold_id=''):
#         # define loss function (criterion) and optimizer
#         learning_rate = 1e-4
#         criterion = mixed_dice_bce_loss
#
#         start_epoch = 0
#         best_prec1 = 0
#         p_valid = None
#
#         model = unet_models.UNetResNet(34, 1, num_filters=32, dropout_2d=0.2,
#                  pretrained=True, is_deconv=False)
#
#         optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
#
#
#         # load pretrained model
#         model_path = '../checkpoint/salt-segmentation-model{}.h5'.format(fold_id)
#         if os.path.isfile(model_path):
#             print("=> loading checkpoint '{}'".format(model_path))
#             checkpoint = torch.load(model_path)
#             start_epoch = checkpoint['epoch']
#             best_prec1 = checkpoint['best_prec1']
#             model.load_state_dict(checkpoint['state_dict'])
#             optimizer.load_state_dict(checkpoint['optimizer'])
#             print("=> loaded checkpoint '{}' (epoch {})"
#                   .format(model_path, checkpoint['epoch']))
#         else:
#             print("=> no checkpoint found at '{}'".format(model_path))
#
#         model.cuda()
#
#         nTrain = len(self.ids_train)
#         nValid = len(self.ids_valid)
#         print('Training on {} samples'.format(nTrain))
#         print('Validating on {} samples'.format(nValid))
#
#         ## Prepare Data
#         train_dataset = TGSSaltDataset(self.ids_train, is_test=False, data_augmentation=True)
#         valid_dataset = TGSSaltDataset(self.ids_valid, is_test=False, data_augmentation=False)
#
#         train_loader = DataLoader(dataset=train_dataset,
#                                   batch_size=self.batch_size,
#                                   shuffle=False,
#                                   num_workers=1)
#
#         valid_loader = DataLoader(dataset=valid_dataset,
#                                   batch_size=self.batch_size,
#                                   shuffle=False,
#                                   num_workers=1)
#
#         for epoch in range(start_epoch, EPOCHS):
#             adjust_learning_rate(optimizer, epoch, self.learn_rate)
#
#             # train for one epoch
#             self.train_one_epoch(model, train_loader, criterion, optimizer, epoch)
#
#             # evaluate on validation set
#             prec1 = self.validate(model, valid_loader, criterion)
#
#             # remember best prec@1 and save checkpoint
#             is_best = prec1 > best_prec1
#             best_prec1 = max(prec1, best_prec1)
#             save_checkpoint({
#                 'epoch': epoch + 1,
#                 'arch': MODEL_TYPE.name.lower() + str(fold_id),
#                 'state_dict': self.model.state_dict(),
#                 'best_prec1': best_prec1,
#                 'optimizer': optimizer.state_dict(),
#             }, is_best)
#
#
#
#         y_valid = []
#         for img_name in self.ids_valid:
#             # j = np.random.randint(self.nAug)
#             mask = cv2.imread(os.path.join(INPUT_PATH, "train", "masks", img_name + ".png"))[..., 0]
#             mask = cv2.resize(mask, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
#             # img = transformations2(img, j)
#             y_valid.append(mask)
#
#         y_valid = np.array(y_valid, np.float32)
#         return y_valid, p_valid
#
#     def test_cv(self):
#         if DEBUG:
#             pred_test = self.test(0)
#         else:
#             pred_test = 0
#             for fold_id in range(self.nfolds):
#                 pred_test_i = self.test(fold_id)
#                 pred_test += pred_test_i
#             pred_test /= float(self.nfolds)
#
#         pred_dict = {idx: RLenc(np.round(cv2.resize(pred_test[i], (ORIG_WIDTH, ORIG_HEIGHT), interpolation=cv2.INTER_LINEAR) > self.best_threshold))
#                      for i, idx in enumerate(tqdm_notebook(self.ids_test))}
#         sub = pd.DataFrame.from_dict(pred_dict, orient='index')
#         sub.index.names = ['id']
#         sub.columns = ['rle_mask']
#         sub.to_csv('submission.csv')
#
#     def test(self, fold_id=''):
#         self.model_path = '../weights/salt-segmentation-model{}.h5'.format(fold_id)
#
#         if not os.path.isfile(self.net_path) or not os.path.isfile(self.model_path):
#             raise RuntimeError("No model found.")
#
#         # json_file = open(self.net_path, 'r')
#         # loaded_model_json = json_file.read()
#         # self.model = model_from_json(loaded_model_json)
#         self.model.load_weights(self.model_path)
#
#         ## test
#         nTest = len(self.ids_test)
#         print('Testing on {} samples'.format(nTest))
#
#         if False:
#             def test_generator():
#                 while True:
#                     for start in range(0, nTest, self.batch_size):
#                         x_batch = []
#
#                         end = min(start + self.batch_size, nTest)
#                         ids_test_batch = self.ids_test[start:end]
#                         for img_name in ids_test_batch:
#
#                             img = cv2.imread(os.path.join(INPUT_PATH, "test", "images", img_name + ".png")) # [..., 0]
#                             img = cv2.resize(img, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
#
#                             if img.ndim == 2:
#                                 img = np.expand_dims(img, axis=-1)  # equivalent to img[..., np.newaxis]
#
#                             x_batch.append(img)
#                         x_batch = np.array(x_batch, np.float32) / 255.0
#                         yield x_batch
#
#             pred_test = self.model.predict_generator(generator=test_generator(),
#                                                   steps=math.ceil(nTest / float(self.batch_size)))
#
#             if USE_REFINE_NET:
#                 pred_test = pred_test[-1]
#         else:
#             nbatch = 0
#             IoU = 0
#             pred_test = np.zeros([nTest, INPUT_HEIGHT, INPUT_WIDTH, 1])
#             for start in range(0, nTest, self.batch_size):
#                 nbatch += 1
#                 x_batch = []
#
#                 end = min(start + self.batch_size, nTest)
#                 ids_test_batch = self.ids_test[start:end]
#                 for img_name in ids_test_batch:
#
#                     img = cv2.imread(os.path.join(INPUT_PATH, "test", "images", img_name + ".png")) # [..., 0]
#                     img = cv2.resize(img, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
#
#                     if img.ndim == 2:
#                         img = np.expand_dims(img, axis=-1)  # equivalent to img[..., np.newaxis]
#
#                     x_batch.append(img)
#
#                 x_batch = np.array(x_batch, np.float32) / 255.0
#
#                 p_test = self.model.predict(x_batch, batch_size=self.batch_size)
#
#                 if USE_REFINE_NET:
#                     p_test = p_test[-1]
#
#                 if self.tta > 0:
#                     p_test_flip = self.model.predict(x_batch[:, :, ::-1, :], batch_size=self.batch_size)
#
#                     if USE_REFINE_NET:
#                         p_test_flip = p_test_flip[-1]
#
#                     p_test = (p_test + p_test_flip[:, :, ::-1, :]) / float(self.tta)
#
#                 pred_test[start: end] = p_test
#
#         print(pred_test.shape)
#         return pred_test
#
#
#
#         # if self.direct_result:
#         #     result, probs = get_final_mask(p_test, self.threshold, apply_crf=self.apply_crf, images=images)
#         # else:
#         #     avg_p_test = p_test[...,1] - p_test[...,0]
#         #     result = get_result(avg_p_test, 0)
#         # str.extend(map(run_length_encode, result))
#
#         # for i in range(len(y_batch)):
#         #     IoU += numpy_dice_score(y_batch[i], result[i]) / nTest
#
#
#
#         # save predicted masks
#         # if not os.path.exists(OUTPUT_PATH):
#         #     os.mkdir(OUTPUT_PATH)
#         #
#         # for i in range(start, end):
#         #     image_path, mask_path = ids_test[i]
#         #     img_path = image_path[image_path.rfind('/')+1:]
#         #     res_mask = (255 * result[i-start]).astype(np.uint8)
#         #     res_mask = np.dstack((res_mask,)*3)
#         #     cv2.imwrite(OUTPUT_PATH + '{}'.format(img_path), res_mask)
#
#         # print('mean IoU: {}'.format(IoU))
#
#     def train_one_epoch(self, model, train_loader, criterion, optimizer, epoch):
#         batch_time = AverageMeter()
#         data_time = AverageMeter()
#         losses = AverageMeter()
#         iou = AverageMeter()
#
#         # switch to train mode
#         model.train()
#
#         end = time.time()
#         for i, (input, target) in enumerate(train_loader):
#             input, target = Variable(input).cuda(), Variable(target).cuda()
#             # input = input.type(torch.float).to("cuda")
#             # target = target.type(torch.float).to("cuda")
#
#             # measure data loading time
#             data_time.update(time.time() - end)
#
#             # compute output
#             output = model(input)
#             loss = criterion(output, target)
#
#             # measure accuracy and record loss
#             dice = dice_loss(output, target)
#             losses.update(loss.item(), input.size(0))
#             iou.update(dice[0], input.size(0))
#
#             # compute gradient and do SGD step
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             # measure elapsed time
#             batch_time.update(time.time() - end)
#             end = time.time()
#
#             if i % PRINT_FREQ == 0:
#                 print('Epoch: [{0}][{1}/{2}]\t'
#                       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#                       'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
#                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                       'IoU {iou.val:.3f} ({iou.avg:.3f})'.format(
#                     epoch, i, len(train_loader), batch_time=batch_time,
#                     data_time=data_time, loss=losses, iou=iou))
#
#
#     def validate(self, model, val_loader, criterion):
#         batch_time = AverageMeter()
#         losses = AverageMeter()
#         iou = AverageMeter()
#
#         # switch to evaluate mode
#         model.eval()
#
#         with torch.no_grad():
#             end = time.time()
#             for i, (input, target) in enumerate(val_loader):
#                 # if args.gpu is not None:
#                 #     input = input.cuda(args.gpu, non_blocking=True)
#                 # target = target.cuda(args.gpu, non_blocking=True)
#
#                 input, target = Variable(input).cuda(), Variable(target).cuda()
#                 # input = input.type(torch.float).to("cuda")
#                 # target = target.type(torch.float).to("cuda")
#
#                 # compute output
#                 output = model(input)
#                 loss = criterion(output, target)
#
#                 # measure accuracy and record loss
#                 dice = dice_loss(output, target)
#                 losses.update(loss.item(), input.size(0))
#                 iou.update(dice[0], input.size(0))
#
#                 # measure elapsed time
#                 batch_time.update(time.time() - end)
#                 end = time.time()
#
#                 if i % PRINT_FREQ == 0:
#                     print('Test: [{0}/{1}]\t'
#                           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#                           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                           'IoU {iou.val:.3f} ({iou.avg:.3f})'.format(
#                         i, len(val_loader), batch_time=batch_time, loss=losses, iou=iou))
#
#             print(' * IoU {iou.avg:.3f}'.format(iou=iou))
#
#         return iou.avg
#
# if __name__ == "__main__":
#     ccs = SaltSeg(input_width=INPUT_WIDTH, input_height=INPUT_HEIGHT)
#     if IS_TRAIN:
#         ccs.train_cv()
#     ccs.test_cv()