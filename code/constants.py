from enum import Enum
class MODEL(Enum):
    UNET = 0
    TIRAMISUNET = 1
    REFINED_UNET = 2
    PSPNET2 = 3
    RESNET = 4
    DENSENET = 5
    INCEPTION = 6


INPUT_PATH = '../input/'
OUTPUT_PATH = '../output/test-result/'

##------------------- Padding Mode ----------------------------
class PADCROPTYPE(Enum):
    ZERO = 0
    RECEPTIVE = 1
    RESIZE = 2
    NONE = 3
MODE = PADCROPTYPE.RECEPTIVE

##-------------------- Configurations ------------------------
ORIG_WIDTH = ORIG_HEIGHT = 101
MIDDLE_WIDTH = MIDDLE_HEIGHT = 192
INPUT_WIDTH = INPUT_HEIGHT = 224
PAD_FRONT = 16
PAD_END = 16

BN_SIZE = 16
EPOCHS = 200
CV_FOLD = 10
N_TTA = 2
NUM_CLASS = 2 # background / foreground till now
PRINT_FREQ = 2
STRATIFIED_BY_COVERAGE = False

M = 4  # number of snapshots
SNAPSHOT_ENSEMBLING = False
##------------------- Setup Dataset ------------------------------
# train/test dataset for head segmentation
TRAIN_DATASET = "trainSet.txt"
TEST_DATASET = "testSet.txt"

NET_FILE = "../weights/model.json"

# train/test dataset for portrait segmentation
# TRAIN_DATASET = "trainSet-0.9-v2.3u.txt"
# TEST_DATASET = "testSet-0.9-v2.3u.txt"

## ----------------- Setup Model ----------------------------------
## ----------------- pspnet2 -----------------
# USE_REFINE_NET = False
# MODEL_DIR = "koutou_tf_180321"
# MODEL_TYPE = MODEL.PSPNET2
## ----------------- unet ---------------------
# USE_REFINE_NET = False
# MODEL_DIR = "180811"
# MODEL_TYPE = MODEL.UNET

# ##----------------- tiramisuNet --------------
# USE_REFINE_NET = False
# MODEL_DIR = "koutou_tf_180123"
# MODEL_TYPE = MODEL.TIRAMISUNET

## ----------------- refinenet -----------------
# USE_REFINE_NET = True
# MODEL_DIR = "koutou_tf_180211"
# MODEL_TYPE = MODEL.REFINED_UNET

## ----------------- resnet -----------------
# USE_REFINE_NET = False
# MODEL_DIR = "koutou_tf_180211"
# MODEL_TYPE = MODEL.RESNET

## ----------------- densenet -----------------
USE_REFINE_NET = False
MODEL_DIR = "koutou_tf_180211"
MODEL_TYPE = MODEL.RESNET

## ----------------- inception -----------------
# USE_REFINE_NET = False
# MODEL_DIR = "koutou_tf_180211"
# MODEL_TYPE = MODEL.INCEPTION

## ---------- Configure train + test or test only ------------------
IS_TRAIN = True
DEBUG = False
START = 8
DEFAULT_THRESHOLD = 0.5