from easydict import EasyDict as edict



__C = edict()
cfg = __C


## Set training options 
__C.DATA_DIR = './data/cub'
__C.SAVE_DIR = './output/cub'
__C.MODEL_PATH = ''
__C.NUM_GT_CLASSES = 200
__C.OVER = 1
__C.BATCH_SIZE = 64
__C.TEMP = 0.1
__C.PERT = 's'
__C.MAX_EPOCH = 1000
__C.EVAL_INTERVAL = 20


## Model options
__C.GAN = edict()
__C.GAN.Z_DIM = 64
__C.GAN.CZ_DIM = 8
