import os
import sys

par_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(par_dir)

# プロジェクトのルートパスの設定
PROJECT_ROOT_PATH = par_dir

CNN_INPUT_SIZE = 28
CNN_CHANNEL = 1
CNN_OUTPUT_NUM = 10

BATCH_SIZE = 128
EPOCHS = 10

SAVE_MODEL_PATH = './data/trained_model.hdf5'
IMAGE_PATH = './images/test_image.jpg'
