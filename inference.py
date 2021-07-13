# -*- coding: utf-8 -*-
import numpy as np

from config import IMAGE_PATH
from loader import Loader
from model_util import ModelUtil


def main() -> None:
    x = Loader.load_target_file(IMAGE_PATH)
    model = ModelUtil.load_trained_model()

    predict = model.predict(x)
    print('predict label: {}'.format(np.argmax(predict[0])))


if __name__ == '__main__':
    main()
