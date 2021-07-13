# -*- coding: utf-8 -*-

from typing import Tuple

import cv2
import numpy as np
from numpy import ndarray
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.utils import to_categorical

from config import CNN_INPUT_SIZE, CNN_CHANNEL, CNN_OUTPUT_NUM


class Loader:

    @classmethod
    def load_mnist(cls) -> Tuple[Tuple[ndarray, ndarray], Tuple[ndarray, ndarray]]:
        """MNISTのデータを読み込み，CNNの入力に適した前処理を行う
        Returns:
            x_train(ndarray): 画像の教師データ
            y_train(ndarray): ラベルの教師データ
            x_test(ndarray): 画像のテストデータ
            y_test(ndarray): ラベルのテストデータ
        """
        (x_train, y_train), (x_test, y_test) = load_data()
        x_train = x_train.reshape(x_train.shape[0], CNN_INPUT_SIZE, CNN_INPUT_SIZE, CNN_CHANNEL)
        x_test = x_test.reshape(x_test.shape[0], CNN_INPUT_SIZE, CNN_INPUT_SIZE, CNN_CHANNEL)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        y_train = to_categorical(y_train, CNN_OUTPUT_NUM)
        y_test = to_categorical(y_test, CNN_OUTPUT_NUM)

        return (x_train, y_train), (x_test, y_test)

    @classmethod
    def load_target_file(cls, path: str) -> ndarray:
        """対象のファイルから画像を読み込む
        Args:
            path(str): 対象のファイルまでのパス
        Returns:
            x(ndarray): 読み込んだ画像
        """
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized_image = cv2.resize(image, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))
        x = np.expand_dims(resized_image, 0)
        x = np.expand_dims(x, 3)
        x = x.astype('float32')
        return x
