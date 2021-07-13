# -*- coding: utf-8 -*-

from config import BATCH_SIZE, EPOCHS
from loader import Loader
from model_util import ModelUtil
from network import Network


def main():
    model = Network.build()
    (x_train, y_train), (x_test, y_test) = Loader.load_mnist()

    model.fit(
        x_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1,
        validation_data=(x_test, y_test)
    )

    ModelUtil.save_model(model)


if __name__ == '__main__':
    main()
