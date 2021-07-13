# coding: utf-8

from tensorflow.keras.models import Sequential, load_model

from config import SAVE_MODEL_PATH


class ModelUtil:

    @classmethod
    def save_model(cls, model: Sequential) -> None:
        """学習したモデルを保存する
        models.load_model('model.h5', compile=False)
        Args:
          model(Sequential): 学習済みのモデルオブジェクト
        """
        model.save(SAVE_MODEL_PATH)

    @classmethod
    def load_trained_model(cls) -> Sequential:
        """学習済みモデルをロードする
        Returns:
            model(Sequential): 学習済みのモデルオブジェクト
        """
        return load_model(SAVE_MODEL_PATH)
