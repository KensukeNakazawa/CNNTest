## CNNTest
計算機サーバーでPythonコードを実行するテストのためのテストコード

## ソフトウェアバージョン
- Ubuntu 20.04.1
- Python: 3.8.8
- Tensorflow: 2.4.1
- OpenCV: 4.5.2
- numpy: 1.18.5

## 実行方法

学習時  
> python train.py

予測時
> python inference.py

画像を変更する時はconfig.pyを以下の様に変更をする

```
・
・
・
# ここのパスを変更すること
IMAGE_PATH = './images/test_image.jpg'
```
