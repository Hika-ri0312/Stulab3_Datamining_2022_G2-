# 浮世絵と洋風絵を分類してみる。

## 概要

2022知能情報特別講義3 Group2 の成果物である.
浮世絵と洋風絵の画像データから、浮世絵と洋風絵を予想する.

## 使用したデータセット

- [arc-ukiyoe-faces](https://github.com/rois-codh/arc-ukiyoe-faces/)
- [metmuseumopenacces](https://github.com/metmuseum/openaccess)

動作環境

- Python 3.x
- scikit-learn
- Tensorflow
- Keras

- pandas 1.3.5

## セットアップ

ソースのクローン

```$ git clone https://github.com/Hika-ri0312/Stulab3_Datamining_2022_G2- ```

上記 ```git clone```によりダンロードしたフォルダを、任意にディレクトリに保存する.

```
Programs #任意のディレクトリ
├── dataset
├── cnn
├── svm
├── randomforest

```


上記 **使用したデータセット** によってダウンロードしたフォルダあるいは画像データを、


```
├── dataset
    |
    ├── bfo_data
        |
        ├── images
	|  ├──#洋風絵の画像ファイル
        |
        └── arc-ukiyoe-faces #ARC浮世絵よりダウンロードしたフォルダ

```


## 実行方法

- 前準備

本学習モデルでは、データの前処理を行うプログラムを実行する必要がある.
前処理は全3工程存在する.(3工程に分割した理由は開発者の都合による)

- 手順1 浮世絵の画像データをpd.DataFrame()型に格納
**Programs**
- ```$ python dataset/program/src/converting_ukiyoe_image_to_gray_size.py ```

- 手順2 洋風絵の画像データをpd.DataFrame()型に格納
**Programs**
- ```$ python dataset/program/src/converting_western_image_to_gray_size.py ```

- 手順3 浮世絵と洋風絵に教師データを結合する
**Programs**
- ```$ python dataset/program/src/image_and_title_to_pickle_format.py ```


```
# SVM
$ python svm/program/src/svc.py
```

```
# LinearSVC
$ python svm/program/src/linear_svc.py
```

```
# RandomForest
$ python svm/program/src/randomforest.py
```

- ※ CNNは、Google Colab での実装のため、cnn.ipynbおよび、ダウンロードした画像データを下記のような配置にすることで実行できる.

学習データと画像

```
temp #任意のディレクトリ
├── program/src/
    ├── cnn
        ├── ukio_west
        │   ├──train
        │      ├──ukio (浮世絵の学習用の画像ファイル)
        │      ├──west (洋風絵の学習用の画像ファイル)
        │
        ├── validation
	    ├──ukio (浮世絵のテスト用の画像ファイル)
	    ├──west (洋風絵のテスト用の画像ファイル)
```

## Author

氏名: Choi Jeongho
連絡先: 185739@ie.u-ryukyu.ac.jp

氏名: 石原 光竜
連絡先: 205217@ie.u-ryukyu.ac.jp
