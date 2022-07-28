""" 
浮世絵の画像をリサイズし、pandas.DataFrame()に格納してpickle形式で保存する.

This is main.py
""" 

import os
import pandas as pd
import numpy as np
import cv2

def load_fhoto_img():
    """ 浮世絵の画像ファイルを読み込み。1次元ベクトルにする。

    return:

    images (numpy.ndarray): 2重配列。画像データを1次元ベクトル化したものをまとめている

    Examples:
    
    >>> datam = load_fhoto_img()
    >>> type(datam)
    <class 'numpy.ndarray'>
    
    >>> len(datam.shape)
    2
     """
    path1 = "../bfo_data/arc-ukiyoe-faces-main/scratch/arc_images"
    # path内のデイレクトリ下にあるフォルダ名をリストで取得.
    folders = os.listdir(path1)

    # 画像データをリサイズし、images に格納
    images=resize(path1, folders)

    return images



def resize(path, folders):
    """ 画像ファイルの読み込み+リサイズをする。すべての画像を 64 * 64 に統一

    Args:

    path (str): 画像ファイルが格納されているディレクトリまでの相対パス

    folders (list): 画像ファイルの名前が拡張子込みで str型 で格納されている

    Return:

    image_np_2_dim (numpy.ndarray): 2重配列。画像データを1次元ベクトル化したものをまとめている

     """
    images=[]
    #画像を読み込み、リサイズを行う。
    for file in folders:
        img = cv2.imread(path+"/"+file)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray_e = cv2.resize(img_gray,dsize=(64,64)) #=> 4,096 の長さの1次元ベクトルになる
        

        images.append(img_gray_e)
    
    images_np=np.array(images)
    #リシェイプで2次元に
    images_np_2dim=images_np.reshape((images_np.shape[0], -1))

    return images_np_2dim



def get_plickle():
    """ numpy配列の画像データを、pandas.DataFrame()に格納し、pkl形式で保存する

    val:

    x (numpy.adarray): 2重配列。画像データを1次元ベクトル化したものをまとめている

    df (pandas.DataFrame): n行 * 1列のデータフレームで、1行ごとに画像データがnumpy配列で保存されている

     """
    x=load_fhoto_img()

    df=pd.DataFrame({"images": iter(x)})

    df.to_pickle("../bfo_data/ukiyoe_grayImg.pkl")
    
    

if __name__=="__main__":
    get_plickle()
