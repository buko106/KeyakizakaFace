# -*- coding: utf-8 -*-
import cv2
import os
import time
import numpy
import math

def check(img,cascade_f):
    r , c , _ = img.shape
    ratio = 200. / max(r,c)
    resized = cv2.resize(img, (int(ratio*c),int(ratio*r))) 

    #
    # cv2.imshow("hoge",resized)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #

    rows, cols , color = resized.shape
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    # 元画像の斜辺サイズの枠を作る(0で初期化)
    hypot = int(math.hypot(rows, cols))
    frame = numpy.zeros((hypot, hypot), numpy.uint8)
    frame[(hypot - rows) * 0.5:(hypot + rows) * 0.5, (hypot - cols) * 0.5:(hypot + cols) * 0.5] = gray
    # 各loopで違う角度の回転行列をかけた結果のものに対して検出を試みる
    for deg in range(-28, 29, 7):
        M = cv2.getRotationMatrix2D((hypot * 0.5, hypot * 0.5), -deg, 1.0)
        rotated = cv2.warpAffine(frame, M, (hypot, hypot))
        faces = cascade_f.detectMultiScale(rotated)
        for (x, y, w, h) in faces:
            cv2.rectangle(rotated, (x, y), (x + w, y + h), (0, 0, 0), 2)
            
        # 画像表示
        cv2.imshow(file+"deg="+str(deg)+")",rotated)
        cv2.moveWindow(file,0,0)
        
        # 何かキーを押したら終了
        cv2.waitKey(100)
        cv2.destroyAllWindows()


# Haar-like特徴分類器の読み込み
#face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt2.xml')

sourcedir = "data/watanaberika/"
path2file = sourcedir.rstrip("/")
files = os.listdir(path2file)

for file in files:
    # イメージファイルの読み込み
    img = cv2.imread(path2file+"/"+file)
    # グレースケール変換
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 顔を検知
    # face = face_cascade.detectMultiScale(gray)
    face = []
    for (x,y,w,h) in face:
        # 検知した顔を矩形で囲む
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        # 顔画像（グレースケール）
        roi_gray = gray[y:y+h, x:x+w]
        # 顔画像（カラースケール）
        roi_color = img[y:y+h, x:x+w]
            
    # 画像表示
    #cv2.imshow(file,img)
    cv2.moveWindow(file,0,0)
    
    # 何かキーを押したら終了
    cv2.waitKey(1500)
    cv2.destroyAllWindows()

    # 回転
    check(cv2.imread(path2file+"/"+file),face_cascade)
