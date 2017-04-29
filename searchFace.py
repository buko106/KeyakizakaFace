# -*- coding: utf-8 -*-
import cv2
import os
import time
import numpy
import math
import argparse

def checkImage( img , winName="No Name" , pos=(0,0) , wait=0 ):
    cv2.imshow(winName,img)
    # left, upper
    x,y=pos
    cv2.moveWindow(winName,x,y)
    # Wait for wait msec
    cv2.waitKey(wait)
    # Destroy
    cv2.destroyAllWindows()

def detectEyes(face,cascade_eye,size=None,isBGR=False):
    
    r = face.shape[0]
    c = face.shape[1]

    if size!=None and r>size and c>size :
        # resized to size*(y<size) or (x<size)*size
        ratio = float(size) / max(r,c)
    else:
        ratio = 1.0

    resized = cv2.resize(face, (int(ratio*c),int(ratio*r))) 
        
    if isBGR:
        # convert to gray
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = resized

    eyes = cascade_eye.detectMultiScale(gray)
    ret = [ [ int(x/ratio) , int(y/ratio) , int(w/ratio) , int(h/ratio) ] for (x,y,w,h) in eyes ]
    return ret

def detectFaces(img,cascade_face,size=None,isBGR=False):
    
    r = img.shape[0]
    c = img.shape[1]

    if size!=None and r>size and c>size :
        # resized to size*(y<size) or (x<size)*size
        ratio = float(size) / max(r,c)
    else:
        ratio = 1.0

    resized = cv2.resize(img, (int(ratio*c),int(ratio*r))) 

    if isBGR:
        # convert to gray
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = resized

    faces = cascade_face.detectMultiScale(gray)
    ret = [ [ int(x/ratio) , int(y/ratio) , int(w/ratio) , int(h/ratio) ] for (x,y,w,h) in faces ]
    return ret


def detectFaceWithRotation(img,cascade_f,cascade_e,size=None,faceSize=None,isBGR=False):
    
    r = img.shape[0]
    c = img.shape[1]

    if size!=None and r>size and c>size :
        # resized to size*(y<size) or (x<size)*size
        ratio = float(size) / max(r,c)
    else:
        ratio = 1.0

    resized = cv2.resize(img, (int(ratio*c),int(ratio*r))) 

    # convert to gray
    if isBGR:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = resized

    rows, cols = gray.shape
    # 元画像の斜辺サイズの枠を作る(0で初期化)
    hypot = int(math.hypot(rows, cols))
    # cv2 image is type=numpy.ndarray of numpy.uint8
    frame = numpy.zeros((hypot, hypot), numpy.uint8)
    frame[ int((hypot - rows) * 0.5): int((hypot + rows) * 0.5) , int((hypot - cols) * 0.5) : int((hypot + cols) * 0.5) ] = gray

    for deg in range(-28, 29, 7):
        M = cv2.getRotationMatrix2D((hypot * 0.5, hypot * 0.5), deg, 1.0)
        rotated = cv2.warpAffine(frame, M, (hypot, hypot))
        # detect face
        faces = detectFaces(rotated,cascade_f)

        for (x, y, w, h) in faces:
            face = rotated[y:y+h,x:x+w]
            eyes = detectEyes(face,cascade_e,faceSize)
            for ( x , y , w , h ) in eyes:
                cv2.rectangle(face, (x, y), (x + w, y + h), (0, 0, 0), 2)
            if eyes :
                checkImage(face,"Detected eyes",wait=100)
        # if faces == () : continue
        # checkImage(rotated,file+"deg="+str(deg)+")", wait=300)


def findAllImage(src):
    for root , dir , files in os.walk(src):
        for file in files:
            yield os.path.join(root,file)


desc="Detect faces from all image files under SRC"

parser = argparse.ArgumentParser(description=desc)
parser.add_argument("src",help="source directory")
parser.add_argument("dst",help="destination directory")
args = parser.parse_args()

# Import haar cascades
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')



# Image Files Path
sourcedir = args.src
# Get a list of all Images
for file in findAllImage(sourcedir):
    print file
    # イメージファイルの読み込み
    img = cv2.imread(file)

    detectFaceWithRotation(cv2.imread(file),face_cascade,eye_cascade,size=200,faceSize=200,isBGR=True)
