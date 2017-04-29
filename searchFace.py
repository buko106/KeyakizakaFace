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

def cutImege( img , pos ):
    x , y , w , h = pos
    return img[y:y+h,x:x+w]

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


def detectFaceWithRotation(img,cascade_f,cascade_e,size=None,faceSize=None,isBGR=False,deglist=range(-20,21,5)):
    
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


    facesWithRank = []
    for deg in deglist:
        M = cv2.getRotationMatrix2D((hypot * 0.5, hypot * 0.5), deg, 1.0)
        rotated = cv2.warpAffine(frame, M, (hypot, hypot))
        # detect face
        
        faces = detectFaces(rotated,cascade_f)
        faces = [ (x,y,w,h) for (x,y,w,h) in faces if w*5 > cols and h*5 > rows ] # small face is eliminated
        if (not faces): # face not found
            continue
        """
        temp = numpy.copy(rotated)
        for (x, y, w, h) in faces:
            cv2.rectangle(temp, (x, y) , (x + w, y + h), (150, 150, 150), 2)

        checkImage(temp,wait=300)
        """
        

        for (x, y, w, h) in faces:
            face = rotated[y:y+h,x:x+w]
            eyes = detectEyes(face,cascade_e,faceSize)

            # eyes are in upper half of  face
            upper = [ (ex,ey,ew,eh) for (ex,ey,ew,eh) in eyes if ey+eh/2 <= y+h/2 and eh*10 >= h ]
            
            # for ( x , y , w , h ) in upper:
            #    cv2.rectangle(face, (x, y), (x + w, y + h), (0, 0, 0), 2)
            # print upper
            # checkImage(face,wait=200)

            if len(upper) == 0 :
                facesWithRank += [ ( 1.0 , numpy.abs(deg) , (x,y,w,h) , deg) ]
            elif len(upper) == 1 or len(upper) > 2 :
                facesWithRank += [ ( 0.5 , numpy.abs(deg) , (x,y,w,h) , deg) ]
            else :
                _ , y0 , _ , h0 = upper[0]
                _ , y1 , _ , h1 = upper[1]
                y0 = float(y0);y1 = float(y1);h0 = float(h0);h1 = float(h1)
                facesWithRank += [ (numpy.abs((y0+h0/2)-(y1+h1/2))/h , numpy.abs(deg) ,(x,y,w,h), deg )]
        
    facesWithRank.sort()
    # print facesWithRank
    if not facesWithRank : return None
    best = facesWithRank[0]
    _ , _ , (x,y,w,h) , deg = best
    return ( deg , (int(x/ratio),int(y/ratio),int(w/ratio),int(h/ratio)) )
    M = cv2.getRotationMatrix2D((hypot * 0.5, hypot * 0.5), deg, 1.0)
    rotated = cv2.warpAffine(frame, M, (hypot, hypot))
    # detect face
    checkImage( cutImege( rotated , best[2] ) , wait=500 )
    #if upper : checkImage(face,"Detected eyes",wait=300)
    # if faces == () : continue
    # checkImage(rotated,file+"deg="+str(deg)+")", wait=300)


def findAllImage(src):
    src = src.rstrip("/")
    for root , dir , files in os.walk(src):
        for file in files:
            yield os.path.join(root,file)[len(src)+1:]


desc="Detect faces from all image files under SRC"

parser = argparse.ArgumentParser(description=desc)
parser.add_argument("src",help="source directory")
parser.add_argument("dst",help="destination directory")
args = parser.parse_args()

# Import haar cascades
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')



# Image Files Path
source = args.src
destination = args.dst
# Get a list of all Images
for file in findAllImage(source):
    print "Read from : "+os.path.join(source,file)
    # イメージファイルの読み込み
    img = cv2.imread(os.path.join(source,file))
    if None == img : continue
    result = detectFaceWithRotation(img,face_cascade,eye_cascade,size=300,faceSize=200,isBGR=True,deglist=range(-35,36,5))
    if not result : continue
    else : deg , pos = result

    rows, cols , _ = img.shape
    hypot = int(math.hypot(rows, cols))
    M = cv2.getRotationMatrix2D((hypot * 0.5, hypot * 0.5), deg, 1.0)
    # cv2 image is type=numpy.ndarray of numpy.uint8
    frame = numpy.zeros((hypot, hypot,3), numpy.uint8)
    frame[ int((hypot - rows) * 0.5): int((hypot + rows) * 0.5) , int((hypot - cols) * 0.5) : int((hypot + cols) * 0.5) ] = img
    rotated = cv2.warpAffine(frame, M, (hypot, hypot))
    # x,y,w,h = pos ; cv2.rectangle( rotated , (x,y) , (x+w,y+h) , (numpy.random.randint(0,256),numpy.random.randint(0,256),numpy.random.randint(0,256)) , 2 )
    
    root, ext = os.path.splitext(os.path.join(destination,file) )
    imgPath = root + ".jpeg"
    path , name = os.path.split(imgPath)
    try:
        os.makedirs(path)
    except OSError :
        pass

    print "Write to : "+imgPath
    cv2.imwrite(imgPath,cv2.resize(cutImege(rotated,pos),(50,50)))
    
