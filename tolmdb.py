# coding:utf-8
import os
import lmdb  # install lmdb by "pip install lmdb"
import cv2
import re
from PIL import Image
import numpy as np
import imghdr
import argparse
from tqdm import tqdm


def init_args():
    args = argparse.ArgumentParser()
    args.add_argument('-l',
                      '--label_file',
                      default='/media/hzl/file/Data-Set/360_data-set/label_test.txt',
                      type=str,
                      help='The file which contains the paths and the labels of the data set')
    args.add_argument('-s',
                      '--save_dir',
                      default='/media/hzl/file/Data-Set/360_data-set/lmdb_360_test/',
                      type=str
                      , help='The generated mdb file save dir')
    args.add_argument('-m',
                      '--map_size',
                      help='map size of lmdb',
                      type=int,
                      default=40000000000000)

    return args.parse_args()


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    try:
        imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
        img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
        imgH, imgW = img.shape[0], img.shape[1]
    except:
        return False
    else:
        if imgH * imgW == 0:
            return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            if type(k) == str:
                k = k.encode()
            if type(v) == str:
                v = v.encode()
            txn.put(k,v)

def createDataset(outputPath, imagePathList, labelList, map_size, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for CRNN training.
    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    assert (len(imagePathList) == len(labelList))
    nSamples = len(imagePathList)
    env = lmdb.open(outputPath, map_size=map_size)
    cache = {}
    cnt = 0
    for i in tqdm(range(nSamples)):
        imagePath = imagePathList[i].replace('\n', '').replace('\r\n', '')
        # print(imagePath)
        label = labelList[i]
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue
        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label
        if lexiconList:
            lexiconKey = 'lexicon-%09d' % cnt
            cache[lexiconKey] = ' '.join(lexiconList[i])
        if cnt != 0 and cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1

    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    env.close()
    print('Created dataset with %d samples' % nSamples)


if __name__ == '__main__':
    args = init_args()
    image_label = open(args.label_file, 'r')
    lines = list(image_label)
    imgPathList = []
    labelList = []
    k = 0
    for line in tqdm(lines):
        k += 1
        if k > 10000:
            break
        imgPath, word = line.strip('\n').strip().split()
        if not os.path.exists(imgPath):
            continue
        imgPathList.append(imgPath)
        labelList.append(word)
    createDataset(args.save_dir, imgPathList, labelList, args.map_size)






