
import numpy as np
import os
from PIL import Image as Image
import cv2
from recog import CRNN_MODEL

crnn_model_path = './expr_model/crnn_Rec_done_2_0.7589375.pth'

crnn = CRNN_MODEL(crnn_model_path)

if __name__ == '__main__':
    path = '/dataset/hzl/personal_card/ctpn_model/data/result_detect_images/'
    image_list = os.listdir(path)
    f = open('./label_test.txt', 'w')
    for i, p in enumerate(image_list):
        image_path = os.path.join(path, p)
        image = Image.open(image_path)
        pred = crnn.rec(image)
        f.writelines(p + "\t" + pred + '\n')
    f.close()









