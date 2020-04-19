import numpy as np
import sys, os
import time
import cv2
import torch
from torch.autograd import Variable
from utils import  strLabelConverter
import net.crnn as crnn
from alphabets import alphabet
import params
from collections import Counter
from tqdm import tqdm
from dataset import *
import os
import traceback


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class CRNN_MODEL():
    def __init__(self, crnn_model_path):
        super(CRNN_MODEL, self).__init__()        
        self.converter = strLabelConverter(alphabet)
        nclass = len(alphabet) + 1
        self.model = crnn.CRNN(32, 1, nclass, 256)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        #self.model.load_state_dict(torch.load(crnn_model_path, map_location=torch.device('cpu')))
        self.model.load_state_dict(torch.load(crnn_model_path))
        self.model.eval()
        """
        self.converter = crnn1.utils.strLabelConverter(alphabet1)
        nclass = len(alphabet1) + 1
        self.model = crnn.CRNN(32, 1, nclass, 256)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.load_state_dict(torch.load(crnn_model_path))
        self.model.eval()
        """
    
    
    def rec(self, image):
        try:
            image = image.convert('L')
            w, h = image.size
            ratio = h / 32
            # w_now = int(w / (280 * 1.0 * ratio / 160))
            w_now = int(w / (1.0 * ratio))
            transform = resizeNormalize((w_now, 32))
            image = transform(image)
            if torch.cuda.is_available():
                image = image.cuda()
            image = image.view(1, *image.size())
            image = Variable(image)
            preds = self.model(image)
            _, preds = preds.max(2)
            #print("recog preds1: ", preds.shape)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            #print("recog preds2: ", preds.shape)
            preds_size = Variable(torch.IntTensor([preds.size(0)]))
            sim_pred = self.converter.decode(preds.data, preds_size.data, raw=False)
            return sim_pred
        except:
            traceback.print_exc()
            return ""
    

if __name__ == '__main__':
    pass











