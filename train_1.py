import sys
import os
from torch.optim.lr_scheduler import StepLR
os.environ["CUDA_VISIBLE_DEVICES"]=str(sys.argv[1])
import torch
torch.backends.cudnn.benchmark = True
from reader import getVideoList
import numpy as np
from model import CNNClassifier
from worker import BasicWorker
if __name__=='__main__':    
    print("Load training data")
    rescale_train_x = np.load('hw4_data/video_0_5_12/trimmed_train.npy')
    train_y = getVideoList('hw4_data/TrimmedVideos/label/gt_train.csv')['Action_labels']
    train_y = np.array(train_y).astype('int')
    print("Load validation data")
    rescale_val_x = np.load('hw4_data/video_0_5_12/trimmed_val.npy')
    val_y = getVideoList('hw4_data/TrimmedVideos/label/gt_valid.csv')['Action_labels']
    val_y = np.array(val_y).astype('int')
    print("Start training")
    model = CNNClassifier().cuda()
    worker = BasicWorker()
    optim = torch.optim.Adam(model.parameters(),lr=1e-5)
    record = worker.train(model,optim,
                          rescale_train_x,train_y,
                          rescale_val_x,val_y)
    name = str(sys.argv[2])
    with open(name+"_record.txt","w") as fp:
        fp.write( str(record) )
    torch.save(model.state_dict(),name+'_classifier.pth')