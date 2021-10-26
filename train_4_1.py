import sys
import os
from torch.optim.lr_scheduler import StepLR
hidden_size = int(sys.argv[1])
num_layers = int(sys.argv[2])
len_threshold=int(sys.argv[3])
name = str(sys.argv[4])
os.environ["CUDA_VISIBLE_DEVICES"]=str(sys.argv[5])
import torch
torch.backends.cudnn.benchmark = True
from reader import getVideoList
import numpy as np
from model import Attention
import torch
from worker import SeqWorker
import pandas as pd
if __name__=='__main__':    
    print("Load training data")
    train_lengths = list(pd.read_csv('hw4_data/video_0_5_12/trimmed_train_full.txt',header=None)[0])
    rescale_train_x = np.load('hw4_data/video_0_5_12/trimmed_train_full.npy')
    train_y = getVideoList('hw4_data/TrimmedVideos/label/gt_train.csv')['Action_labels']
    train_y = np.array(train_y).astype('int')
    print("Load validation data")
    rescale_val_x = np.load('hw4_data/video_0_5_12/trimmed_val_full.npy')
    val_lengths = list(pd.read_csv('hw4_data/video_0_5_12/trimmed_val_full.txt',header=None)[0])
    val_y = getVideoList('hw4_data/TrimmedVideos/label/gt_valid.csv')['Action_labels']
    val_y = np.array(val_y).astype('int')
    classifier = Attention(hidden=hidden_size,num_layers=num_layers).cuda()
    worker = SeqWorker()
    worker.batch_size=1
    worker.epoch=20
    stop_value=50
    optim = torch.optim.Adam(classifier.parameters(),lr=1e-5,amsgrad=True)
    scheduler = None
    print("Start training")
    record = worker.train(classifier,optim,
                          rescale_train_x,train_y,
                          rescale_val_x,val_y,
                          train_lengths,val_lengths,
                          len_threshold=len_threshold,
                          scheduler=scheduler,stop_value=stop_value)
    name = name+"_"+str(record['val_accuracy'][-1])
    if record['val_accuracy'][-1]>=stop_value:
        with open(name+"_record.txt","w") as fp:
            fp.write( str(record) )
        torch.save(classifier.state_dict(),name+'_classifier.pth')
        with open(name+"_setting.txt","w") as fp:
            fp.write( str(sys.argv) )