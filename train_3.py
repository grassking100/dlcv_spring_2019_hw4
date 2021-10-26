import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"]=str(sys.argv[1])
import torch
torch.backends.cudnn.benchmark = True
import numpy as np
from model import CRClassifier
import torch
from utils import load_full_videos
from worker import SeqWorker
if __name__=='__main__':
    print("Load training data")
    train_video,train_label,train_length = load_full_videos('hw4_data/FullLengthVideos/videos/train',
                                                      'hw4_data/FullLengthVideos/labels/train')
    print("Load validation data")
    val_video,val_label,val_length = load_full_videos('hw4_data/FullLengthVideos/videos/valid',
                                                      'hw4_data/FullLengthVideos/labels/valid')
    print("Start training")
    model = CRClassifier(hidden=512,num_layers=9,get_last=False).cuda()
    model.load_state_dict(torch.load('model/answer_2_classifier_46.55.pth'))
    worker = SeqWorker()
    worker.batch_size=1
    worker.epoch=10
    optim = torch.optim.Adam(model.parameters(),lr=1e-5)
    record = worker.train(model,optim,
                          train_video,train_label,
                          val_video,val_label,
                          train_length,val_length,
                          len_threshold=int(sys.argv[2]),
                          is_seq2seq=True)
    name = str(sys.argv[3])+"_"+str(record['val_accuracy'][-1])
    with open(name+"_record.txt","w") as fp:
        fp.write( str(record) )
    torch.save(model.state_dict(),name+'_classifier.pth')