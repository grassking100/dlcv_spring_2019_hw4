import sys
import os
import torch
torch.backends.cudnn.benchmark = True
import numpy as np
from model import CRClassifier
from worker import seq_predict
from utils import load_full_videos

def post_process(ann):
    new_ann = []
    for index in range(len(ann)):
        if index>=1 and index<len(ann)-1:
            if ann[index-1]==ann[index+1]:
                new_ann.append(ann[index-1])
            else:
                new_ann.append(ann[index])
        else:
            new_ann.append(ann[index])
    return new_ann

if __name__=='__main__':
    input_path = str(sys.argv[1])
    output_path = str(sys.argv[2])
    model_path = str(sys.argv[3])
    model = CRClassifier(hidden=512,num_layers=9,get_last=False).cuda()
    model.load_state_dict(torch.load(model_path))
    videos,lengths = load_full_videos(input_path)
    names = [path for path in os.listdir(input_path)]

    result = seq_predict(model,videos,lengths,len_threshold=256,is_seq2seq=True)
    new_result = []
    for item in result:
        new_result.append(post_process(item))
    if output_path[-1] != '/':
        output_path += '/'
    for name,predict_labels in zip(names,new_result):
        with open(output_path+name+".txt","w") as fp:
            for item in predict_labels:
                fp.write(str(item)+"\n")
