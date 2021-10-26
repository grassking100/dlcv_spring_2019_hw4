import sys
import torch
torch.backends.cudnn.benchmark = True
from utils import load_data
from model import CRClassifier
from worker import seq_predict
import pandas as pd

if __name__=='__main__':
    input_path = str(sys.argv[1])
    answer_path = str(sys.argv[2])
    output_path = str(sys.argv[3])
    model_path = str(sys.argv[4])

    classifier = CRClassifier(hidden=512,num_layers=9).cuda()
    classifier.load_state_dict(torch.load(model_path))
    inputs,lengths = load_data(input_path,answer_path,use_sample=False,
                                       rescale_factor=0.5,downsample_factor=12,return_label=False)
    result = seq_predict(classifier,inputs,lengths,len_threshold=256,is_seq2seq=False)
    if output_path[-1] != '/':
        output_path += '/'
    with open(output_path+"p2_result.txt","w") as fp:
        for item in result:
            fp.write(str(item)+"\n")
