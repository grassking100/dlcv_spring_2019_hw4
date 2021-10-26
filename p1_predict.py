import sys
import torch
torch.backends.cudnn.benchmark = True
from utils import load_data
from model import CNNClassifier
from worker import basic_predict
import pandas as pd

if __name__=='__main__':
    input_path = str(sys.argv[1])
    answer_path = str(sys.argv[2])
    output_path = str(sys.argv[3])
    model_path = str(sys.argv[4])
    classifier = CNNClassifier().cuda()
    classifier.load_state_dict(torch.load(model_path))
    inputs = load_data(input_path,answer_path,use_sample=True,rescale_factor=0.5,return_label=False)
    result = basic_predict(classifier,inputs)
    if output_path[-1] != '/':
        output_path += '/'
    with open(output_path+"p1_valid.txt","w") as fp:
        for item in result:
            fp.write(str(item)+"\n")
