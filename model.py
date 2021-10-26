import math
import torchvision.models as models
import torch.nn as nn
from torch.nn import Sequential
from torch.nn import functional as F
import torch
celu = nn.CELU()
leaky_relu = nn.LeakyReLU()

def init_norm(norm):
    norm.weight.data.normal_(1.0, 0.02)
    norm.bias.data.fill_(0)
    
def init_cnn(cnn):
    cnn.weight.data.normal_(0.0, 0.02)
    cnn.bias.data.fill_(0)
    
def init_linear(linear):
    linear.weight.data.normal_(0.0, 0.02)
    linear.bias.data.fill_(0)

class LRNBlock(nn.Module):
    def __init__(self, input_size=100, output_size=32,use_leaky_relu=False):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size,bias=True)
        self.norm = nn.LayerNorm(output_size)
        init_norm(self.norm)
        self.use_leaky_relu = use_leaky_relu
    def forward(self, x):
        x = self.linear(x)
        if self.use_leaky_relu:
            x = leaky_relu(x)
        else:
            x = F.relu(x)
        x = self.norm(x)
        return x

class CNNClassifier(nn.Module):
    def __init__(self,hidden=512,output_size=11,use_leaky_relu=True):
        super().__init__()
        if use_leaky_relu:
            self.activation = leaky_relu
        else:
            self.activation = F.relu
        self.features = models.vgg16(pretrained=True).features
        self.norm0 = nn.LayerNorm(7680)
        self.cnn = nn.Conv2d(512,hidden,3)
        self.norm1 = nn.LayerNorm(hidden*3)
        self.norm2 = nn.LayerNorm(output_size)
        self.classifier = Sequential(
            LRNBlock(hidden*9,hidden,use_leaky_relu=use_leaky_relu),
            nn.Linear(hidden,output_size,bias=True)
        )

    def forward(self, x,use_softmax=False):
        N,C,W,H = x.shape
        real_N = N
        x = x.reshape(-1,3,W,H)
        x = self.features(x)
        N,C,W,H = x.shape
        x = x.reshape(N,-1)
        x = self.norm0(x)
        x = x.reshape(N,C,W,H)
        x = self.cnn(x)
        x = self.activation(x)
        N,C,W,H = x.shape
        x = x.reshape(N,-1)
        x = self.norm1(x)
        x = x.reshape(real_N,-1)
        x = self.classifier(x)
        x = self.norm2(x)
        return x

class CNNFeature(nn.Module):
    def __init__(self,hidden=512,use_leaky_relu=True):
        super().__init__()
        if use_leaky_relu:
            self.activation = leaky_relu
        else:
            self.activation = F.relu
        self.features = models.vgg16(pretrained=True).features
        self.norm0 = nn.LayerNorm(7680)
        self.cnn = nn.Conv2d(512,hidden,3)
        self.norm1 = nn.LayerNorm(hidden*3)

    def forward(self, x,use_softmax=False):
        N,C,W,H = x.shape
        x = x.reshape(-1,3,W,H)
        x = self.features(x)
        N,C,W,H = x.shape
        x = x.reshape(N,-1)
        x = self.norm0(x)
        x = x.reshape(N,C,W,H)
        x = self.cnn(x)
        x = self.activation(x)
        N,C,W,H = x.shape
        x = x.reshape(N,-1)
        x = self.norm1(x)
        return x    
    
def init_GRU(gru):
    for name, param in gru.named_parameters():
        if 'weight_ih' in name:
            nn.init.xavier_uniform_(param.data)
        elif 'weight_hh' in name:
            nn.init.orthogonal_(param.data)
        elif 'bias' in name:
            param.data.fill_(0)

def rnn_forward_wrapper(rnn,x,lengths,previous_state=None,get_last=True):
    x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
    x,current_state = rnn(x,previous_state)
    x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
    if get_last:
        temp = []
        for index in range(len(x)):
            length = int(lengths[index])
            temp.append(x[index,length-1,:].unsqueeze(0))
        x = torch.cat(temp)
    use_cuda = next(rnn.parameters()).is_cuda
    if use_cuda:
        x = x.cuda()
        current_state = current_state.cuda()
    return x,current_state

class CRClassifier(nn.Module):
    def __init__(self,output_size=11,hidden=16,num_layers=2,activation=None,get_last=True):
        super().__init__()
        if activation is None:
            activation = leaky_relu
        self.activation = activation
        self.get_last = get_last
        self.hidden = hidden
        self.features = models.vgg16(pretrained=True).features
        self.norm0 = nn.LayerNorm(7680)
        self.cnn = nn.Conv2d(512,hidden,3)
        p_hidden_size = hidden*3
        self.norm1 = nn.LayerNorm(p_hidden_size)
        self.rnn = nn.GRU(p_hidden_size,hidden,batch_first=True,
                          bidirectional=False,num_layers=num_layers)
        self.pre_classifier = nn.Linear(hidden,hidden,bias=True)
        self.norm2 = nn.LayerNorm(hidden)
        self.classifier = nn.Linear(hidden,output_size,bias=True)
        self.norm3 = nn.LayerNorm(output_size)
        init_norm(self.norm0)
        init_cnn(self.cnn)
        init_norm(self.norm1)
        init_linear(self.pre_classifier)
        init_linear(self.classifier)
        init_norm(self.norm2)
        init_norm(self.norm3)

    def forward(self, x,lengths,previous_state=None):
        N,L,C,W,H = x.shape
        x = x.reshape(-1,C,W,H)
        x = self.features(x)
        shape = x.shape
        x = x.reshape(N*L,-1)
        x = self.norm0(x)
        x = x.reshape(*shape)
        x = self.cnn(x)
        x = self.activation(x)
        x = x.reshape(N*L,-1)
        x = self.norm1(x)
        x = x.reshape(N,L,-1)
        x,current_state = rnn_forward_wrapper(self.rnn,x,lengths,previous_state=previous_state,get_last=self.get_last)
        x = self.pre_classifier(x)
        x = self.activation(x)
        x = self.norm2(x)
        x = self.classifier(x)
        x = self.norm3(x)
        return x,current_state

class RNNFeature(nn.Module):
    def __init__(self,output_size=11,hidden=16,num_layers=2,activation=None,get_last=True):
        super().__init__()
        if activation is None:
            activation = leaky_relu
        self.activation = activation
        self.get_last = get_last
        self.hidden = hidden
        self.features = models.vgg16(pretrained=True).features
        self.norm0 = nn.LayerNorm(7680)
        self.cnn = nn.Conv2d(512,hidden,3)
        p_hidden_size = hidden*3
        self.norm1 = nn.LayerNorm(p_hidden_size)
        self.rnn = nn.GRU(p_hidden_size,hidden,batch_first=True,
                          bidirectional=False,num_layers=num_layers)
        init_norm(self.norm0)
        init_cnn(self.cnn)
        init_norm(self.norm1)

    def forward(self, x,lengths,previous_state=None):
        N,L,C,W,H = x.shape
        x = x.reshape(-1,C,W,H)
        x = self.features(x)
        shape = x.shape
        x = x.reshape(N*L,-1)
        x = self.norm0(x)
        x = x.reshape(*shape)
        x = self.cnn(x)
        x = self.activation(x)
        x = x.reshape(N*L,-1)
        x = self.norm1(x)
        x = x.reshape(N,L,-1)
        x,current_state = rnn_forward_wrapper(self.rnn,x,lengths,previous_state=previous_state,get_last=self.get_last)
        return x,current_state    
    
class Decoder(nn.Module):
    def __init__(self,input_size=16,num_layers=2):
        super().__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.attention = nn.Linear(input_size+input_size,1)
        self.rnn = nn.GRU(input_size,input_size,batch_first=True,num_layers=num_layers)
        self.attention.weight.data.normal_(0, 0.01)
        self.attention.bias.data.fill_(-1)
    def forward(self, x,lengths,previous_state=None):
        N,L,C = x.shape
        if len(x)!=1:
            raise Exception("Batch size should be one")
        output = torch.zeros(1,self.input_size).cuda()
        outputs = []
        length = lengths[0].item()
        #print("~~~~~~~")
        #print(self.attention.weight)
        for decoder_index in range(length):
            aligns = []
            temp = output.reshape(1,-1).repeat(length,1)
            concated = torch.cat([temp,x[0]],1)
            #print(concated)
            attention_value = self.attention(concated).unsqueeze(0)
            #print(attention_value)
            attention_value = F.softmax(attention_value,dim=1)
            #print(attention_value)
            attention_value = attention_value*x
            attention_value = attention_value.sum(1)
            attention_value = attention_value.unsqueeze(1)
            output,previous_state = self.rnn(attention_value,previous_state)
            outputs.append(output[0])
        outputs = torch.cat(outputs,0).unsqueeze(0)
        return outputs,previous_state

class Attention(nn.Module):
    def __init__(self,output_size=11,hidden=16,num_layers=2,activation=None,get_last=True):
        super().__init__()
        if activation is None:
            activation = leaky_relu
        self.activation = activation
        self.features = models.vgg16(pretrained=True).features
        self.norm0 = nn.LayerNorm(7680)
        self.cnn = nn.Conv2d(512,hidden,3)
        p_hidden_size = hidden*3
        self.norm1 = nn.LayerNorm(p_hidden_size)
        self.decoder = Decoder(input_size=p_hidden_size,num_layers=num_layers)
        self.get_last = get_last
        self.pre_classifier = nn.Linear(p_hidden_size,hidden)
        self.classifier = nn.Linear(hidden,output_size)        
        self.norm2 = nn.LayerNorm(hidden)
        self.norm3 = nn.LayerNorm(output_size)
        init_norm(self.norm0)
        init_cnn(self.cnn)
        init_norm(self.norm1)
        init_linear(self.pre_classifier)
        init_linear(self.classifier)
        init_norm(self.norm2)
        init_norm(self.norm3)
        
    def forward(self, x,lengths,previous_state=None):
        #print(x.shape)
        N,L,C,W,H = x.shape
        if N!=1:
            raise Exception("Batch size should be one")
        x = x.reshape(-1,C,W,H)
        x = self.features(x)
        shape = x.shape
        x = x.reshape(L,-1)
        x = self.norm0(x)
        x = x.reshape(*shape)
        x = self.cnn(x)
        x = leaky_relu(x)
        x = x.reshape(L,-1)
        x = self.norm1(x)
        x = x.reshape(1,L,-1)
        x,hidden_state = self.decoder(x,lengths,previous_state)
        x = x.reshape(L,-1)
        if self.get_last:
            x = x[-1].unsqueeze(0)
        x = self.pre_classifier(x)
        x = self.activation(x)
        x = self.norm2(x)
        x = self.classifier(x)
        x = self.norm3(x)
        if not self.get_last:
            x = x.reshape(1,L,-1)
        return x,hidden_state
