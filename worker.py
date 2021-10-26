from torch.utils.data import TensorDataset,DataLoader
import torch
import numpy as np
import time
import torch.nn as nn
from torch.nn import functional as F
import math
try:
    from .utils import BasicDataset
    from .callback import Callbacks,Accumulator,Recorder,Accuracy
except:
    from utils import BasicDataset
    from callback import Callbacks,Accumulator,Recorder,Accuracy

def get_data(indice,inputs,answers,lengths):
    inputs = [inputs[i] for i in indice]
    answers = [answers[i] for i in indice]
    lengths = [lengths[i] for i in indice]
    inputs = torch.Tensor(inputs).type(torch.get_default_dtype())
    answers = torch.Tensor(answers).long()
    lengths = torch.Tensor(lengths).long()
    return inputs,answers,lengths
    
def seq_calculate(model,inputs,lengths,previous_state=None):
    torch.cuda.empty_cache()
    inputs = inputs.cuda()
    lengths = lengths.cuda()
    if previous_state is not None:
        result = model(inputs,lengths,previous_state=previous_state)
    else:
        result = model(inputs,lengths)
    return result
    
def callbacks_init(callbacks,prefix=''):
    callbacks = callbacks or []
    accum = Accumulator()
    accum.name='loss'
    callbacks.append(accum)
    for callback in callbacks:
        if hasattr(callback,'prefix'):
            callback.prefix=prefix
    callbacks = Callbacks(callbacks)
    return callbacks

def basic_predict(model,inputs,get_max=True):
    model.train(False)
    data = TensorDataset(torch.Tensor(inputs))
    loader = DataLoader(dataset=data,batch_size=1,shuffle=False)
    outputs = []
    for input_ in loader:
        with torch.no_grad():
            output = model(input_[0].cuda())
        output = output.cpu().numpy()[0]    
        if get_max:    
            output = np.argmax(output)
        outputs.append(output.tolist())
    return outputs

def seq_predict(model,inputs,lengths,len_threshold=50,is_seq2seq=False,get_max=True):
    data = BasicDataset(list(range(len(inputs))))
    loader = DataLoader(dataset=data,batch_size=1,shuffle=False)
    model.train(False)
    outputs = []
    for index in loader:
        index = index[0]
        input_ = torch.Tensor([inputs[index]]).type(torch.get_default_dtype())
        length = torch.Tensor([lengths[index]]).long()
        n = math.ceil(lengths[index]/len_threshold)
        previous_state = None
        output_list = []
        for index in range(n):
            t_i,t_l = get_truncated(index,len_threshold,input_[0],is_seq2seq=is_seq2seq)
            N,L,C,W,H = t_i.shape
            with torch.no_grad():
                result = seq_calculate(model,t_i,t_l,previous_state=previous_state)
                if len(result)==2:
                    output,previous_state = result
                else:
                    output = result
            if is_seq2seq:
                output = output.reshape(L,-1)
                output_list.append(output)
        if is_seq2seq:
            output = torch.cat(output_list,0)
        else:
            output = output[0]
        output = output.cpu().numpy()
        if get_max:
            if is_seq2seq:
                output = np.argmax(output,axis=1)
            else:
                output = np.argmax(output,axis=0)
        outputs.append(output.tolist())
    return outputs
    
def frozen(model):
    for children in model.parameters():
        children.requires_grad = False

def unfrozen(model):
    for children in model.parameters():
        children.requires_grad = True

def basic_evaluate(model,loss,inputs,answers,callbacks):
    torch.cuda.empty_cache()
    callbacks.on_batch_begin()
    inputs = inputs.cuda()
    answers = answers.cuda()
    model.train(False)
    with torch.no_grad():
        outputs = model(inputs)
        loss_ = loss(outputs, answers).item()
    callbacks.on_batch_end(outputs=outputs,answers=answers,metric=loss_)
    
def basic_fit(model,loss,optimizer,inputs,answers,callback):
    torch.cuda.empty_cache()
    callback.on_batch_begin()
    inputs = inputs.cuda()
    answers = answers.cuda()
    unfrozen(model)
    model.train(True)
    outputs = model(inputs)
    loss_ = loss(outputs,answers)
    optimizer.zero_grad()
    loss_.backward()
    optimizer.step()
    callback.on_batch_end(outputs=outputs,answers=answers,metric=loss_.item())

def get_truncated(index,len_threshold,input_,answer=None,is_seq2seq=False):
    start = index*len_threshold
    end = (index+1)*len_threshold
    input_ = input_[start:end].unsqueeze(0)
    length = torch.Tensor([input_.shape[1]]).long()
    if answer is not None:
        if is_seq2seq:
            answer = answer[start:end]
        else:
            answer = answer.unsqueeze(0)
        return input_, answer, length
    else:
        return input_, length

def seq_evaluate(model,loss,inputs,answers,lengths,callbacks,len_threshold=50,is_seq2seq=False):
    if not (len(inputs)==len(answers)==len(lengths)==1):
        raise Exception("Batch size should be one")
    callbacks.on_batch_begin()
    model.train(False)
    inputs = inputs[0]
    answers = answers[0]
    n = math.ceil(lengths[0].item()/len_threshold)
    previous_state = None
    outputs_list = []
    for index in range(n):
        t_i,t_a,t_l = get_truncated(index,len_threshold,inputs,answers,is_seq2seq)
        with torch.no_grad():
            result = seq_calculate(model,t_i,t_l,previous_state=previous_state)
            if len(result)==2:
                outputs,previous_state = result
                previous_state = previous_state.detach()
            else:
                outputs = result
            if is_seq2seq:
                outputs = outputs[0]
            loss_ = loss(outputs,t_a.cuda())
        if is_seq2seq:
            outputs = outputs.detach()
            outputs_list.append(outputs)
    if is_seq2seq:
        outputs = torch.cat(outputs_list,0)
    callbacks.on_batch_end(outputs=outputs.cuda(),answers=answers.cuda(),metric=loss_.item())
    
def seq_fit(model,loss,optimizer,inputs,answers,lengths,callback,len_threshold=50,is_seq2seq=False):
    if not (len(inputs)==len(answers)==len(lengths)==1):
        raise Exception("Batch size should be one")
    callback.on_batch_begin()
    unfrozen(model)
    model.train(True)
    inputs = inputs[0]
    answers = answers[0]
    n = math.ceil(lengths[0].item()/len_threshold)
    previous_state = None
    outputs_list = []
    for index in range(n):
        t_i,t_a,t_l = get_truncated(index,len_threshold,inputs,answers,is_seq2seq)
        result = seq_calculate(model,t_i,t_l,previous_state=previous_state)
        if len(result)==2:
            outputs,previous_state = result
            previous_state = previous_state.detach()
        else:
            outputs = result
        if is_seq2seq:
            outputs = outputs[0]
        loss_ = loss(outputs,t_a.cuda())
        optimizer.zero_grad()
        loss_.backward()
        optimizer.step()
        if is_seq2seq:
            outputs = outputs.detach()
            outputs_list.append(outputs)
    if is_seq2seq:
        outputs = torch.cat(outputs_list,0)
    callback.on_batch_end(outputs=outputs.cuda(),answers=answers.cuda(),metric=loss_.item())

def _prepare_callbacks(lhs_callbacks,rhs_callbacks,l_prefix,r_prefix):
    lhs_callbacks = callbacks_init(lhs_callbacks,l_prefix)
    rhs_callbacks = callbacks_init(rhs_callbacks,r_prefix)
    recoder = Recorder()
    callbacks = Callbacks([lhs_callbacks,rhs_callbacks])
    all_callbacks = Callbacks([lhs_callbacks,rhs_callbacks,recoder])
    return recoder,callbacks,all_callbacks,lhs_callbacks,rhs_callbacks

def _process_record(callbacks,size,names):
    record = callbacks.get_data()
    for name in names:
        if name in record.keys():
            record[name] = np.round(record[name]/size,2)
    return record

class Worker:
    def __init__(self):
        self.batch_size = 32
        self.epoch = 10

class BasicWorker(Worker):
    def train(self,model,optimizer,train_inputs,train_answers,val_inputs,val_answers,scheduler=None):
        loss = nn.CrossEntropyLoss(reduction='mean')
        callbacks_list = _prepare_callbacks([Accuracy()],[Accuracy()],'train','val')
        recoder,callbacks,all_callbacks,train_callbacks,val_callbacks = callbacks_list
        all_callbacks.on_work_begin(worker=self)
        train_inputs = torch.Tensor(train_inputs)
        train_answers = torch.Tensor(train_answers).long()
        train_data = TensorDataset(train_inputs,train_answers)
        val_inputs = torch.Tensor(val_inputs)
        val_answers = torch.Tensor(val_answers).long()
        val_data = TensorDataset(val_inputs,val_answers)
        train_loader = DataLoader(dataset=train_data,batch_size=self.batch_size,shuffle=True)
        val_loader = DataLoader(dataset=val_data,batch_size=self.batch_size,shuffle=False)
        for epoch_num in range(self.epoch):
            pre_time = time.time()
            all_callbacks.on_epoch_begin(counter=epoch_num)
            for (t_input,t_answer) in train_loader:
                basic_fit(model,loss,optimizer,t_input,t_answer,train_callbacks)
            for (v_input,v_answer) in val_loader:
                basic_evaluate(model,loss,v_input,v_answer,val_callbacks)
            train_record = _process_record(train_callbacks,len(train_loader),['train_accuracy','train_loss'])
            val_record = _process_record(val_callbacks,len(val_loader),['val_accuracy','val_loss'])
            record = {}
            record.update(train_record)
            record.update(val_record)
            elap_time = str(round((time.time() - pre_time)/60,2))+" minutes"
            print(elap_time+"/ epoch:"+str(epoch_num+1)+"/ record:"+str(record))
            all_callbacks.on_epoch_end(metric=record)
            if scheduler is not None:
                scheduler.step()
        all_callbacks.on_work_end()
        return recoder.data
    
class SeqWorker(Worker):
    def train(self,model,optimizer,train_inputs,train_answers,val_inputs,val_answers,train_lengths,val_lengths,
              len_threshold=50,scheduler=None,is_seq2seq=False,stop_value=None):
        if self.batch_size!=1:
            raise Exception("Data batch size shold be one")
        loss = nn.CrossEntropyLoss(reduction='mean')
        callbacks_list = _prepare_callbacks([Accuracy()],[Accuracy()],'train','val')
        recoder,callbacks,all_callbacks,train_callbacks,val_callbacks = callbacks_list
        train_data = BasicDataset(list(range(len(train_inputs))))
        val_data = BasicDataset(list(range(len(val_inputs))))
        train_loader = DataLoader(dataset=train_data,batch_size=self.batch_size,shuffle=True)
        val_loader = DataLoader(dataset=val_data,batch_size=self.batch_size,shuffle=False)
        for epoch_num in range(self.epoch):
            pre_time = time.time()
            all_callbacks.on_epoch_begin(counter=epoch_num)
            for index in train_loader:
                index = index[0].numpy()
                t_input,t_answer,t_length = get_data(index,train_inputs,train_answers,train_lengths)
                seq_fit(model,loss,optimizer,t_input,t_answer,t_length,train_callbacks,
                           len_threshold,is_seq2seq)
            for index in val_loader:
                index = index[0].numpy()
                v_input,v_answer,v_length = get_data(index,val_inputs,val_answers,val_lengths)
                seq_evaluate(model,loss,v_input,v_answer,v_length,val_callbacks,
                                len_threshold,is_seq2seq)
            train_record = _process_record(train_callbacks,len(train_loader),['train_accuracy','train_loss'])
            val_record = _process_record(val_callbacks,len(val_loader),['val_accuracy','val_loss'])
            record = {}
            record.update(train_record)
            record.update(val_record)
            elap_time = str(round((time.time() - pre_time)/60,2))+" minutes"
            print(elap_time+"/ epoch:"+str(epoch_num+1)+"/ record:"+str(record))
            all_callbacks.on_epoch_end(metric=record)
            if scheduler is not None:
                scheduler.step()
            if stop_value is not None:    
                if record['val_accuracy']>=stop_value:
                    break
        all_callbacks.on_work_end()
        return recoder.data