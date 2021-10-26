import torch
from torch.nn import functional as F

class Callback:
    def on_work_begin(self,worker,**kwargs):
        pass
    def on_work_end(self):
        pass
    def on_epoch_begin(self,counter,**kwargs):
        pass
    def on_epoch_end(self,metric,**kwargs):
        pass
    def on_batch_begin(self):
        pass
    def on_batch_end(self,outputs,answers,metric,**kwargs):
        pass

class Callbacks(Callback):
    def __init__(self,callbacks):
        callbacks = callbacks or []
        list_ = []
        for callback in callbacks:
            if hasattr(callback,'callbacks'):
                list_ += callback.callbacks
            else:
                list_ += [callback]
        self.callbacks = list_
    def on_work_begin(self,worker,**kwargs):
        for callback in self.callbacks:
            callback.on_work_begin(worker=worker,**kwargs)
    def on_work_end(self):
        for callback in self.callbacks:
            callback.on_work_end()
    def on_epoch_begin(self,counter,**kwargs):
        for callback in self.callbacks:
            callback.on_epoch_begin(counter=counter,**kwargs)
    def on_epoch_end(self,metric,**kwargs):
        for callback in self.callbacks:
            callback.on_epoch_end(metric,**kwargs)        
    def on_batch_begin(self):
        for callback in self.callbacks:
            callback.on_batch_begin()
    def on_batch_end(self,outputs,answers,metric,**kwargs):
        for callback in self.callbacks:
            callback.on_batch_end(outputs=outputs,answers=answers,metric=metric,**kwargs)
    def get_data(self):
        record = {}
        for callback in self.callbacks:
            if hasattr(callback,'data') and callback.data is not None:
                for type_,value in callback.data.items():
                    record[type_]=value
        return record

class DataCallback(Callback):
    def __init__(self):
        self._data = None
        self._prefix = ""
    @property
    def prefix(self):
        return self._prefix
    @prefix.setter
    def prefix(self,value):
        if value is not None and len(value)>0:
            value+="_"
        else:
            value=""
        self._prefix = value
    @property
    def data(self):
        pass

class Recorder(DataCallback):
    def __init__(self):
        super().__init__()
        self.path = None
        self._data = {}
        
    def on_epoch_end(self,metric,**kwargs):
        for type_,value in metric.items():
            if type_ not in self._data.keys():
                self._data[type_] = []
            self._data[type_].append(value)
    def on_work_end(self):
        if self.path is not None:
            df = pd.DataFrame.from_dict(self._data)
            df.to_csv(self.path)
    @property
    def data(self):
        return self._data

class Accuracy(DataCallback):
    def on_batch_end(self,outputs,answers,**kwargs):
        with torch.no_grad():
            outputs = F.softmax(outputs,dim=1)
            data = (outputs.max(1)[1]==answers).float().mean()
            self._data += data.cpu().numpy()*100
    @property
    def data(self):
        return {self._prefix+'accuracy':self._data}
    def on_epoch_begin(self,**kwargs):
        self._reset()
    def _reset(self):
        self._data = 0

class Accumulator(DataCallback):
    def __init__(self):
        super().__init__()
        self.name = ""
        self._batch_count = 0
        
    def _reset(self):
        self._data = 0
        self._batch_count = 0

    def on_epoch_begin(self,**kwargs):
        self._reset()

    def on_batch_end(self,metric,**kwargs):
        if metric is not None:
            with torch.no_grad():
                self._data+=metric
                self._batch_count+=1
    @property
    def data(self):
        if self._batch_count > 0:
            return {self._prefix+self.name:self._data}
        else:
            return None