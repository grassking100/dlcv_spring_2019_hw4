from keras.preprocessing.sequence import pad_sequences
from reader import getVideoList,readShortVideo
import numpy as np
from torch.utils.data import Dataset
import os
import matplotlib.image as mpimg
import pandas as pd
import skimage.transform

class BasicDataset(Dataset):
    def __init__(self,*data):
        super().__init__()
        length = None
        for item in data:
            if length is None:
                length=len(item)
            else:
                if length != len(item):
                    raise Exception()
        self._length  = length          
        self._data = data
    def __len__(self):
        return self._length
    def __getitem__(self,index):
        items = []
        for item in self._data:
            items.append(item[index])
        return items

def load_data(video_path,label_path,return_label=True,rescale=True,rescale_factor=0.5,downsample_factor=1,use_sample=True):
    videos = []
    video_infos = getVideoList(label_path)
    
    lengths = []
    i = 0
    for category,name in zip(video_infos['Video_category'],video_infos['Video_name']):
        i+=1
        print("Parse: "+str(round(100*i/len(video_infos['Video_category']),2))+"%")
        item = readShortVideo(video_path,category,name,
                              rescale_factor=rescale_factor,
                              downsample_factor=downsample_factor)
        if rescale:
            item = item/255
        length = len(item)
        lengths.append(length)
        if use_sample:
            if length>=3:
                video_sample = item[0],item[int(length/2)],item[length-1]
                video = np.concatenate(video_sample,axis=2).transpose(2,0,1)
                videos.append([video])
            else:
                raise Exception()
        else:
            videos.append(item.transpose(0,3,1,2))
    if use_sample:
        videos = np.concatenate(videos)
    if return_label:
        labels = video_infos['Action_labels']
        labels = np.array(labels).astype('int')
        if use_sample:
            return videos,labels
        else:
            return videos,labels,lengths
    else:
        if use_sample:
            return videos
        else:
            return videos,lengths

def read_frames(frame_folder_path,label_path=None,rescale=True,rescale_factor=1):
    names = [path for path in os.listdir(frame_folder_path) if path.endswith('.jpg')]
    paths = [frame_folder_path+"/"+name for name in names]
    frames = []
    for path in sorted(paths):
        frame = mpimg.imread(path)
        frame = skimage.transform.rescale(frame, rescale_factor,
                                          mode='constant', preserve_range=True,
                                          multichannel=True, anti_aliasing=True)
        frames.append(frame)
    frames = np.array(frames)
    frames = frames.transpose((0,3,1,2))
    if rescale:
        frames = frames/255
    if label_path is not None:
        labels = np.array(pd.read_csv(label_path,header=None)[0]).astype(int)
        return frames,labels
    else:
        return frames    

def load_full_videos(video_root,label_root=None,rescale_factor=0.5):
    train_names = [path for path in os.listdir(video_root)]
    videos = []
    labels = []
    lengths = []
    label_path = None
    for name in train_names:
        video_path = video_root+"/"+name
        if label_root is not None:
            label_path = label_root+'/'+name+".txt"
            video,label = read_frames(video_path,label_path,rescale_factor=rescale_factor)
            labels.append(label)
        else:
            video = read_frames(video_path,rescale_factor=rescale_factor)
        videos.append(video)
        lengths.append(len(video))
    if label_root is not None:
        return videos,labels,lengths
    else:
        return videos,lengths