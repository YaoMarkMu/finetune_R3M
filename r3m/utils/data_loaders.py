import warnings

from pathlib import Path

import hydra
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import IterableDataset
import pandas as pd
import json
import time
import pickle
from torchvision.utils import save_image
import json
import random
import h5py


def get_dataset(num_labeled=40000, path = "./data/"):
    langs,image_data,actions=load_data(path,num_labeled)
    num_labeled=len(langs)
    return langs,image_data, actions, num_labeled

def load_data(path,num_labeled):
    f = h5py.File(path+ "data.hdf5", 'r')
    langs = pd.read_csv(path + "labels.csv")
    filtr = [True] * num_labeled
    langs = langs["Text Description"].str.strip().to_numpy().reshape(-1)
    langs = langs[:num_labeled]
    nuq = len(np.unique(langs[:]))
    print("Unique Instructions", len(np.unique(langs[:])))
    image_data = []
    ct = 1000
    for dti in range(0, num_labeled, ct):
        image_data.append((f['sim']['ims'][dti:(dti+ct)]*255).astype(np.uint8))
    filtr = np.array([int(("nothing" in l) or ("nan" in l) or ("wave" in l)) for l in langs]) == 0
    langs = langs[filtr]
    image_data = np.concatenate(image_data)[filtr]
    actions = f['sim']['actions'][:num_labeled][filtr]
    num_labeled = int(np.sum(filtr))
    NUMEP = image_data.shape[0]
    NUMSTEP = image_data.shape[1]
    H, W, C = image_data.shape[2:]
    print("load {} data includes {} steps H,W,C:{},{},{}".format(NUMEP,NUMSTEP,H, W, C))
    return langs,image_data,actions

## Data Loader for Ego4D
class R3MBuffer(IterableDataset):
    def __init__(self, ego4dpath, num_workers, source1, source2, alpha, datasources, doaug = "none"):
        self._num_workers = max(1, num_workers)
        self.alpha = alpha
        self.curr_same = 0
        self.data_sources = datasources
        self.doaug = doaug

        # Augmentations
        if doaug in ["rc", "rctraj"]:
            self.aug = torch.nn.Sequential(
                transforms.RandomResizedCrop(224, scale = (0.2, 1.0)),
            )
        else:
            self.aug = lambda a : a

    
        self.data_langs,self.data_image, self.data_actions, self.ego4dlen=get_dataset()


    def _sample(self):
        vidid = np.random.randint(0, self.ego4dlen)
        # m = self.manifest.iloc[vidid]
        vidlen = len(self.data_langs)
        txt = self.data_langs[vidid]
        label = str(txt)
        video = self.data_image[vidid]

        start_ind = np.random.randint(1, 2 + int(self.alpha * vidlen))
        end_ind = np.random.randint(int((1-self.alpha) * vidlen)-1, vidlen)
        s1_ind = np.random.randint(2, vidlen)
        s0_ind = np.random.randint(1, s1_ind)
        s2_ind = np.random.randint(s1_ind, vidlen+1)

        if self.doaug == "rctraj":
            ### Encode each image in the video at once the same way
            im0 = video[start_ind]
            img = video[end_ind]
            imts0 = video[s0_ind]
            imts1 = video[s1_ind]
            imts2 = video[s2_ind]
            allims = torch.stack([im0, img, imts0, imts1, imts2], 0)
            allims_aug = self.aug(allims / 255.0) * 255.0

            im0 = allims_aug[0]
            img = allims_aug[1]
            imts0 = allims_aug[2]
            imts1 = allims_aug[3]
            imts2 = allims_aug[4]
        else:
            ### Encode each image individually
            im0 = self.aug(video[start_ind] / 255.0) * 255.0
            img = self.aug(video[end_ind] / 255.0) * 255.0
            imts0 = self.aug(video[s0_ind] / 255.0) * 255.0
            imts1 = self.aug(video[s1_ind] / 255.0) * 255.0
            imts2 = self.aug(video[s2_ind] / 255.0) * 255.0

        im = torch.stack([im0, img, imts0, imts1, imts2])
        return (im, label)

    def __iter__(self):
        while True:
            yield self._sample()