import os
from PIL import Image
import torch
import kaolin
import torchvision
import numpy as np
from torch.utils.data import Dataset

class Shoes(Dataset):
    def __init__(self, dir) -> None:
        super().__init__()
        self.dir = dir
        self.max = 36
        self.a = 1.0
        self.b = 1.2
        self.height = 0.20
        self.fov = 1.293764705882353
        self.start = 3.142/2
        self.end = self.start + (2*3.142)
        self.folders = [ os.path.join(self.dir, dI) for dI in os.listdir( self.dir ) if os.path.isdir(os.path.join( self.dir, dI))]
        self.imgs = []

        self.load = lambda i: np.load(i)

        for folder in self.folders:

            paths = {
                "masks": [],
                "images": []
            }

            for i in range(self.max-1):
                paths["images"].append( folder + "/" + str(i) + ".npy" )
                paths["masks"].append( folder + "/" + str(i) + "_mask.npy" )

            # extra img and masks cause i forgot to process the 36th image...
            paths["images"].append( folder + "/" + str(i) + ".npy" )
            paths["masks"].append( folder + "/" + str(i) + "_mask.npy" )

            self.imgs.append(paths)    

        z = self.a * torch.sin( torch.linspace(self.start, self.end, self.max) )
        x = self.b * torch.cos( torch.linspace(self.start, self.end, self.max) )

        cam_locs = torch.vstack( (x, torch.full(x.shape, self.height), -z) ).T

        self.R, self.T = kaolin.render.camera.generate_rotate_translate_matrices(
            cam_locs,
            torch.tensor([[0.0, self.height, -self.height / 4]]).repeat(self.max, 1),
            torch.tensor([[0.0, 1.0, 0.0]]).repeat(self.max, 1)
        )

        self.P = kaolin.render.camera.generate_perspective_projection(self.fov, 96/128)

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, index):

        imgs = torch.from_numpy( np.stack([self.load(img) for img in self.imgs[index]["images"] ]) )
        masks = torch.from_numpy( np.stack([self.load(img) for img in self.imgs[index]["masks"] ]) )        

        return {
            "masks": masks,
            "images": imgs,
            "R": self.R,
            "T": self.T,
            "P": self.P,
        }