import os

import cv2
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from torch.utils.data import DataLoader
import random
import copy
import torch
from tqdm import tqdm
from cvcities_base.transforms import get_transforms_train, get_transforms_val
import time
import matplotlib.pyplot as plt
fov = 70

def unnormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor
def ground_to_sate(x_ps, y_ps, W_s, H_s, W_ps, H_ps):
    """
    Convert pixel coordinates (x_ps, y_ps) to the new coordinate system (x_s, y_s)

    Parameters:
    x_ps: Original pixel coordinate x-value
    y_ps: Original pixel coordinate y-value
        W_s: Width of the target image
        H_s: Height of the target image
        W_ps: Width of the original pixel coordinates
        H_ps: Height of the original pixel coordinates

    Returns:
    x_s, y_s: Coordinates in the new coordinate system
    """
    x_s = W_s / 2 + (W_s / 2) * (y_ps / H_ps) * np.sin(2 * np.pi * x_ps / W_ps)

    y_s = H_s / 2 - (H_s / 2) * (y_ps / H_ps) * np.cos(2 * np.pi * x_ps / W_ps)
    x = W_s - x_s
    y = H_s - y_s
    return x, y
def random_crop(image, target_width, target_height):
    """
    Randomly crop an image to dimensions (target_width, target_height)

    Parameters:
        image: PIL.Image or np.ndarray (H, W, C)
        target_width: int, cropping width
        target_height: int, cropping height

    Returns:
        The cropped image, same type as input
    """
    is_np = isinstance(image, np.ndarray)
    if is_np:
        height, width = image.shape[:2]
    else:
        raise ValueError("Input must be NumPy array (H, W, C)")

    if width < target_width:
        raise ValueError(f"Image dimensions {width} are smaller than the target crop size {target_width}")
    if height < target_height:
        target_height = height

    x = random.randint(0, width - target_width)
    y = random.randint(0, height - target_height)


    cropped_image = image[y:y + target_height, x:x + target_width]

    return cropped_image,[x,y,x + target_width,y + target_height]
def coord2corner(pts):
    right_top = [pts[2],pts[1]]
    left_bottom = [pts[0],pts[3]]
    center = [(pts[0]+pts[2])/2,(pts[1]+pts[3])/2]
    return np.array(pts+right_top+left_bottom+center).reshape(-1,2)
def show_crop(pts,width,height):
    plt.figure()
    plt.xlim(0, width)
    plt.ylim(height, 0)  
    mask = np.zeros((height, width), dtype=np.float32)
    cv2.fillPoly(mask, [pts[:4]], color=1.0)
    plt.imshow(mask)
    plt.scatter(pts[4:5,0],pts[4:5,1],c='r',s=10)
    plt.show()



def generate_triangle_mask(pts, size=256, epsilon=1e-4):
    
    mask = np.zeros((size, size), dtype=np.float32)

    cv2.fillPoly(mask, [pts], color=1.0)

    mask[mask < 1.0] = epsilon
    return mask


class CVUSADatasetTrain(Dataset):

    def __init__(self,
                 data_folder,
                 transforms_query=None,
                 transforms_reference=None,
                 prob_flip=0.0,
                 prob_rotate=0.0,
                 shuffle_batch_size=128,
                 ):

        super().__init__()

        self.data_folder = data_folder
        self.prob_flip = prob_flip
        self.prob_rotate = prob_rotate
        self.shuffle_batch_size = shuffle_batch_size

        self.transforms_query = transforms_query  # ground
        self.transforms_reference = transforms_reference  # satellite

        self.df = pd.read_csv(f'{data_folder}/splits/train-19zl.csv', header=None)
        self.df = self.df.rename(columns={0: "sat", 1: "ground", 2: "ground_anno"})
        self.df["idx"] = self.df.sat.map(lambda x: int(x.split("/")[-1].split(".")[0]))

        self.idx2sat = dict(zip(self.df.idx, self.df.sat))
        self.idx2ground = dict(zip(self.df.idx, self.df.ground))
        self.pairs = list(zip(self.df.idx, self.df.sat, self.df.ground))

        self.idx2pair = dict()
        train_ids_list = list()

        # for shuffle pool
        for pair in self.pairs:
            idx = pair[0]
            self.idx2pair[idx] = pair
            train_ids_list.append(idx)

        self.train_ids = train_ids_list
        self.samples = copy.deepcopy(self.train_ids)

    def __getitem__(self, index):

        idx, sat, ground = self.idx2pair[self.samples[index]]
        ground_name = f'{self.data_folder}/{ground}'
        # load query -> ground image
        query_img = cv2.imread(ground_name)
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        h_g,w_g = query_img.shape[:2]
        pad_height = (w_g//2 - h_g)//2
        # efficent_img = query_img[h//4:h//4*3]
        width = int(fov / 360 * w_g)
        query_img, coord = random_crop(query_img, width, h_g)
        coord[1]+=pad_height
        coord[3] += pad_height
        # load reference -> satellite image
        reference_img = cv2.imread(f'{self.data_folder}/{sat}')
        reference_img = cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB)
        h = self.transforms_reference.transforms[1].height
        w = self.transforms_reference.transforms[1].width
        corner = coord2corner(coord)
        corner[[1, 2]] = corner[[2, 1]]

        x,y = ground_to_sate(corner[:,0:1],corner[:,1:2],w,h,w_g,w_g//2)
        sate_coord = np.concatenate([x,y],axis=1)
        sate_corner = sate_coord[0:4].astype(np.int32)
        sate_pos = sate_coord[4:5]
        sate_pos[0,0] -=  w / 2
        sate_pos[0,1] -=  h / 2
        mask = torch.tensor(generate_triangle_mask(sate_corner, size=h))
        # Flip simultaneously query and reference
        if np.random.random() < self.prob_flip:
            query_img = cv2.flip(query_img, 1)
            reference_img = cv2.flip(reference_img, 1)
            mask = torch.flip(mask, [1])
            sate_pos[0,0] = - sate_pos[0,0]
        # image transforms
        if self.transforms_query is not None:
            query_img = self.transforms_query(image=query_img)['image']

        if self.transforms_reference is not None:
            reference_img = self.transforms_reference(image=reference_img)['image']

        # Rotate simultaneously query and reference
        if np.random.random() < self.prob_rotate:
            r = np.random.choice([1, 2, 3])
            reference_img = torch.rot90(reference_img, k=r, dims=(1, 2))
            mask = torch.rot90(mask, k=r, dims=(0, 1))
            rot_mat = np.array(
                [np.cos(r * np.pi / 2), -np.sin(r * np.pi / 2), np.sin(r * np.pi / 2), np.cos(r * np.pi / 2)]).reshape(
                2, 2)
            sate_pos = sate_pos @ rot_mat

        sate_pos = sate_pos.tolist()[0]
        sate_pos = [sate_pos[0] + w / 2.0, sate_pos[1] + h / 2.0]
        label = torch.tensor(idx, dtype=torch.long)
        return query_img, reference_img, label, sate_pos, mask,os.path.basename(ground_name)

    def __len__(self):
        return len(self.samples)

    def shuffle(self, sim_dict=None, neighbour_select=64, neighbour_range=128):

        '''
        custom shuffle function for unique class_id sampling in batch 
        '''

        print("\nShuffle Dataset:")

        idx_pool = copy.deepcopy(self.train_ids)
        neighbour_split = neighbour_select // 2

        if sim_dict is not None:
            similarity_pool = copy.deepcopy(sim_dict)

        # Shuffle pairs order
        random.shuffle(idx_pool)  

        # Lookup if already used in epoch
        idx_epoch = set()
        idx_batch = set()

        # buckets
        batches = []
        current_batch = []

        # counter
        break_counter = 0

        # progressbar
        pbar = tqdm()

        while True:

            pbar.update()

            if len(idx_pool) > 0:
                idx = idx_pool.pop(0)

                if idx not in idx_batch and idx not in idx_epoch and len(current_batch) < self.shuffle_batch_size:

                    idx_batch.add(idx)
                    current_batch.append(idx)
                    idx_epoch.add(idx)
                    break_counter = 0

                    if sim_dict is not None and len(current_batch) < self.shuffle_batch_size:

                        near_similarity = similarity_pool[idx][:neighbour_range]

                        near_neighbours = copy.deepcopy(near_similarity[:neighbour_split])

                        far_neighbours = copy.deepcopy(near_similarity[neighbour_split:])

                        random.shuffle(far_neighbours)

                        far_neighbours = far_neighbours[:neighbour_split]

                        near_similarity_select = near_neighbours + far_neighbours

                        for idx_near in near_similarity_select:

                            # check for space in batch
                            if len(current_batch) >= self.shuffle_batch_size:
                                break

                            # check if idx not already in batch or epoch
                            if idx_near not in idx_batch and idx_near not in idx_epoch and idx_near:
                                idx_batch.add(idx_near)
                                current_batch.append(idx_near)
                                idx_epoch.add(idx_near)
                                similarity_pool[idx].remove(idx_near)
                                break_counter = 0

                else:
                    # if idx fits not in batch and is not already used in epoch -> back to pool
                    if idx not in idx_batch and idx not in idx_epoch:
                        idx_pool.append(idx)

                    break_counter += 1

                if break_counter >= 1024:
                    break

            else:
                break

            if len(current_batch) >= self.shuffle_batch_size:
                # empty current_batch bucket to batches
                batches.extend(current_batch)
                idx_batch = set()
                current_batch = []

        pbar.close()

        # wait before closing progress bar
        time.sleep(0.3)

        self.samples = batches
        print("idx_pool:", len(idx_pool))
        print("Original Length: {} - Length after Shuffle: {}".format(len(self.train_ids), len(self.samples)))
        print("Break Counter:", break_counter)
        print("Pairs left out of last batch to avoid creating noise:", len(self.train_ids) - len(self.samples))
        print("First Element ID: {} - Last Element ID: {}".format(self.samples[0], self.samples[-1]))


class CVUSADatasetEval(Dataset):

    def __init__(self,
                 data_folder,
                 split,
                 img_type,
                 transforms=None,
                 ):

        super().__init__()

        self.data_folder = data_folder
        self.split = split
        self.img_type = img_type
        self.transforms = transforms

        if split == 'train':
            self.df = pd.read_csv(f'{data_folder}/splits/train-19zl.csv', header=None)
        else:
            self.df = pd.read_csv(f'{data_folder}/splits/val-19zl.csv', header=None)

        self.df = self.df.rename(columns={0: "sat", 1: "ground", 2: "ground_anno"})

        self.df["idx"] = self.df.sat.map(lambda x: int(x.split("/")[-1].split(".")[0]))
        self.crop_dir = f'{self.data_folder}/streetview_crop'
        os.makedirs(self.crop_dir, exist_ok=True)
        self.idx2sat = dict(zip(self.df.idx, self.df.sat))
        self.idx2ground = dict(zip(self.df.idx, self.df.ground))

        if self.img_type == "reference":
            self.images = self.df.sat.values
            self.label = self.df.idx.values

        elif self.img_type == "query":
            self.images = self.df.ground.values
            self.label = self.df.idx.values
        else:
            raise ValueError("Invalid 'img_type' parameter. 'img_type' must be 'query' or 'reference'")

    def __getitem__(self, index):
        import os
        img = cv2.imread(f'{self.data_folder}/{self.images[index]}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.img_type == "query":
            h_g, w_g = img.shape[:2]
            pad_height = (w_g // 2 - h_g) // 2
            width = int(fov / 360 * w_g)
            img, coord = random_crop(img, width, h_g)
            self.idx2ground[index] = os.path.join(self.crop_dir, os.path.basename(self.images[index])).split(self.data_folder)[1]
        # image transforms
        if self.transforms is not None:
            img = self.transforms(image=img)['image']

        label = torch.tensor(self.label[index], dtype=torch.long)

        return img, label

    def __len__(self):
        return len(self.images)


class CVUSADatasetRAEval(Dataset):

    def __init__(self,
                 data_folder,
                 split,
                 img_type,
                 transforms=None,
                 ):

        super().__init__()

        self.data_folder = data_folder
        self.split = split
        self.img_type = img_type
        self.transforms = transforms

        if split == 'train':
            self.df = pd.read_csv(f'{data_folder}/splits/train-19zl.csv', header=None)
        else:
            self.df = pd.read_csv(f'{data_folder}/splits/val-19zl.csv', header=None)

        self.df = self.df.rename(columns={0: "sat", 1: "ground", 2: "ground_anno"})

        self.df["idx"] = self.df.sat.map(lambda x: int(x.split("/")[-1].split(".")[0]))

        self.idx2sat = dict(zip(self.df.idx, self.df.sat))

        if self.img_type == "reference":
            self.images = self.df.sat.values
            self.label = self.df.idx.values
        else:
            raise ValueError("Invalid 'img_type' parameter. 'img_type' must be 'reference'")

    def __getitem__(self, index):
        img = cv2.imread(f'{self.data_folder}/{self.idx2sat[self.idx2label[index]]}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # image transforms
        if self.transforms is not None:
            img = self.transforms(image=img)['image']

        label = torch.tensor(self.idx2label[index], dtype=torch.long)
        # print(1)
        return img, label

    def __len__(self):
        return len(self.idx2label)






