import cv2
import numpy as np
from torch.utils.data import Dataset
import random
import copy
import torch
from tqdm import tqdm
import time
import scipy.io as sio
import os
from glob import glob
from torchvision import io
import matplotlib.pyplot as plt
from cvcities_base.transforms import get_transforms_train, get_transforms_val
from torch.utils.data import DataLoader

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
    """
    Generate a true weight matrix covering the right-angled triangle directly below
    Args:
        size: Satellite image dimensions (default 256x256)
        epsilon: Minimum non-overlapping area threshold
    Returns:
        mask: Weight matrix [H, W]
    """
    
    mask = np.zeros((size, size), dtype=np.float32)

    cv2.fillPoly(mask, [pts], color=1.0)

    mask[mask < 1.0] = epsilon
    return mask


class CVACTDatasetTrain(Dataset):

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

        self.transforms_query = transforms_query  
        self.transforms_reference = transforms_reference  

        anuData = sio.loadmat(f'{data_folder}/ACT_data.mat')

        ids = anuData['panoIds']

        train_ids = ids[anuData['trainSet'][0][0][1] - 1]

        train_ids_list = []
        train_idsnum_list = []
        self.idx2numidx = dict()
        self.numidx2idx = dict()
        self.ground_path = dict()
        self.idx_ignor = set()
        i = 0

        for idx in train_ids.squeeze():

            idx = str(idx)

            grd_path = f'streetview/{idx}_grdView.jpg'
            sat_path = f'satview_polish/{idx}_satView_polish.jpg'

            if not os.path.exists(f'{self.data_folder}/{grd_path}') or not os.path.exists(
                    f'{self.data_folder}/{sat_path}'):
                self.idx_ignor.add(idx)
            else:
                self.idx2numidx[idx] = i
                self.numidx2idx[i] = idx
                self.ground_path[idx] = grd_path
                train_ids_list.append(idx)
                train_idsnum_list.append(i)
                i += 1

        print("IDs not found in training images:", self.idx_ignor)

        self.train_ids = train_ids_list
        self.train_idsnum = train_idsnum_list
        self.samples = copy.deepcopy(self.train_idsnum)

    def __getitem__(self, index):

        idnum = self.samples[index]

        idx = self.numidx2idx[idnum]

        idx = self.numidx2idx[idnum]
        ground = f'{self.data_folder}/{self.ground_path[idx]}'

        query_img = cv2.imread(ground)
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        h_g, w_g = query_img.shape[:2]
        efficent_img = query_img[h_g//4:h_g//4*3]
        width = int(fov / 360 * w_g)
        query_img, coord = random_crop(efficent_img, width, h_g//2)
        coord[1] += h_g//4
        coord[3] += h_g//4
        # load reference -> satellite image
        reference_img = cv2.imread(f'{self.data_folder}/satview_polish/{idx}_satView_polish.jpg')
        reference_img = cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB)

        h = self.transforms_reference.transforms[1].height
        w = self.transforms_reference.transforms[1].width
        corner = coord2corner(coord)
        corner[[1, 2]] = corner[[2, 1]]
        x, y = ground_to_sate(corner[:, 0:1], corner[:, 1:2], w, h, w_g, w_g // 2)
        sate_coord = np.concatenate([x, y], axis=1)
        sate_corner = sate_coord[0:4].astype(np.int32)
        sate_pos = sate_coord[4:5]
        sate_pos[0, 0] -= w / 2
        sate_pos[0, 1] -= h / 2
        mask = torch.tensor(generate_triangle_mask(sate_corner, size=h))
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
            # rotate sat img 90 or 180 or 270
            reference_img = torch.rot90(reference_img, k=r, dims=(1, 2))
            mask = torch.rot90(mask, k=r, dims=(0, 1))
            rot_mat = np.array(
                [np.cos(r * np.pi / 2), -np.sin(r * np.pi / 2), np.sin(r * np.pi / 2), np.cos(r * np.pi / 2)]).reshape(
                2, 2)
            sate_pos = sate_pos @ rot_mat
        sate_pos = sate_pos.tolist()[0]
        sate_pos = [sate_pos[0] + w / 2.0, sate_pos[1] + h / 2.0]
        label = torch.tensor(idnum, dtype=torch.long)
        return query_img, reference_img, label, sate_pos, mask,os.path.basename(ground)


    def __len__(self):
        return len(self.samples)

    def shuffle(self, sim_dict=None, neighbour_select=64, neighbour_range=128):

        print("\nShuffle Dataset:")

        idx_pool = copy.deepcopy(self.train_idsnum)

        neighbour_split = neighbour_select // 2

        if sim_dict is not None:
            similarity_pool = copy.deepcopy(sim_dict)

        random.shuffle(idx_pool)

        idx_epoch = set()
        idx_batch = set()

        batches = []
        current_batch = []

        break_counter = 0

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

                            if len(current_batch) >= self.shuffle_batch_size:
                                break

                            if idx_near not in idx_batch and idx_near not in idx_epoch:
                                idx_batch.add(idx_near)
                                current_batch.append(idx_near)
                                idx_epoch.add(idx_near)
                                similarity_pool[idx].remove(idx_near)
                                break_counter = 0

                else:
                    if idx not in idx_epoch:
                        idx_pool.append(idx)

                    break_counter += 1

                if break_counter >= 1024:
                    break

            else:
                break

            if len(current_batch) >= self.shuffle_batch_size:
                batches.extend(current_batch)
                idx_batch = set()
                current_batch = []

        pbar.close()

        time.sleep(0.3)

        self.samples = batches
        print("idx_pool:", len(idx_pool))
        print("Original length: {} - Shuffled length: {}".format(len(self.train_ids), len(self.samples)))
        print("Breakpoint counter:", break_counter)
        print("Pairs skipped to avoid noise:", len(self.train_ids) - len(self.samples))
        print("First element ID: {} - Last element ID: {}".format(self.samples[0], self.samples[-1]))


class CVACTDatasetEval(Dataset):

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

        anuData = sio.loadmat(f'{data_folder}/ACT_data.mat')

        ids = anuData['panoIds']

        if split != "train" and split != "val":
            raise ValueError("Invalid 'split' parameter. 'split' must be 'train' or 'val'")

        if img_type != 'query' and img_type != 'reference':
            raise ValueError("Invalid 'img_type' parameter. 'img_type' must be 'query' or 'reference'")

        ids = ids[anuData[f'{split}Set'][0][0][1] - 1]

        ids_list = []
        self.ground_path = dict()
        self.idx2label = dict()
        self.idx_ignor = set()

        i = 0

        for idx in ids.squeeze():

            idx = str(idx)
            grd_path = f'{self.data_folder}/streetview/{idx}_grdView.jpg'
            sat_path = f'{self.data_folder}/satview_polish/{idx}_satView_polish.jpg'


            if not os.path.exists(grd_path) or not os.path.exists(sat_path):
                self.idx_ignor.add(idx)
            else:
                self.idx2label[idx] = i
                self.ground_path[idx] = grd_path
                ids_list.append(idx)
                i += 1

        self.samples = ids_list

    def __getitem__(self, index):

        idx = self.samples[index]

        if self.img_type == "reference":
            path = f'{self.data_folder}/satview_polish/{idx}_satView_polish.jpg'
        elif self.img_type == "query":
            path = self.ground_path[idx]

        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.img_type == "query":
            h_g, w_g = img.shape[:2]
            efficent_img = img[h_g // 4:h_g // 4 * 3]
            width = int(fov / 360 * w_g)
            img, coord = random_crop(efficent_img, width, h_g // 2)
        # image transforms
        if self.transforms is not None:
            img = self.transforms(image=img)['image']

        label = torch.tensor(self.idx2label[idx], dtype=torch.long)

        return img, label

    def __len__(self):
        return len(self.samples)


        