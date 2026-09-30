import cv2
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import random
import copy
import torch
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
def compass_to_math_angle(compass_deg):
    """
    将北偏角转换为数学坐标系角度

    Args:
        compass_deg: 北偏角（0-360度，0为正北，顺时针增加）

    Returns:
        数学坐标系角度（0度为正东，逆时针为正），范围0-2π
    """
    # 北偏角: 0°=正北, 90°=正东, 180°=正南, 270°=正西
    # 数学坐标系: 0°=正东, 90°=正北, 180°=正西, 270°=正南
    # 转换公式: 数学角 = 90° - 北偏角
    math_angle_deg = 90 - compass_deg

    # 规范化到0-360度
    math_angle_deg = math_angle_deg % 360

    # 转换为弧度
    return np.deg2rad(math_angle_deg)
def line_intersect_with_image_border(center, angle_rad, img_h, img_w):
    cx, cy = center
    dx = np.cos(angle_rad)
    dy = -np.sin(angle_rad)

    t_vals = []

    if abs(dx) > 1e-9:
        t1 = (0 - cx) / dx   # left
        t2 = (img_w - cx) / dx  # right
        t_vals.extend([t1, t2])
    if abs(dy) > 1e-9:
        t3 = (0 - cy) / dy   # top
        t4 = (img_h - cy) / dy  # bottom
        t_vals.extend([t3, t4])

    t_vals = [t for t in t_vals if t >= 0]

    if not t_vals:
        return (cx, cy)

    t_min = min(t_vals)
    x = cx + t_min * dx
    y = cy + t_min * dy
    return (int(x), int(y))
def is_point_in_fov(px, py, cx, cy, compass_rad, half_fov_rad):
    """判断点 (px, py) 是否在以 (cx,cy) 为中心、方向 compass_rad、半角 half_fov_rad 的扇形内"""
    if px == cx and py == cy:
        return True

    # 计算从中心到该点的角度（弧度）
    dx = px - cx
    dy = py - cy
    angle_to_point = np.arctan2(dy, dx)  # 返回 [-π, π]

    # 转换为顺时针方向（OpenCV 坐标系）：0 表示正北
    angle_to_point = 2 * np.pi - angle_to_point  # 逆时针转为顺时针
    angle_to_point = angle_to_point % (2 * np.pi)

    # 将 compass_rad 也归一化到 [0, 2π)
    compass_rad = compass_rad % (2 * np.pi)

    # 处理角度跨 0° 的情况（如 compass=350°, fov=40° → 范围 [330°, 370°] → [330°, 10°]）
    low = (compass_rad - half_fov_rad) % (2 * np.pi)
    high = (compass_rad + half_fov_rad) % (2 * np.pi)

    if low <= high:
        return low <= angle_to_point <= high
    else:
        # 跨越 0° 边界
        return angle_to_point >= low or angle_to_point <= high

def fill_polygon_from_unordered_points(points, h, w):
    points = np.array(points)
    if len(points) < 3:
        raise ValueError("At least 3 points are required.")
    # 计算凸包
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]  # 按正确顺序排列的点
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.fillPoly(mask, [hull_points.astype(np.int32)], color=1)
    epsilon = 1e-4
    # 非重叠区域设为极小值
    mask[mask < 0.8] = epsilon
    return mask

def unnormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    对tensor执行反标准化操作。

    :param tensor: 经过标准化后的图像tensor (C,H,W)
    :param mean: 标准化时使用的均值
    :param std: 标准化时使用的标准差
    :return: 反标准化后的图像tensor
    """
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor
def generate_mask_from_fov_and_compass(fov, compass, ref_img_shape):
    h, w = ref_img_shape
    cx, cy = w / 2.0, h / 2.0
    compass_rad = compass_to_math_angle(compass)
    half_fov_rad = np.deg2rad(fov / 2.0)
    pt_left = line_intersect_with_image_border((cx, cy), compass_rad - half_fov_rad, h, w)
    pt_right = line_intersect_with_image_border((cx, cy), compass_rad + half_fov_rad, h, w)
    mid_point = line_intersect_with_image_border((cx, cy), compass_rad, h, w)
    mid_point = ((mid_point[0] + cx) // 2 - w // 2, (mid_point[1] + cy) // 2 - h // 2)#转换为中点坐标系
    # 角点
    corners = [(0, 0), (w, 0), (w, h), (0, h)]
    # 4. 筛选出落在扇形内的角点
    inside_corners = []
    for corner in corners:
        if is_point_in_fov(corner[0], corner[1], cx, cy, compass_rad, half_fov_rad):
            inside_corners.append(corner)
    satellite_center = (cx, cy)
    # 5. 构建多边形顶点列表（按顺时针顺序）
    polygon = []

    # 起点：左交点
    polygon.append(pt_left)
    for corner in corners:
        if corner in inside_corners:
            polygon.append(corner)

    # 添加右交点
    polygon.append(pt_right)
    polygon.append(satellite_center)
    mask = fill_polygon_from_unordered_points(polygon, h, w)

    return mask, np.array(mid_point).reshape(1,2)
class CVNAFDatasetTrain(Dataset):

    def _read_cities_csv(self, cities, data_folder):
        all_df = pd.DataFrame()
        for city in cities:
            df = pd.read_csv(f'{data_folder}/{city}/img_info_train.csv', header=None)
            if all_df.empty:
                all_df = df
            else:
                all_df = pd.concat([all_df, df], ignore_index=True)
        return all_df

    def __init__(self,
                 data_folder,
                 transforms_query=None,
                 transforms_reference=None,
                 prob_flip=0.0,
                 prob_rotate=0.0,
                 shuffle_batch_size=128,
                 cities=['taipei', 'maynila']
                 ):

        super().__init__()

        self.data_folder = data_folder
        self.prob_flip = prob_flip
        self.prob_rotate = prob_rotate
        self.shuffle_batch_size = shuffle_batch_size

        self.transforms_query = transforms_query  # ground
        self.transforms_reference = transforms_reference  # satellite

        self.cities = cities
        self.df = self._read_cities_csv(self.cities, data_folder)
        self.df = self.df.rename(columns={0: 'name',  1: 'longitude', 2: 'latitude', 3: 'fov',4: 'compass',5: 'city', 6: 'ground_dir', 7: 'sat_dir'})


        self.idx2sat = dict(zip(self.df.index, self.df.sat_dir))
        # self.idx2ground = dict(zip(self.df.index, self.df.ground_dir,self.df.fov,self.df.compass))
        self.idx2ground = dict(zip(self.df.index, self.df.ground_dir))
        self.pairs = list(zip(self.df.index, self.df.sat_dir, self.df.ground_dir,self.df.fov,self.df.compass))

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
        """
             获取数据集中一个样本的查询图像、参考图像和标签
             Args:
                 index (int): 数据集中的样本索引
             Returns:
                 tuple: 包含查询图像、参考图像和标签的元组
             """
        # 从idx2pair字典中获取样本的索引、卫星图像路径和地面图像路径
        try:
            idx, sat_dir, ground_dir,fov,compass = self.idx2pair[self.samples[index]]
        except:
            print('stop')
        # idx, sat_dir, ground_dir = row.name, row['sat_dir'], row['ground_dir']

        # load query -> ground image 加载查询->地面图像
        query_img = cv2.imread(f'{self.data_folder}/{ground_dir}')
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)

        # load reference -> satellite image 加载参考-> 卫星图像
        reference_img = cv2.imread(f'{self.data_folder}/{sat_dir}')
        reference_img = cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB)
        h = self.transforms_reference.transforms[2].height
        w = self.transforms_reference.transforms[2].width

        # 根据给定的水平视场角（fov）和北偏角（compass），生成掩膜与中心点坐标
        mask, sate_pos = generate_mask_from_fov_and_compass(fov, compass, (h, w))
        mask = torch.tensor(mask)

        # Flip simultaneously query and reference 同时翻转查询和引用
        if np.random.random() < self.prob_flip:  # 如果随机数小于prob_flip，则同时翻转查询图像和参考图像
            query_img = cv2.flip(query_img, 1)
            reference_img = cv2.flip(reference_img, 1)
            mask = torch.flip(mask, [1])
            sate_pos[0, 0] = - sate_pos[0, 0]

            # image transforms 图像转换
        if self.transforms_query is not None:
            query_img = self.transforms_query(image=query_img)['image']

        if self.transforms_reference is not None:
            reference_img = self.transforms_reference(image=reference_img)['image']

        # Rotate simultaneously query and reference 同时旋转查询t图像和引用图像
        if np.random.random() < self.prob_rotate:
            r = np.random.choice([1, 2, 3])
            # rotate sat img 90 or 180 or 270 旋转卫星图像90度、180度或270度
            reference_img = torch.rot90(reference_img, k=r, dims=(1, 2))
            mask = torch.rot90(mask, k=r, dims=(0, 1))
            rot_mat = np.array(
                [np.cos(r * np.pi / 2), -np.sin(r * np.pi / 2), np.sin(r * np.pi / 2), np.cos(r * np.pi / 2)]).reshape(
                2, 2)
            sate_pos = sate_pos @ rot_mat
            # # use roll for ground view if rotate sat view 如果旋转卫星图像，则使用滚动作为地面视图
            # c, h, w = query_img.shape
            # shifts = - w // 4 * r
            # query_img = torch.roll(query_img, shifts=shifts, dims=2)
        sate_pos = sate_pos.tolist()[0]
        sate_pos = [sate_pos[0] + w / 2.0, sate_pos[1] + h / 2.0]
        label = torch.tensor(idx, dtype=torch.long)
        ground_name = [ground_dir]
        # plt.imshow(mask.numpy())
        # plt.scatter(sate_pos[0], sate_pos[1], c='g', s=40)
        # plt.show()
        # x1_vis1 = unnormalize(query_img)
        # x1_vis1 = cv2.cvtColor(x1_vis1.permute(1, 2, 0).detach().cpu().numpy(), cv2.COLOR_BGR2RGB)
        # x1_im = (255 * x1_vis1).astype(np.uint8)
        # plt.imshow(x1_im)
        # plt.show()
        # x1_vis1 = unnormalize(reference_img)
        # x1_vis1 = cv2.cvtColor(x1_vis1.permute(1, 2, 0).detach().cpu().numpy(), cv2.COLOR_BGR2RGB)
        # x1_im = (255 * x1_vis1).astype(np.uint8)
        # plt.imshow(x1_im)
        # plt.scatter(sate_pos[0], sate_pos[1], c='r', s=40)
        # plt.show()

        return query_img, reference_img, label, sate_pos, mask,ground_name

    def __len__(self):
        return len(self.samples)

    def shuffle(self, sim_dict=None, neighbour_select=64, neighbour_range=128):

        '''
        custom shuffle function for unique class_id sampling in batch 自定义用于批次中唯一类ID采样的洗牌函数
        '''

        print("\nShuffle Dataset:")

        idx_pool = copy.deepcopy(self.train_ids)
        neighbour_split = neighbour_select // 2

        if sim_dict is not None:
            similarity_pool = copy.deepcopy(sim_dict)

        # Shuffle pairs order
        random.shuffle(idx_pool)  # 随机洗牌idx_pool

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

                        try:
                            # 尝试获取相似性数据
                            near_similarity = similarity_pool[idx][:neighbour_range]
                        except KeyError as e:
                            # 当指定的索引不存在时
                            print(f"KeyError: 索引 {idx} 不存在于 similarity_pool 中")
                            print(
                                f"similarity_pool 中可用的索引范围: {list(similarity_pool.keys())[:10] if similarity_pool else '空'}...")

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


class CVNAFDatasetEval(Dataset):

    def _read_cities_csv(self, cities, data_folder):
        all_df = pd.DataFrame()
        for city in cities:
            df = pd.read_csv(f'{data_folder}/{city}/img_info_eval.csv', header=None)
            if all_df.empty:
                all_df = df
            else:
                all_df = pd.concat([all_df, df], ignore_index=True)
        return all_df

    def __init__(self,
                 data_folder,
                 split,
                 img_type,
                 transforms=None,
                 train_cities= ['taipei', 'maynila'],
                 test_cities= ['taipei', 'maynila'],
                 ):

        super().__init__()

        self.data_folder = data_folder
        self.split = split
        self.img_type = img_type
        self.transforms = transforms
        self.train_cities = train_cities
        self.test_cities = test_cities

        if split == 'train':
            self.df = self._read_cities_csv(self.train_cities, self.data_folder)
        else:  # test
            self.df = self._read_cities_csv(self.test_cities, self.data_folder)

        self.df = self.df.rename(columns={0: 'name',  1: 'longitude', 2: 'latitude', 3: 'fov',4: 'compass',5: 'city', 6: 'ground_dir', 7: 'sat_dir'})

        self.idx2sat = dict(zip(self.df.index, self.df.sat_dir))
        self.idx2ground = dict(zip(self.df.index, self.df.ground_dir))

        if self.img_type == "reference":
            self.images = self.df.sat_dir.values
            self.label = self.df.index.tolist()

        elif self.img_type == "query":
            self.images = self.df.ground_dir.values
            self.label = self.df.index.tolist()
        else:
            raise ValueError("Invalid 'img_type' parameter. 'img_type' must be 'query' or 'reference'")



    def __getitem__(self, index):

        img = cv2.imread(f'{self.data_folder}/{self.images[index]}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # image transforms
        if self.transforms is not None:
            img = self.transforms(image=img)['image']

        label = torch.tensor(self.label[index], dtype=torch.long)

        return img, label

    def __len__(self):
        return len(self.images)
class CVNAFDatasetSim(Dataset):

    def _read_cities_csv(self, cities, data_folder):
        all_df = pd.DataFrame()
        for city in cities:
            df = pd.read_csv(f'{data_folder}/{city}/img_info_train.csv', header=None)
            if all_df.empty:
                all_df = df
            else:
                all_df = pd.concat([all_df, df], ignore_index=True)
        return all_df

    def __init__(self,
                 data_folder,
                 split,
                 img_type,
                 transforms=None,
                 train_cities= ['taipei', 'maynila'],
                 test_cities= ['taipei', 'maynila'],
                 ):

        super().__init__()

        self.data_folder = data_folder
        self.split = split
        self.img_type = img_type
        self.transforms = transforms
        self.train_cities = train_cities
        self.test_cities = test_cities

        if split == 'train':
            self.df = self._read_cities_csv(self.train_cities, self.data_folder)
        else:  # test
            self.df = self._read_cities_csv(self.test_cities, self.data_folder)

        self.df = self.df.rename(columns={0: 'name',  1: 'longitude', 2: 'latitude', 3: 'fov',4: 'compass',5: 'city', 6: 'ground_dir', 7: 'sat_dir'})

        self.idx2sat = dict(zip(self.df.index, self.df.sat_dir))
        self.idx2ground = dict(zip(self.df.index, self.df.ground_dir))

        if self.img_type == "reference":
            self.images = self.df.sat_dir.values
            self.label = self.df.index.tolist()

        elif self.img_type == "query":
            self.images = self.df.ground_dir.values
            self.label = self.df.index.tolist()
        else:
            raise ValueError("Invalid 'img_type' parameter. 'img_type' must be 'query' or 'reference'")



    def __getitem__(self, index):

        img = cv2.imread(f'{self.data_folder}/{self.images[index]}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # image transforms
        if self.transforms is not None:
            img = self.transforms(image=img)['image']

        label = torch.tensor(self.label[index], dtype=torch.long)

        return img, label

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    s = CVCITIESDatasetTrain(data_folder="E:/datasets/CVCITIES_raw")
    # s = CVCITIESDatasetTrain(data_folder="I:/CVCities",)
    a = s[1]
    print(a)
    s.__getitem__(2)




