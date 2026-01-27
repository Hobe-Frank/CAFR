import pandas as pd
from sklearn.metrics import DistanceMetric
import torch
import pickle
import os

TRAIN_CITIES = [
    'Barcelona',  # 18364
    'Austria',  # 10586
    'Miami',  # 18178
    'NewYork',  # 9971
]


def read_cities_csv(cities, data_folder):
    all_df = pd.DataFrame()
    for city in cities:
        df = pd.read_csv(f'{data_folder}/{city}/img_info_train.csv', header=None)
        if all_df.empty:
            all_df = df
        else:
            all_df = pd.concat([all_df, df], ignore_index=True)
    return all_df

TOP_K = 128
data_folder = '/root/autodl-tmp/cross_view/CVNAF'

df_train = read_cities_csv(TRAIN_CITIES, data_folder)

df_train = df_train.rename(
    columns={0: 'name',  1: 'longitude', 2: 'latitude', 3: 'fov',4: 'compass',5: 'city', 6: 'sat_dir', 7: 'ground_dir'})

train_sat_ids = df_train.index.tolist()

print("Length Train names:", len(train_sat_ids))
print(f"{str(len(TRAIN_CITIES))}_cities totally")

gps_coords = {}
gps_coords_list = []

for _, df in df_train.iterrows():
    
    coordinates = (float(df.loc["latitude"]), float(df.loc["longitude"]))
    gps_coords[df.name] = coordinates
    gps_coords_list.append(coordinates)
    
print("Length of gps coords : " + str(len(gps_coords_list)))
print("Calculation...")

dist = DistanceMetric.get_metric('haversine')
dm = dist.pairwise(gps_coords_list, gps_coords_list)
print("Distance Matrix:", dm.shape)

dm_torch = torch.from_numpy(dm)
dm_torch = dm_torch.fill_diagonal_(dm.max())

values, ids = torch.topk(dm_torch, k=TOP_K, dim=1, largest=False)

values_near_numpy = values.numpy()
ids_near_numpy = ids.numpy()

near_neighbors = dict()

for index, df in df_train.iterrows():
    # name = df.loc['name']
    near_neighbors[index] = ids_near_numpy[index].tolist()

print("Saving...") 
with open(os.path.join(data_folder, f"gps_dict_{str(len(TRAIN_CITIES))}_cities.pkl"), "wb") as f:
    pickle.dump(near_neighbors, f)
