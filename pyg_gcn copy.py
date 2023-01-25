import networkx as nx
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import HeteroData
import copy
# importing pandas
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Union
import pandas as pd
import networkx as nx
from scipy.spatial import distance
import pickle
import torch.nn.functional as F


# read text file into pandas DataFrame
df = pd.read_csv("data/Ave.txt", sep=",",header=0)

# preprocessing the df
namelist=  ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear']
indexlist=[*range(0, 78, 1)]
mapdict = {indexlist[i]: namelist[i] for i in range(len(indexlist))}

df['path'] = df['path'].str.extract(r'(\d+)(?!.*\d)')
df['path'].astype(int)
df['video_no'].astype(int)
df['x1'].astype(float)
df['x2'].astype(float)
df['y1'].astype(float)
df['y2'].astype(float)
df['class_name']=df['detclass']
df['class_name']=df.class_name.map(mapdict)
df['centroid']= list(zip((df['x1'] + df['x2'])*0.5, (df['y1'] + df['y2'])*0.5))
node_features=np.array(df.drop(['centroid', 'path','class_name','node_id','video_amount','frame_amount','video_no','frame_no','conf'], axis=1)).astype(float)
node_labels=torch.ones(node_features.shape[0])
#.type(torch.LongTensor)
weight_list=[]
edgeTemporal_from=[]
edgeTemporal_to=[]
edgeTemporal_weight=[]
edgeSpatial_from=[]
edgeSpatial_to=[]
edgeSpatial_weight=[]
# print(df.columns.get_loc('node_id'))
for i in range(node_features.shape[0]):
  for j in range(node_features.shape[0]):
    if i!=j:
      if df.iloc[j,3]-df.iloc[i,3]==1 and df.iloc[j,5]-df.iloc[i,5]==0:
        edgeTemporal_from.append(i)
        edgeTemporal_to.append(j)
        edgeTemporal_weight.append(distance.euclidean(df.iloc[i,-1],df.iloc[j,-1]))
      elif df.iloc[j,3]-df.iloc[i,3]==0:
        edgeSpatial_from.append(i)
        edgeSpatial_to.append(j)
        edgeSpatial_weight.append(distance.euclidean(df.iloc[i,-1],df.iloc[j,-1]))

data = HeteroData()
data['node'].x = torch.from_numpy(node_features).to(torch.float32)
data['node'].id = torch.from_numpy(np.array(df.index))
data['node'].y = node_labels
data[('node','temporal_link','node')].edge_index = torch.tensor([edgeTemporal_from,edgeTemporal_to])
data[('node','spatial_link','node')].edge_index = torch.tensor([edgeSpatial_from,edgeSpatial_to])
data[('node','temporal_link','node')].edge_attr = torch.tensor(edgeTemporal_weight)
data[('node','spatial_link','node')].edge_attr = torch.tensor(edgeSpatial_weight)


train_mask = torch.rand(data.num_nodes) < 0.8
test_mask = ~train_mask
data['node'].train_mask=train_mask
data['node'].test_mask=test_mask

import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG
from torch_geometric.nn import SAGEConv, to_hetero

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(-1, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self):
        x, edge_index = data.x_dict, data.edge_index_dict
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return self.softmax(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data =  data.to(device)

model = GNN(hidden_channels=3, out_channels=1)
model = to_hetero(model, data.metadata(), aggr='sum')



def train():
    model.train()
    optimizer.zero_grad()
    out = model()
    mask = data['node'].train_mask
    print(out['node'][mask])
    print(data['node'].y[mask].reshape(-1,1))
    loss = F.cross_entropy(out['node'][mask], data['node'].y[mask].reshape(-1,1))
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test():
  model.eval()
  logits = model()
  mask1 = data['train_mask']
  pred1 = logits[mask1].max(1)[1]
  acc1 = pred1.eq(data.y[mask1]).sum().item() / mask1.sum().item()
  mask = data['test_mask']
  pred = logits[mask].max(1)[1]
  acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
  return acc1,acc


optimizer_name = "Adam"
lr = 1e-1
optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)
epochs = 200

for epoch in range(1, epochs):
  train()

train_acc,test_acc = test()

print('#' * 70)
print('Train Accuracy: %s' %train_acc )
print('Test Accuracy: %s' % test_acc)
print('#' * 70)