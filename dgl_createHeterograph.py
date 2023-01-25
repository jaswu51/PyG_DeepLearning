import dgl
from dgl.data import DGLDataset
import torch
import os
import pandas as pd
import numpy as np
"""
Parameters
----------
gidx : HeteroGraphIndex
    Graph index object.
ntypes : list of str, pair of list of str
    Node type list. ``ntypes[i]`` stores the name of node type i.
    If a pair is given, the graph created is a uni-directional bipartite graph,
    and its SRC node types and DST node types are given as in the pair.
etypes : list of str
    Edge type list. ``etypes[i]`` stores the name of edge type i.
node_frames : list[Frame], optional
    Node feature storage. If None, empty frame is created.
    Otherwise, ``node_frames[i]`` stores the node features
    of node type i. (default: None)
edge_frames : list[Frame], optional
    Edge feature storage. If None, empty frame is created.
    Otherwise, ``edge_frames[i]`` stores the edge features
    of edge type i. (default: None)
"""


g = dgl.heterograph({('user', 'plays', 'game'): (torch.tensor([0, 1, 1, 2]),
                     torch.tensor([0, 0, 1, 1])),
  ('developer', 'develops', 'game'): (torch.tensor([0, 1]),
                                    torch.tensor([0, 1]))
   })

print(g.nodes('user'))
g.nodes['user'].data['hv'] = torch.rand(3, 6)
g.edges['plays'].data['weight']=  torch.from_numpy(np.array([1,2,0,2.0]))
g.edges['plays'].data['weight']=  torch.from_numpy(np.array([1,2,0.1,2]))
print(g)