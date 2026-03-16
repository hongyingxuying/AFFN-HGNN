import pywt
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
import torch.nn.functional as F

from sklearn.neighbors import NearestNeighbors

def generate_time_freq_images(data, wavelet='db1', maxlevel=3):
    images = []
    # wavelet = 'db1'
    # maxlevel = np.min([pywt.dwt_max_level(data_len=len(x), filter_len=pywt.Wavelet(wavelet).dec_len), 3])
    for segment in data:
        wp = pywt.WaveletPacket(data=segment, wavelet=wavelet, mode='symmetric', maxlevel=maxlevel)
        # 获取所有叶子节点数据
        leaf_nodes = wp.get_leaf_nodes(decompose=True)
        values = np.array([node.data for node in leaf_nodes])
        images.append(values)
    return images

# 假设 data 是一个形状为 (N, 1024) 的 NumPy 数组，其中N是段的数量
# data = np.random.rand(100, 1024)  # 示例数据
# images = generate_time_freq_images(data)


def create_graph_data(images, n_neighbors=5):
    # 将时频图“展平”以用作特征
    features = [img.flatten() for img in images]

    # 使用k-NN找到最近的邻居
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(features)
    distances, indices = nbrs.kneighbors(features)

    # 创建边和边的权重
    edges = []
    edge_weights = []
    for i, neighbors in enumerate(indices):
        for neighbor in neighbors:
            if i != neighbor:  # 排除自身
                edges.append((i, neighbor))
                edge_weights.append(1)  # 所有边的权重都是一样的

    return edges, edge_weights, features


# 假设 images 是之前步骤生成的时频图列表
# edges, edge_weights, features = create_graph_data(images)


class GAT(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_features, 8, heads=8, dropout=0.6)
        self.conv2 = GATConv(8*8, num_classes, concat=False, dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


def main():
    signal_size = 1024
    root = r'../GNNforIFD/data/PU/data/PU'

    # 加载数据
    data = np.load('../GNNforIFD/data/0_test.npy')
    images = generate_time_freq_images(data)

    edges, edge_weights, features = create_graph_data(images, n_neighbors=5)
    x = torch.tensor(features, dtype=torch.float)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # 假设您已经根据您的数据标记了训练集和测试集，以及类别标签
    num_classes = 2  # 示例：假设有4个类别
    data = Data(x=x, edge_index=edge_index)  # 实际应用中还需要y（标签）和train_mask

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GAT(num_features=x.shape[1], num_classes=num_classes).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    # 训练模型（简化版本，实际应用中需要完整的训练/测试循环）
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data)
        # 这里省略了loss计算和backward，因为没有定义y和train_mask
        optimizer.step()

if __name__ == '__main__':
    main()

