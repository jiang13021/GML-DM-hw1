import torch
from torch_geometric.data import Data
import bisect

from config import EDGE_LIST_FILENAME, FEATURE_FILENAME
import matplotlib.pyplot as plt
from config import ACTOR_PATH, CHAMELEON_PATH, CORA_PATH


def read_dataset_from_file(dataset_path: str) -> Data:
    """
    Read edge information from dataset and put the data into torch_geometric.data.Data
    :param dataset_path: the path of the dataset
    :return: torch_geometric.data.Data
    """
    edge_index_data = []
    with open(dataset_path + EDGE_LIST_FILENAME, "r") as edge_list_file:
        for line in edge_list_file.readlines():
            edge = [int(s) for s in line.split() if s.isdigit()]
            if len(edge) != 2:
                print("unresolvable string : " + line)
                continue
            edge_index_data.append(edge)
    edge_index = torch.tensor(edge_index_data, dtype=torch.long)
    x_data = []
    with open(dataset_path + FEATURE_FILENAME, "r") as feature_file:
        for line in feature_file.readlines():
            x_data.append([float(s) for s in line.split()])
    x = torch.tensor(x_data, dtype=torch.float)
    return Data(x=x, edge_index=edge_index.t().contiguous())


def cal_avg_degree(data: Data) -> float:
    """
    :param data: graph data
    :return: The average degree of the graph
    """
    if data.is_undirected():
        return 2 * data.num_edges / data.num_nodes
    return data.num_edges / data.num_nodes


def draw_degree_distribution_histogram(data: Data):
    edge_cnt = data.num_edges
    node_cnt = data.num_nodes
    # node_id 到 degree 的映射
    degree_dict = {}
    node_list0 = data.edge_index[0].tolist()
    for i in range(edge_cnt):
        degree_dict.setdefault(node_list0[i], 0)
        degree_dict[node_list0[i]] += 1
    # degree 到 出现频数（frequency）的映射
    degree_frequency = {}
    for v in degree_dict.values():
        degree_frequency.setdefault(v, 0)
        degree_frequency[v] += 1
    # k 到 P(k) 的映射
    degree_distribution = {}
    for k, v in degree_frequency.items():
        # 只保留频率大于0.01的数据
        if v / node_cnt > 0.01:
            degree_distribution[k] = v / node_cnt
    # 绘图
    plt.bar(list(degree_distribution.keys()), list(degree_distribution.values()))
    plt.xlabel("degree: k")
    plt.ylabel("P(k)")
    plt.show()


if __name__ == '__main__':
    dataset_list = [CORA_PATH, CHAMELEON_PATH, ACTOR_PATH]
    # 图的平均节点度数
    for dataset_name in dataset_list:
        dataset = read_dataset_from_file(dataset_name)
        print("{}'s average degree is {}".format(dataset_name, cal_avg_degree(dataset)))
