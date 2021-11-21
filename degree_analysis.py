import torch
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import numpy as np
import logging
from config import EDGE_LIST_FILENAME, FEATURE_FILENAME
from config import ACTOR_PATH, CHAMELEON_PATH, CORA_PATH, EASY_PATH

# logger information
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
handler.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


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
                logger.info("unresolvable string : " + line)
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


def get_adjacency_matrix(data: Data) -> np.ndarray:
    node_cnt = data.num_nodes
    adj_matrix = np.zeros((node_cnt, node_cnt))
    node_list0 = data.edge_index[0].tolist()
    node_list1 = data.edge_index[1].tolist()
    for i in range(data.num_edges):
        adj_matrix[node_list0[i]][node_list1[i]] = 1
    return adj_matrix


def cal_avg_clustering_coefficient(data: Data) -> float:
    adj_matrix = get_adjacency_matrix(data)
    all_clustering_coefficient = 0
    for node_id in range(data.num_nodes):
        # 获取当前node_id的所有相邻节点
        neighbor_nodes_list = []
        for j in range(data.num_nodes):
            if adj_matrix[node_id][j] == 1:
                neighbor_nodes_list.append(j)
        node_degree = len(neighbor_nodes_list)
        logger.debug("node_id = {}, list={}".format(node_id, neighbor_nodes_list))
        if node_degree <= 1:
            logger.debug("node_degree = {}, it's cc is undefined".format(node_degree))
            continue
        # 对于所有相邻的节点，判断它们是否相邻
        neighbor_links_cnt = 0
        for i in range(node_degree):
            for j in range(i + 1, node_degree):
                if adj_matrix[neighbor_nodes_list[i]][neighbor_nodes_list[j]] == 1:
                    neighbor_links_cnt += 1
        logger.debug("cc = {}".format(2 * neighbor_links_cnt / (node_degree * (node_degree - 1))))
        all_clustering_coefficient += 2 * neighbor_links_cnt / (node_degree * (node_degree - 1))
    return all_clustering_coefficient / data.num_nodes


if __name__ == '__main__':
    dataset_list = [CORA_PATH, CHAMELEON_PATH, ACTOR_PATH]
    # dataset_list = [EASY_PATH]
    # 图的平均节点度数
    for dataset_name in dataset_list:
        dataset = read_dataset_from_file(dataset_name)
        logger.info("{}'s average degree is {}".format(dataset_name[:-1], cal_avg_degree(dataset)))

    # # 画出度分布直方图
    # for dataset_name in dataset_list:
    #     dataset = read_dataset_from_file(dataset_name)
    #     draw_degree_distribution_histogram(dataset)

    # 计算平均节点聚集系数
    for dataset_name in dataset_list:
        dataset = read_dataset_from_file(dataset_name)
        logger.info("{}'s average clustering coefficient is {}".format(dataset_name[:-1], cal_avg_clustering_coefficient(dataset)))
