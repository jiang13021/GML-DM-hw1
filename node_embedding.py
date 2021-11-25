import numpy as np

from ge.classify import read_node_label, Classifier
from ge import Node2Vec, DeepWalk, LINE
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE

from config import EDGE_LIST_FILENAME, LABEL_FILENAME
from config import ACTOR_PATH, CHAMELEON_PATH, CORA_PATH


def read_edge_list(dataset_path):
    return nx.read_edgelist(dataset_path + EDGE_LIST_FILENAME,
                            create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])


def evaluate_embeddings(embeddings, dataset_path):
    X, Y = read_node_label(dataset_path + LABEL_FILENAME)
    tr_frac = 0.8
    print("Training classifier using {:.2f}% nodes...".format(
        tr_frac * 100))
    clf = Classifier(embeddings=embeddings, clf=LogisticRegression())
    return clf.split_train_evaluate(X, Y, tr_frac)


def plot_embeddings(embeddings, dataset_path):
    X, Y = read_node_label(dataset_path + LABEL_FILENAME)

    emb_list = []
    for k in X:
        emb_list.append(embeddings[k])
    emb_list = np.array(emb_list)

    model = TSNE(n_components=2)
    node_pos = model.fit_transform(emb_list)

    color_idx = {}
    for i in range(len(X)):
        color_idx.setdefault(Y[i][0], [])
        color_idx[Y[i][0]].append(i)

    for c, idx in color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    dataset_list = [CORA_PATH, CHAMELEON_PATH, ACTOR_PATH]
    for dataset in dataset_list:
        print("dataset: " + dataset)
        G = read_edge_list(dataset)

        # DeepWalk
        model = DeepWalk(G, walk_length=20, num_walks=50, workers=1)
        model.train(window_size=10, iter=5)
        embeddings = model.get_embeddings()
        deepwalk_result = evaluate_embeddings(embeddings, dataset)
        print("DeepWalk: ", end="")
        print(deepwalk_result)
        # plot_embeddings(embeddings, dataset)

        # Node2Vec
        model = Node2Vec(G, walk_length=20, num_walks=50,
                         p=0.25, q=0.25, workers=1, use_rejection_sampling=1)
        model.train(window_size=10, iter=5)
        embeddings = model.get_embeddings()
        node2vec_result = evaluate_embeddings(embeddings, dataset)
        print("Node2Vec: ", end="")
        print(node2vec_result)
        # plot_embeddings(embeddings, dataset)

        # LINE
        model = LINE(G, embedding_size=128, order='all')
        model.train(batch_size=1024, epochs=150, verbose=0)
        embeddings = model.get_embeddings()
        node2vec_result = evaluate_embeddings(embeddings, dataset)
        print("LINE: ", end="")
        print(node2vec_result)
        plot_embeddings(embeddings, dataset)
