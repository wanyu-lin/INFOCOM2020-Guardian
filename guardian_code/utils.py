import json
import torch
import numpy as np
import pandas as pd
import networkx as nx
from math import sqrt
from scipy import sparse
from texttable import Texttable
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable() 
    t.add_rows([["Parameter", "Value"]] +  [[k.replace("_"," ").capitalize(),args[k]] for k in keys])
    print(t.draw())

def read_graph(args):
    """
    Method to read graph and create a target matrix with pooled adjacency matrix powers up to the order.
    :param args: Arguments object.
    :return edges: Edges dictionary.
    """
    edges = {}
    ecount = 0
    ncount = []
    edg = []
    lab = []
    with open (args.edge_path) as dataset:
        for edge in dataset:
            ecount += 1
            ncount.append(edge.split()[0])
            ncount.append(edge.split()[1])
            edg.append(list(map(float, edge.split()[0:2])))
            lab.append(list(map(float, edge.split()[2:])))
    edges["labels"] = np.array(lab)
    edges["edges"] = np.array(edg)
    edges["ecount"] = ecount
    edges["ncount"] = len(set(ncount))
    return edges

def setup_features(args):
    """
    Setting up the node features as a numpy array.
    :param args: Arguments object.
    :return X: Node features.
    """
    feature = []
    with open(args.features_path) as vec:
    	for node in vec:
    		feature.append(node.split()[1:])
    embedding = np.array(feature, np.float32)
    if args.normalize_embedding:
        return embedding / np.linalg.norm(embedding)
    else:
        return embedding

def calculate_auc(scores, prediction, label, edge):
    label_vector = [i for line in label for i in range(len(line)) if line[i] == 1]
    """
    label_vector = []
    for line in label:
        if line[0] == 0.9:
            label_vector.append(0)
        if line[0] == 0.63:
            label_vector.append(1)
        if line[0] == 0.36:
            label_vector.append(2)
        if line[0] == 0.09:
            label_vector.append(3)
    """
    # f1_vector = [0 if item == 1 else 1 for item in label_vector]
    
    val, prediction_vector = torch.narrow(scores, 1, 0, len(label[0])).max(1)
    auc = accuracy_score(label_vector, prediction_vector.cpu())

    f1_micro = f1_score(label_vector, prediction_vector.cpu(), average="micro") # average="weighted"
    f1_macro = f1_score(label_vector, prediction_vector.cpu(), average="macro")
    f1_weighted =f1_score(label_vector, prediction_vector.cpu(), average="weighted")

    mae_convert = {0:0.9, 1:0.7, 2:0.4, 3:0.1}
    label_mae = [mae_convert[a] for a in label_vector]
    prediction_mae = [mae_convert[a] for a in prediction_vector.cpu().numpy()]

    neg_ratio = sum(edge["labels"][:,1])/edge["ecount"]
    print(sum(edge["labels"][:,1]))
    print(edge["ecount"])
    label_vector = [0 if i == 1 else 1 for i in label_vector]
    f1 = f1_score(label_vector, [1 if p > neg_ratio else 0 for p in  prediction])

    mae = mean_absolute_error(label_mae, prediction_mae)
    rmse = sqrt(mean_squared_error(label_mae, prediction_mae))
    # print(label_vector)
    # print(prediction_vector)
    # print([1 if p > neg_ratio else 0 for p in  prediction])
    # print(f1_vector)
    # print(prediction_vector.sum())
    return auc, f1_micro, f1_macro, f1_weighted, f1, rmse

def score_printer(logs):
    """
    Print the performance for every 10th epoch on the test dataset.
    :param logs: Log dictionary.
    """
    t = Texttable() 
    t.add_rows([per for i, per in enumerate(logs["performance"]) if i % 10 == 0])
    print(t.draw())

def best_printer(log):
    t = Texttable()
    t.add_rows([per for per in log])
    print(t.draw())

"""
def save_logs(args, logs):

    #Save the logs at the path.
    #:param args: Arguments objects.
    #:param logs: Log dictionary.

    with open(args.log_path,"w") as f:
            json.dump(logs,f)
"""