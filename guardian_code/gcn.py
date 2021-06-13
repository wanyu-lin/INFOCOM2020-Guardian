import json
import time
import torch
import random
import numpy as np
import pandas as pd
from tqdm import trange
import torch.nn.init as init
from torch.nn import Parameter
import torch.nn.functional as F
from mean import scatter_add, scatter_mean
from utils import calculate_auc, setup_features
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from convolution import ConvolutionBase, ConvolutionBase_in_out, ConvolutionDeep, ConvolutionDeep_in_out, ListModule

class GraphConvolutionalNetwork(torch.nn.Module):
    """
    Graph Convolutional Network Class.
    """
    def __init__(self, device, args, X, num_labels):
        super(GraphConvolutionalNetwork, self).__init__()
        """
        GCN Initialization.
        :param device: Device for calculations.
        :param args: Arguments object.
        :param X: Node features.
        :param num_labels: Number of labels
        """
        self.args = args
        torch.manual_seed(self.args.seed)
        self.device = device
        self.X = X
        self.dropout = self.args.dropout
        self.num_labels = num_labels
        self.setup_layers()

    def setup_layers(self):
        """
        Adding Base Layers, Deep GraphSAGE layers and Regression Parameters if the model is not a single layer model.
        """
        self.nodes = range(self.X.shape[0])
        self.neurons = self.args.layers
        self.layers = len(self.neurons)
        # Base SAGE class for the first layer of the model.
        self.aggregators = []
        if self.args.both_degree:
            self.base_aggregator = ConvolutionBase_in_out(self.X.shape[1]*4, self.neurons[0],self.num_labels).to(self.device)
            for i in range(1,self.layers):
                # Deep SAGE class for multi-layer models.
                self.aggregators.append(ConvolutionDeep_in_out(4*self.neurons[i-1], self.neurons[i], self.num_labels).to(self.device))
        else:
            self.base_aggregator = ConvolutionBase(self.X.shape[1]*2+self.num_labels, self.neurons[0]).to(self.device)
            for i in range(1,self.layers):
                # Deep SAGE class for multi-layer models.
                self.aggregators.append(ConvolutionDeep(2*self.neurons[i-1]+self.num_labels, self.neurons[i]).to(self.device))

        self.aggregators = ListModule(*self.aggregators)
        self.regression_weights = Parameter(torch.Tensor(2*self.neurons[-1], self.num_labels))
        # self.regression_weights = Parameter(torch.Tensor(self.neurons[-1], 1+self.num_labels))
        init.xavier_normal_(self.regression_weights) # initialize regression_weights

    def calculate_loss_function(self, z, sorted_train_edges, target):
    # def calculate_loss_function(self, z, master_edge, apprentice_edge, target):
        """
        Calculating the embedding losses, regression loss and weight regularization loss.
        :param z: Node embedding.
        :param train_edges
        :param target: Target vector.
        :return loss: Value of loss.
        """

        features = torch.tensor([]).to(self.device)
        nolink = torch.tensor([]).to(self.device)

    
        for label in range(self.num_labels):
            edge = sorted_train_edges[label]
            start_node, end_node = z[edge[:,0],:],z[edge[:,1],:]
            node_node = torch.cat((start_node, end_node),1)
            features = torch.cat((features, node_node))
            surrogates = [random.choice(self.nodes) for node in range(edge.shape[0])]
            surr_nodes = z[surrogates,:]
            if self.args.nolink_size == 1:
                nolink = torch.cat((nolink, torch.cat((start_node, surr_nodes),1)))
            elif self.args.nolink_size == 2:
                nolink = torch.cat((nolink, torch.cat((start_node, surr_nodes),1), torch.cat((end_node, surr_nodes),1)))

        features = torch.cat((features, nolink))
        predictions = torch.mm(features,self.regression_weights)

        #deal with imbalance data, default is false
        class_weight = 1/np.bincount(target.cpu())*features.size(0)
        self.class_weight = torch.FloatTensor(class_weight)

        self.prediction = F.log_softmax(predictions, dim=1)

        
        loss_term = F.nll_loss(self.prediction, target)

        return loss_term

    def forward(self, train_edges, sorted_train_edges, y, y_train):
    # def forward(self, train_edges, master_edge, apprentice_edge, y, y_train):
        """
        Model forward propagation pass. Can fit deep and single layer SGCN models.
        :param edges: edges
        :param y: Target vectors.
        :return loss: Loss value.
        :return self.z: Hidden vertex representations.
        """
        self.h = []
        self.X = F.dropout(self.X, self.dropout, training=self.training)

        self.h.append(torch.tanh(self.base_aggregator(self.X, train_edges, y_train)))
        for i in range(1,self.layers):
            self.h[-1] = F.dropout(self.h[-1], self.dropout, training=self.training)
            self.h.append(torch.tanh(self.aggregators[i-1](self.h[i-1], train_edges, y_train)))
        self.z = self.h[-1]
        loss = self.calculate_loss_function(self.z, sorted_train_edges, y)
        return loss, self.z

class GCNTrainer(object):
    """
    Object to train and score the SGCN, log the model behaviour and save the output.
    """
    def __init__(self, args, edges):
        """
        Constructing the trainer instance and setting up logs.
        :param args: Arguments object.
        :param edges: Edge data structure
        """
        self.args = args
        self.edges = edges 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_start_time = time.time()
        self.setup_logs()

    def setup_logs(self):
        """
        Creating a log dictionary for recording performance.
        """
        self.logs = {}
        self.logs["parameters"] =  vars(self.args)
        self.logs["performance"] = [["Epoch","AUC","F1_micro", "F1_macro","F1_weighted","MAE", "RMSE"]]
        self.logs["training_time"] = [["Epoch","Seconds"]]

    def setup_dataset(self):
        """
        Creating train and test split.
        """
        if self.args.stratified_split:
            stratSplit = StratifiedShuffleSplit(n_splits=1, test_size=self.args.test_size)
            for train_index, test_index in stratSplit.split(self.edges["edges"], self.edges["labels"]):
                self.train_edges, self.test_edges = self.edges["edges"][train_index], self.edges["edges"][test_index]
                self.y_train, self.y_test = self.edges["labels"][train_index], self.edges["labels"][test_index]
        else:
            self.train_edges, self.test_edges, self.y_train, self.y_test = train_test_split(self.edges["edges"], self.edges["labels"], 
                                                                                            test_size = self.args.test_size)

        self.ecount = len(self.train_edges)
        self.train_edges = np.array(self.train_edges)
        self.num_labels = np.shape(self.y_train)[1]

        self.X = setup_features(self.args)
        row, col = torch.tensor(np.transpose(self.train_edges)).long()
        self.train_trust_out = scatter_mean(torch.tensor(self.y_train), row, dim=0, dim_size=np.shape(self.X)[0])
        self.train_trust_in = scatter_mean(torch.tensor(self.y_train), col, dim=0, dim_size=np.shape(self.X)[0])

        self.sorted_train_edges = []
        index = []
        for label in range(self.num_labels):
            label_index = [i for i in range(len(self.y_train)) if self.y_train[i][label] == 1]
            self.sorted_train_edges.append(self.train_edges[label_index,:])
            index.append(len(label_index))

        self.y = []
        for i in range(len(index)):
            self.y = np.append(self.y, [i]*index[i])
        self.y = np.append(self.y, [len(index)]*self.args.nolink_size*self.ecount)
        self.y = torch.from_numpy(self.y).type(torch.LongTensor).to(self.device)

        self.train_edges = torch.from_numpy(np.array(self.train_edges, dtype=np.int64).T).type(torch.long).to(self.device)
        self.y_train = torch.from_numpy(np.array(self.y_train, dtype=np.float32)).type(torch.float).to(self.device)
        self.num_labels = torch.from_numpy(np.array(self.num_labels, dtype=np.int64)).type(torch.long).to(self.device)
        # self.sorted_train_edges = torch.tensor(self.sorted_train_edges).type(torch.long).to(self.device)

        self.X = torch.from_numpy(self.X).to(self.device)

    def create_and_train_model(self):
        """
        Model training and scoring.
        """
        print("\nTraining started.\n")
        # X: node features as a numpy array (training set)
        self.model = GraphConvolutionalNetwork(self.device, self.args, self.X, self.num_labels).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        self.model.train()
        self.epochs = trange(self.args.epochs, desc="Loss")
        for epoch in self.epochs:
            start_time = time.time()
            self.optimizer.zero_grad()
            loss, _ = self.model(self.train_edges, self.sorted_train_edges, self.y, self.y_train)
            loss.backward()
            self.epochs.set_description("TrustGCN (Loss=%g)" % round(loss.item(),4))
            self.optimizer.step()
            self.logs["training_time"].append([epoch+1,time.time()-start_time])
            if self.args.test_size >0:
                self.score_model(epoch)
        self.logs["training_time"].append(["Total",time.time()-self.global_start_time])

    def score_model(self, epoch):
        """
        Score the model on the test set edges in each epoch.
        :param epoch: Epoch number. 
        """
        loss, self.train_z = self.model(self.train_edges, self.sorted_train_edges, self.y, self.y_train)
        score_edges = torch.from_numpy(np.array(self.test_edges, dtype=np.int64).T).type(torch.long).to(self.device)
        test_z = torch.cat((self.train_z[score_edges[0,:],:], self.train_z[score_edges[1,:],:]),1)
        scores = torch.mm(test_z, self.model.regression_weights.to(self.device))

        probability_scores = torch.exp(F.softmax(scores, dim=1))
        #print (probability_scores[1:10,:])
        predictions = probability_scores[:,0]/probability_scores[:,0:2].sum(1)
        #probability_scores = torch.FloatTensor(probability_scores)
        predictions = predictions.cpu().detach().numpy()



        auc, f1_micro, f1_macro,f1_weighted, mae, rmse = calculate_auc(probability_scores, predictions, self.y_test, self.edges)

        self.logs["performance"].append([epoch+1, auc, f1_micro, f1_macro,f1_weighted, mae, rmse])

    def save_model(self):
        """
        Saving the embedding and model weights.
        """
        print("\nEmbedding is saved.\n")
        self.train_z = self.train_z.cpu().detach().numpy()
        embedding_header = ["id"] + ["x_" + str(x) for x in range(self.train_z.shape[1])]
        self.train_z = np.concatenate([np.array(range(self.train_z.shape[0])).reshape(-1,1),self.train_z],axis=1)
        self.train_z = pd.DataFrame(self.train_z, columns = embedding_header)
        self.train_z.to_csv(self.args.embedding_path, index = None)
        print("\nRegression weights are saved.\n")
        self.regression_weights = self.model.regression_weights.cpu().detach().numpy().T
        regression_header = ["x_" + str(x) for x in range(self.regression_weights.shape[1])]
        self.regression_weights = pd.DataFrame(self.regression_weights, columns = regression_header)
        self.regression_weights.to_csv(self.args.regression_weights_path, index = None)     