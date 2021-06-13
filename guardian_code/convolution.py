import math
import torch
import torch.nn.functional as F
from torch.nn import Parameter
from mean import scatter_add, scatter_mean
from loop import remove_self_loops, add_self_loops

def uniform(size, tensor):
    """
    Uniform weight initialization.
    :param size: Size of the tensor.
    :param tensor: Tensor initialized.
    """
    stdv = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)

class ListModule(torch.nn.Module):
    """
    Abstract list layer class.
    """
    def __init__(self, *args):
        """
        Model initializing.
        """
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        """
        Getting the indexed layer.
        """
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        """
        Iterating on the layers.
        """
        return iter(self._modules.values())

    def __len__(self):
        """
        Number of layers.
        """
        return len(self._modules)

class Convolution(torch.nn.Module):
    """
    Abstract Signed SAGE convolution class.
    :param in_channels: Number of features.
    :param out_channels: Number of filters.
    :param norm_embed: Normalize embedding -- boolean.
    :param bias: Add bias or no.
    """
    def __init__(self,
                 in_channels, # 256 (node_feature dim * 2)
                 out_channels,
                  num_labels, # 32 (dim of neurons)
                 aggregation_mean=True,
                 norm_embed=False,
                 bias=True):
        super(Convolution, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggregation_mean = aggregation_mean
        self.norm_embed = norm_embed
        self.num_labels = num_labels
        self.weight = Parameter(torch.Tensor(self.in_channels, out_channels)) 
        self.trans_weight = Parameter(torch.Tensor(self.num_labels, int(self.in_channels/4)))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize parameters.
        """
        size = self.weight.size(0)
        size2 = self.trans_weight.size(0)
        uniform(size, self.weight)
        uniform(size, self.bias)
        uniform(size2, self.trans_weight)

    def __repr__(self):
        """
        Create formal string representation.
        """
        return "{}({}, {})".format(self.__class__.__name__, self.in_channels, self.out_channels)

class ConvolutionBase(Convolution):
    """
    Base Signed SAGE class for the first layer of the model.
    """
    def forward(self, x, edge_index, edge_label):
        """
        Forward propagation pass with features an indices.
        :param x: node feature matrix.
        :param edge_index: Indices.
        """
        row, col = edge_index
        if self.aggregation_mean:
            opinion = scatter_mean(edge_label, row, dim=0, dim_size=x.size(0))
            out = scatter_mean(x[col], row, dim=0, dim_size=x.size(0))
        else:
            opinion = scatter_add(edge_label, row, dim=0, dim_size=x.size(0))
            out = scatter_add(x[col], row, dim=0, dim_size=x.size(0))

        # out has 2* columes
        out = torch.cat((out,opinion,x),1)
        out = torch.matmul(out, self.weight)
        
        if self.bias is not None:
            out = out + self.bias
        if self.norm_embed:
            out = F.normalize(out, p=2, dim=-1)
        return out

class ConvolutionDeep(Convolution):
    """
    Deep Signed SAGE class for multi-layer models.
    """
    def forward(self, x, edge_index, edge_label):
        """
        Forward propagation pass with features an indices.
        :param x: Features from pervious layer
        :param edge_index
        :return out: Abstract convolved features.
        """
        row, col = edge_index
        
        if self.aggregation_mean:
            opinion = scatter_mean(edge_label, row, dim=0, dim_size=x.size(0))
            out = scatter_mean(x[col], row, dim=0, dim_size=x.size(0))
        else:
            opinion = scatter_add(edge_label, row, dim=0, dim_size=x.size(0))
            out = scatter_add(x[col], row, dim=0, dim_size=x.size(0))
            
        out = torch.cat((out,opinion,x),1)
        out = torch.matmul(out, self.weight)
        if self.bias is not None:
            out = out + self.bias

        if self.norm_embed:
            out = F.normalize(out, p=2, dim=-1)
        return out

class ConvolutionBase_in_out(Convolution):
    """
    Base Signed SAGE class for the first layer of the model.
    """
    def forward(self, x, edge_index, edge_label):
        """
        Forward propagation pass with features an indices.
        :param x: node feature matrix.
        :param edge_index: Indices.
        """
    
        row, col = edge_index

        edge_label_trans = torch.matmul(edge_label, self.trans_weight)

        if self.aggregation_mean:
            opinion = scatter_mean(edge_label_trans, row, dim=0, dim_size=x.size(0))
            out = scatter_mean(x[col], row, dim=0, dim_size=x.size(0))
            inn_opinion = scatter_mean(edge_label_trans, col, dim=0, dim_size=x.size(0))
            inn = scatter_mean(x[row], col, dim=0, dim_size=x.size(0))
        else:
            opinion = scatter_add(edge_label_trans, row, dim=0, dim_size=x.size(0))
            out = scatter_add(x[col], row, dim=0, dim_size=x.size(0))
            inn_opinion = scatter_add(edge_label_trans, col, dim=0, dim_size=x.size(0))
            inn = scatter_add(x[row], col, dim=0, dim_size=x.size(0))

        out = torch.cat((out,opinion,inn,inn_opinion),1)
        out = torch.matmul(out, self.weight)
        
        if self.bias is not None:
            out = out + self.bias
        if self.norm_embed:
            out = F.normalize(out, p=2, dim=-1)
        return out

class ConvolutionDeep_in_out(Convolution):
    """
    Deep Signed SAGE class for multi-layer models.
    """
    def forward(self, x, edge_index, edge_label):
        """
        Forward propagation pass with features an indices.
        :param x: Features from pervious layer
        :param edge_index
        :return out: Abstract convolved features.
        """
        row, col = edge_index

        edge_label_trans = torch.matmul(edge_label, self.trans_weight)
        if self.aggregation_mean:
            opinion = scatter_mean(edge_label_trans, row, dim=0, dim_size=x.size(0))
            out = scatter_mean(x[col], row, dim=0, dim_size=x.size(0))
            inn_opinion = scatter_mean(edge_label_trans, col, dim=0, dim_size=x.size(0))
            inn = scatter_mean(x[row], col, dim=0, dim_size=x.size(0))
        else:
            opinion = scatter_add(edge_label_trans, row, dim=0, dim_size=x.size(0))
            out = scatter_add(x[col], row, dim=0, dim_size=x.size(0))
            inn_opinion = scatter_add(edge_label_trans, col, dim=0, dim_size=x.size(0))
            inn = scatter_add(x[row], col, dim=0, dim_size=x.size(0))

        out = torch.cat((out,opinion,inn,inn_opinion),1)
        out = torch.matmul(out, self.weight)
        
        if self.bias is not None:
            out = out + self.bias

        if self.norm_embed:
            out = F.normalize(out, p=2, dim=-1)
        return out