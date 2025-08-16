import torch
from torch import Tensor
from torch_scatter import scatter_sum
from torch_sparse import SparseTensor
import torch.nn as nn
from torch_geometric.nn import MetaLayer
from typing import Union
import torchvision.transforms as T
from PIL import Image
import numpy as np
import networkx as nx
from scipy.spatial import cKDTree

## Definition of a MLP
class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 128,
        hidden_layers: int = 2,
        activation: str = "ReLU",
        initializer: None | str = None,
        norm_type: None | str = "LayerNorm",
    ):
        """
        Flexible Multi-layer perceptron.
        """

        super(MLP, self).__init__()
        self.activation = getattr(nn, activation)()
        if initializer is not None:
            self.initializer = getattr(nn.init, initializer)
        layers = [nn.Linear(in_dim, hidden_dim), self.activation]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), self.activation]
        layers.append(nn.Linear(hidden_dim, out_dim))

        if norm_type is not None:
            assert norm_type in [
                "LayerNorm",
                "BatchNorm1d",
            ]
            norm_layer = getattr(nn, norm_type)
            layers.append(norm_layer(out_dim))

        self.model = nn.Sequential(*layers)

        if initializer is not None:
            params = self.model.parameters()
            for param in params:
                if param.requires_grad and len(param.shape) > 1:
                    self.initializer(param)

    def forward(self, x: Tensor):
        return self.model(x.float())




## Definition of Processors
# Edge processor :

class EdgeProcessor(nn.Module):
    def __init__(
        self,
        in_dim_node: int,
        in_dim_edge: int,
        hidden_dim: int = 128,
        hidden_layers: int = 2,
        activation: str = "ReLU",
        initializer: None | str = None,
        norm_type: None | str = "LayerNorm",
    ):
        """
        Edge processor for the GraphNet block. This step processes the edge features.
        """

        super(EdgeProcessor, self).__init__()
        self.edge_processor = MLP(
            2 * in_dim_node + in_dim_edge,
            in_dim_edge,
            hidden_dim,
            hidden_layers,
            activation,
            initializer,
            norm_type,
        )

    def forward(self, src, dest, edge_attr, u = None, batch = None):
        out = torch.cat(
            [src, dest, edge_attr], -1
        )  # concatenate source node, destination node, and edge embeddings
        out = self.edge_processor(out)
        out += edge_attr  # residual connection

        return out
    

# Node processor:

class NodeProcessor(nn.Module):
    def __init__(
        self,
        in_dim_node: int,
        in_dim_edge: int,
        hidden_dim: int = 128,
        hidden_layers: int = 2,
        activation: str = "ReLU",
        initializer: None | str = None,
        norm_type: None | str = "LayerNorm",
    ):
        """
        Node processor from the GraphNet block. This step processes the node features.
        """

        super(NodeProcessor, self).__init__()
        self.node_processor = MLP(
            in_dim_node + in_dim_edge,
            in_dim_node,
            hidden_dim,
            hidden_layers,
            activation,
            initializer,
            norm_type,
        )

    def forward(
        self, x: Tensor, edge_index: Union[Tensor, SparseTensor], edge_attr: Tensor, u = None, batch = None
    ):
        _, col = edge_index
        out = scatter_sum(edge_attr, col, dim=0)  # aggregation
        out = torch.cat([x, out], dim=-1)
        out = self.node_processor(out)
        out += x  # residual connection

        return out
    

# Graph processor:


def build_GN_block(
    in_dim_node: int,
    in_dim_edge: int,
    hidden_dim_node: int = 128,
    hidden_dim_edge: int = 128,
    hidden_layers_node: int = 2,
    hidden_layers_edge: int = 2,
    activation: str = "ReLU",
    initializer: None | str = None,
    norm_type: None | str = "LayerNorm",
):
    """
    Builds a Braph Network processor block with the previously defined EdgeProcessor and NodeProcessor.


    Parameters
    ----------
    in_dim_node : int
        Input dimension of node features
    in_dim_edge : int
        Input dimension of edge features
    hidden_dim_node : int, optional
        Width of hidden layers of node processor, by default 128
    hidden_dim_edge : int, optional
        Width of hidden layers of edge processor, by default 128
    hidden_layers_node : int, optional
        Number of hidden layers of node processor, by default 2
    hidden_layers_edge : int, optional
        Number of hidden layers of edge processor, by default 2
    activation : str, optional
        Activation functions, by default "ReLU"
    initializer : None | str, optional
        Initialization method, by default None
    norm_type : None | str, optional
        Normalization method, by default "LayerNorm"
    """
    return MetaLayer(
        edge_model=EdgeProcessor(
            in_dim_node,
            in_dim_edge,
            hidden_dim_edge,
            hidden_layers_edge,
            activation,
            initializer,
            norm_type,
        ),
        node_model=NodeProcessor(
            in_dim_node,
            in_dim_edge,
            hidden_dim_node,
            hidden_layers_node,
            activation,
            initializer,
            norm_type,
        ),
    )


class GraphProcessor(nn.Module):
    def __init__(
        self,
        n_iterations: int,
        in_dim_node: int,
        in_dim_edge: int,
        hidden_dim_node: int = 128,
        hidden_dim_edge: int = 128,
        hidden_layers_node: int = 2,
        hidden_layers_edge: int = 2,
        activation: str = "ReLU",
        initializer: None | str = None,
        norm_type="LayerNorm",
    ):
        """
        Graph processor
        n_iterations: number of message-passing iterations (graph processor blocks)
        in_dim_node: input node feature dimension
        in_dim_edge: input edge feature dimension
        hidden_dim_node: number of nodes in a hidden layer for graph node processing
        hidden_dim_edge: number of nodes in a hidden layer for graph edge processing
        hidden_layers_node: number of hidden layers for graph node processing
        hidden_layers_edge: number of hidden layers for graph edge processing
        activation: activation function
        initializer: weight initializer
        norm_type: normalization type; one of 'LayerNorm', 'BatchNorm1d' or None
        """
        super(GraphProcessor, self).__init__()

        self.blocks = nn.ModuleList()
        for _ in range(n_iterations):
            self.blocks.append(
                build_GN_block(
                    in_dim_node,
                    in_dim_edge,
                    hidden_dim_node,
                    hidden_dim_edge,
                    hidden_layers_node,
                    hidden_layers_edge,
                    activation,
                    initializer,
                    norm_type,
                )
            )

    def forward(self, x, edge_index, edge_attr):
        for block in self.blocks:
            x, edge_attr, _ = block(x, edge_index, edge_attr)
        return x, edge_attr
    


## Global GNN:

class GraphNet(nn.Module):
    def __init__(
        self,
        **kwargs,
    ):
        super(GraphNet, self).__init__()

        # shapes
        num_global_features = kwargs.get("num_global_features", 0)
        num_local_features = kwargs.get("num_local_features", 1)
        space_dim = kwargs.get("space_dim", 3)
        in_dim_node = num_local_features + num_global_features
        in_dim_edge = 1 + space_dim
        out_dim = kwargs.get("out_channels", 1)
        n_blocks= kwargs.get("n_blocks", 10)
        out_dim_node= kwargs.get("out_dim_node", 128)
        out_dim_edge= kwargs.get("out_dim_edge", 128)
        # hidden dim
        hidden_dim_node= kwargs.get("hidden_dim_node", 128)
        hidden_dim_edge= kwargs.get("hidden_dim_edge", 128)
        hidden_dim_decoder= kwargs.get("hidden_dim_decoder", 128)
        hidden_dim_processor_node= kwargs.get("hidden_dim_processor_node", 128)
        hidden_dim_processor_edge= kwargs.get("hidden_dim_processor_edge", 128)
        # hidden layers
        hidden_layers_node= kwargs.get("hidden_layers_node", 2)
        hidden_layers_edge= kwargs.get("hidden_layers_edge", 2)
        hidden_layers_decoder= kwargs.get("hidden_layers_decoder", 2)
        hidden_layers_processor_node= kwargs.get("hidden_layers_processor_node", 2)
        hidden_layers_processor_edge= kwargs.get("hidden_layers_processor_edge", 2)
        # MLP param
        norm_type= kwargs.get("norm_type", "LayerNorm")
        activation= kwargs.get("activation", "ReLU")
        initializer= kwargs.get("initializer", None)

        self.name = "GraphNet"

        self.node_encoder = MLP(
            in_dim_node,
            out_dim_node,
            hidden_dim_node,
            hidden_layers_node,
            activation = activation,
            initializer = initializer,
            norm_type = norm_type,
        )
        self.edge_encoder = MLP(
            in_dim_edge,
            out_dim_edge,
            hidden_dim_edge,
            hidden_layers_edge,
            activation = activation,
            initializer = initializer,
            norm_type = norm_type,
        )
        self.graph_processor = GraphProcessor(
            n_blocks,
            out_dim_node,
            out_dim_edge,
            hidden_dim_processor_node,
            hidden_dim_processor_edge,
            hidden_layers_processor_node,
            hidden_layers_processor_edge,
            activation = activation,
            initializer = initializer,
            norm_type = norm_type,
        )
        self.node_decoder = MLP(
            out_dim_node,
            out_dim,
            hidden_dim_decoder,
            hidden_layers_decoder,
            norm_type=None,
        )

    def forward(self, x, pos, edge_index):
        # Edge Features
        pos_j = torch.clone(pos)
        dist = torch.sum(torch.abs(pos_j[edge_index[1, :]] - pos[edge_index[0, :]]), dim=1)
        relat_pos = pos_j[edge_index[1, :]] - pos[edge_index[0, :]]
        edge_attr = torch.cat([relat_pos, dist.unsqueeze(1)], dim=1)
        # Processing
        #out = self.node_encoder(torch.cat([pos, x], dim=1))
        out = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        out, _ = self.graph_processor(out, edge_index, edge_attr)
        out = self.node_decoder(out)
        return out

class LinearClassifier(nn.Module):
    def __init__(self):
        super(LinearClassifier, self).__init__()
        self.fc1 = nn.Linear(in_features=196*128, out_features=128)  # Adjusted in_features to 128
        self.fc2 = nn.Linear(in_features=128, out_features=32)   # Adjusted out_features for better processing
        self.fc3 = nn.Linear(in_features=32, out_features=2)    # Final output for 2 classes
        self.relu = nn.ReLU()  # ReLU should be used as a function, not as a module

    def forward(self, x):
        #x = x.view(x.size(0), -1)  # Flatten the tensor while keeping the batch dimension
        x = self.relu(self.fc1(x))  # Apply ReLU as a function
        x = self.relu(self.fc2(x))  # Apply ReLU as a function
        x = torch.sigmoid(self.fc3(x))  # Use sigmoid for binary classification
        return x
    
class CombinedModel(nn.Module):
    def __init__(self, graph_net: GraphNet):
        super(CombinedModel, self).__init__()
        self.graph_net = graph_net
        self.classifier = LinearClassifier()

    def forward(self, x, pos, edge_index):
        x = self.graph_net(x, pos, edge_index)
        x = x.flatten() 
        x = self.classifier(x)
        return x

