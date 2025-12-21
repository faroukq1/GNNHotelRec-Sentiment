import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCNModel(torch.nn.Module):
    def __init__(self, input_dim=64, hidden_dim=64, num_classes=3):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = torch.nn.Linear(hidden_dim * 2, num_classes)  # u + b embeddings
        
    def forward(self, x, edge_index, edge_attr):
        # Node embeddings
        h = self.conv1(x, edge_index)
        h = torch.tanh(h)  # First layer: Tanh (like article)
        h = self.conv2(h, edge_index)
        h = torch.clamp(h, min=0, max=6)  # Second layer: ReLU clipped to 6
        
        # Edge prediction: concatenate source + target node embeddings
        row, col = edge_index
        edge_emb = torch.cat([h[row], h[col]], dim=-1)
        
        # Add edge features
        if edge_attr is not None:
            edge_emb = torch.cat([edge_emb, edge_attr], dim=-1)
            
        out = self.classifier(edge_emb)
        return out
