from torch_geometric.nn import SAGEConv

class SAGEModel(torch.nn.Module):
    def __init__(self, input_dim=64, hidden_dim=64, num_classes=3):
        super().__init__()
        self.sage1 = SAGEConv(input_dim, hidden_dim)
        self.sage2 = SAGEConv(hidden_dim, hidden_dim)
        self.classifier = torch.nn.Linear(hidden_dim * 2 + 6, num_classes)  # u+b+6 edge feats
        
    def forward(self, x, edge_index, edge_attr):
        # Node embeddings
        h = self.sage1(x, edge_index).relu()
        h = torch.tanh(h)  # Tanh activation
        h = self.sage2(h, edge_index)
        h = torch.clamp(h, min=0, max=6)  # Clipped ReLU
        
        # Edge prediction
        row, col = edge_index
        edge_emb = torch.cat([h[row], h[col], edge_attr], dim=-1)
        out = self.classifier(edge_emb)
        return out
