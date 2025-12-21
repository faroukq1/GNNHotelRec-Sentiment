from torch_geometric.nn import GATConv

class GATModel(torch.nn.Module):
    def __init__(self, input_dim=64, hidden_dim=32, num_classes=3):
        super().__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads=4, concat=True, dropout=0.1)
        self.gat2 = GATConv(hidden_dim * 4, hidden_dim, heads=1, concat=False, dropout=0.1)
        self.classifier = torch.nn.Linear(hidden_dim * 2 + 6, num_classes)
        
    def forward(self, x, edge_index, edge_attr):
        # Node embeddings with attention
        h = self.gat1(x, edge_index)
        h = torch.tanh(h)
        h = self.gat2(h, edge_index)
        h = torch.clamp(h, min=0, max=6)
        
        # Edge prediction
        row, col = edge_index
        edge_emb = torch.cat([h[row], h[col], edge_attr], dim=-1)
        out = self.classifier(edge_emb)
        return out
