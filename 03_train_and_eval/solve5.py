"""
solve4_corrected_binary.py
===========================
Edge-level sentiment classification on a user–hotel graph
CORRECTED VERSION: Removes neutral sentiment, maps -1→0 (negative), 1→1 (positive)
"""

# ============================================================
# IMPORTS
# ============================================================

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 80)
print("SOLVE4 CORRECTED — GCN vs GraphSAGE vs GAT")
print("BINARY CLASSIFICATION: Negative (0) vs Positive (1)")
print("Neutral reviews (sentiment_predicted == 0) are REMOVED")
print("=" * 80)

# ============================================================
# STEP 1: LOAD DATA AND FILTER
# ============================================================

df_original = pd.read_csv('../data/tokenized/03_data_cleaned_full_tokens_final_score.csv')

print(f"Original dataset: {len(df_original):,} rows")

# Create a clean copy for binary classification
df = df_original.copy()

# CRITICAL FIX: Remove neutral sentiment (0) and map -1 to 0, 1 to 1
print("\nSentiment distribution BEFORE filtering:")
print(df["sentiment_predicted"].value_counts().sort_index())

# Filter out neutral reviews
df = df[df["sentiment_predicted"] != 0].copy()

print(f"\nAfter removing neutral reviews: {len(df):,} rows")
print("Sentiment distribution AFTER filtering:")
print(df["sentiment_predicted"].value_counts().sort_index())

# Map: -1 (negative) → 0, 1 (positive) → 1
df["sentiment_bin"] = df["sentiment_predicted"].map({-1: 0, 1: 1})

print("\nBinary sentiment distribution:")
print(df["sentiment_bin"].value_counts().sort_index())
print(f"  Class 0 (Negative): {(df['sentiment_bin'] == 0).sum():,} samples")
print(f"  Class 1 (Positive): {(df['sentiment_bin'] == 1).sum():,} samples")
print(f"  Class balance: {(df['sentiment_bin'] == 1).sum() / len(df) * 100:.2f}% positive")

# ============================================================
# STEP 2: NODE INDEXING
# ============================================================

users = df["user_id"].unique()
hotels = df["hotel_id"].unique()

user2id = {u: i for i, u in enumerate(users)}
hotel2id = {h: i + len(users) for i, h in enumerate(hotels)}

num_nodes = len(users) + len(hotels)
print(f"\nNodes: {num_nodes:,} ({len(users):,} users + {len(hotels):,} hotels)")

# ============================================================
# STEP 3: ENHANCED NODE FEATURES
# ============================================================

print("\nBuilding node features...")

# ---------- USER FEATURES (EXPANDED) ----------
user_df = df.groupby("user_id").agg(
    rating_mean=("rating", "mean"),
    rating_std=("rating", "std"),
    rating_min=("rating", "min"),
    rating_max=("rating", "max"),
    review_count=("rating", "count"),
    sentiment_mean=("sentiment_bin", "mean"),
    sentiment_std=("sentiment_bin", "std"),
    camelbert_mean=("camelbert_sentiment_score", "mean"),
    camelbert_std=("camelbert_sentiment_score", "std"),
).fillna(0)

# ---------- HOTEL FEATURES (EXPANDED) ----------
hotel_df = df.groupby("hotel_id").agg(
    rating_mean=("rating", "mean"),
    rating_std=("rating", "std"),
    rating_min=("rating", "min"),
    rating_max=("rating", "max"),
    review_count=("rating", "count"),
    hotel_rating=("hotel_rating", "first"),
    price=("price_dzd", "first"),
    distance=("distance_center_km", "first"),
    sentiment_mean=("sentiment_bin", "mean"),
    sentiment_std=("sentiment_bin", "std"),
).fillna(0)

# Log transform for skewed features
for feat in ["review_count", "price", "distance"]:
    if feat in user_df.columns:
        user_df[feat] = np.log1p(user_df[feat])
    if feat in hotel_df.columns:
        hotel_df[feat] = np.log1p(hotel_df[feat])

sc_user = StandardScaler()
sc_hotel = StandardScaler()

user_x = torch.tensor(sc_user.fit_transform(user_df), dtype=torch.float)
hotel_x = torch.tensor(sc_hotel.fit_transform(hotel_df), dtype=torch.float)

# Pad to same dimension
dim = max(user_x.shape[1], hotel_x.shape[1])
user_x = F.pad(user_x, (0, dim - user_x.shape[1]))
hotel_x = F.pad(hotel_x, (0, dim - hotel_x.shape[1]))

x = torch.cat([user_x, hotel_x], dim=0)

print(f"Node feature dimension: {x.shape[1]}")

# ============================================================
# STEP 4: EDGES
# ============================================================

print("Building edges...")

edges = []
labels = []

for _, r in df.iterrows():
    u = user2id[r.user_id]
    h = hotel2id[r.hotel_id]
    y = r.sentiment_bin

    edges.append([u, h])
    edges.append([h, u])
    labels.extend([y, y])

edge_index = torch.tensor(edges, dtype=torch.long).t()
edge_labels = torch.tensor(labels, dtype=torch.long)

print(f"Total edges: {edge_index.shape[1]:,} (bidirectional)")

# ============================================================
# STEP 5: TRAIN / VAL / TEST SPLIT
# ============================================================

print("\nSplitting data...")

num_edges = edge_index.shape[1] // 2
perm = torch.randperm(num_edges)

train_n = int(0.6 * num_edges)
val_n = int(0.2 * num_edges)

train_mask = torch.zeros(num_edges, dtype=torch.bool)
val_mask = torch.zeros(num_edges, dtype=torch.bool)
test_mask = torch.zeros(num_edges, dtype=torch.bool)

train_mask[perm[:train_n]] = True
val_mask[perm[train_n:train_n + val_n]] = True
test_mask[perm[train_n + val_n:]] = True

print(f"Train edges: {train_mask.sum():,}")
print(f"Val edges: {val_mask.sum():,}")
print(f"Test edges: {test_mask.sum():,}")

graph_data = Data(x=x, edge_index=edge_index, y=edge_labels).to(DEVICE)
train_mask = train_mask.to(DEVICE)
val_mask = val_mask.to(DEVICE)
test_mask = test_mask.to(DEVICE)

print("Graph ready\n")

# ============================================================
# STEP 6: ENHANCED EDGE GNN WITH RESIDUAL CONNECTIONS
# ============================================================

class EdgeGNN(nn.Module):
    def __init__(self, gnn, in_dim, hidden, layers, dropout):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.skip_connections = nn.ModuleList()

        for i in range(layers):
            ic = in_dim if i == 0 else hidden
            if gnn == "GCN":
                conv = GCNConv(ic, hidden)
            elif gnn == "SAGE":
                conv = SAGEConv(ic, hidden)
            elif gnn == "GAT":
                conv = GATConv(ic, hidden // 4, heads=4, concat=True)
            self.convs.append(conv)
            self.bns.append(nn.BatchNorm1d(hidden))
            
            # Skip connection for residual learning
            if i > 0:
                self.skip_connections.append(nn.Linear(hidden, hidden))

        # Enhanced edge MLP with more layers
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden * 2, hidden * 2),
            nn.BatchNorm1d(hidden * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2)
        )
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, (c, b) in enumerate(zip(self.convs, self.bns)):
            x_new = c(x, edge_index)
            x_new = b(x_new)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            
            # Add residual connection after first layer
            if i > 0:
                x = x + self.skip_connections[i-1](x_new)
            else:
                x = x_new
        return x

    def predict(self, z, s, d):
        return self.edge_mlp(torch.cat([z[s], z[d]], dim=1))

# ============================================================
# STEP 7: TRAIN & EVAL WITH DETAILED LOGGING
# ============================================================

def train_epoch(model, opt):
    model.train()
    opt.zero_grad()
    z = model(graph_data.x, graph_data.edge_index)
    idx = torch.where(train_mask)[0]
    s = graph_data.edge_index[0, idx * 2]
    d = graph_data.edge_index[1, idx * 2]
    y = graph_data.y[idx * 2]
    
    logits = model.predict(z, s, d)
    loss = F.cross_entropy(logits, y)
    
    # Calculate training accuracy
    with torch.no_grad():
        pred = logits.argmax(1)
        train_acc = accuracy_score(y.cpu(), pred.cpu())
    
    loss.backward()
    opt.step()
    return loss.item(), train_acc

@torch.no_grad()
def eval_model(model, mask):
    model.eval()
    z = model(graph_data.x, graph_data.edge_index)
    idx = torch.where(mask)[0]
    s = graph_data.edge_index[0, idx * 2]
    d = graph_data.edge_index[1, idx * 2]
    y = graph_data.y[idx * 2]
    logits = model.predict(z, s, d)
    p = logits.argmax(1)
    prob = F.softmax(logits, 1)[:, 1]
    return {
        "acc": accuracy_score(y.cpu(), p.cpu()),
        "prec": precision_score(y.cpu(), p.cpu(), zero_division=0),
        "rec": recall_score(y.cpu(), p.cpu(), zero_division=0),
        "f1": f1_score(y.cpu(), p.cpu(), zero_division=0),
        "auc": roc_auc_score(y.cpu(), prob.cpu())
    }

# ============================================================
# STEP 8: RUN ALL MODELS WITH ENHANCED SETTINGS
# ============================================================

results = {}
HIDDEN = 512
LAYERS = 5
LR = 0.001
EPOCHS = 13

for gnn in ["GCN", "SAGE", "GAT"]:
    print("=" * 70)
    print(f"Training {gnn}")
    print("=" * 70)

    model = EdgeGNN(
        gnn, graph_data.num_node_features,
        HIDDEN, LAYERS, 0.3
    ).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='max', factor=0.5, patience=15
    )
    
    best_f1 = 0
    best_state = None
    patience_counter = 0
    
    print(f"{'Epoch':<8}{'Loss':<10}{'Train Acc':<12}{'Val Acc':<10}{'Val F1':<10}")
    print("-" * 60)

    for ep in range(EPOCHS):
        loss, train_acc = train_epoch(model, opt)
        val = eval_model(model, val_mask)
        
        # Learning rate scheduling
        scheduler.step(val["f1"])
        
        if val["f1"] > best_f1:
            best_f1 = val["f1"]
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Print every epoch
        print(f"{ep+1:<8}{loss:<10.4f}{train_acc:<12.4f}{val['acc']:<10.4f}{val['f1']:<10.4f}")
        
        # Early stopping
        if patience_counter >= 30:
            print(f"Early stopping at epoch {ep+1}")
            break

    model.load_state_dict(best_state)
    results[gnn] = eval_model(model, test_mask)
    torch.save(best_state, f"best_{gnn}_solve4_corrected.pt")
    print()

# ============================================================
# FINAL COMPARISON
# ============================================================

print("=" * 80)
print("FINAL TEST RESULTS (Binary: Negative vs Positive, Neutrals Removed)")
print("=" * 80)
print(f"{'Model':<8}{'Acc':<10}{'Prec':<10}{'Rec':<10}{'F1':<10}{'AUC':<10}")
print("-" * 60)
for k, v in results.items():
    print(f"{k:<8}{v['acc']:<10.4f}{v['prec']:<10.4f}{v['rec']:<10.4f}{v['f1']:<10.4f}{v['auc']:<10.4f}")

print("\n✓ solve4_corrected_binary.py finished")
print("\nKey improvements in this version:")
print("  - Removed neutral sentiment (0) for cleaner binary classification")
print("  - Properly mapped: -1 → 0 (negative), 1 → 1 (positive)")
print("  - This should significantly improve model performance and interpretability")