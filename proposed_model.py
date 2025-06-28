import os

import torch
import torch.nn as nn
import math
from torch import nn

from Sub_Functions.Evaluate import main_est_parameters
from Sub_Functions.layer_HGNN import HGNN_conv
from data_loader import *

def Proposed_model(x_train, x_test, y_train, y_test, percentage,DB,epochs,ep):
    from torch import nn
    import torch
    import torch.nn.functional as F
    import math

    print("PositionalEncoding")

    class PositionalEncoding(nn.Module):
        def __init__(self, dim, max_len=6000):
            super().__init__()
            pe = torch.zeros(max_len, dim, dtype=torch.float32)
            position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * -(math.log(10000.0) / dim))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe.unsqueeze(0), persistent=False)

        def forward(self, x):
            return x + self.pe[:, :x.size(1)]


    class MultiHeadSelfAttention(nn.Module):
        def __init__(self, dim, heads=1, dropout=0.0):
            super().__init__()
            self.heads = heads
            self.head_dim = dim // heads
            self.scale = self.head_dim ** -0.5

            self.qkv = nn.Linear(dim, dim * 3, bias=False)
            self.out = nn.Linear(dim, dim, bias=False)

        def forward(self, x):
            B, N, C = x.size()
            qkv = self.qkv(x)  # (B, N, 3*dim)
            qkv = qkv.view(B, N, 3, self.heads, self.head_dim).transpose(1, 3)  # (B, heads, N, 3, head_dim)
            q, k, v = qkv.unbind(dim=2)  # each: (B, heads, N, head_dim)

            # Use fused scaled dot-product attention if available (PyTorch 2.0+)
            attn_output = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)

            out = attn_output.transpose(1, 2).contiguous().view(B, N, C)  # (B, N, dim)
            return self.out(out)

    print("ff")

    class FeedForward(nn.Module):
        def __init__(self, dim, hidden_dim, dropout=0.0):
            super().__init__()
            self.ff = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, dim)
            )

        def forward(self, x):
            return self.ff(x)

    print("TB")

    class TransformerBlock(nn.Module):
        def __init__(self, dim, heads, hidden_dim):
            super().__init__()
            self.norm1 = nn.LayerNorm(dim)
            self.attn = MultiHeadSelfAttention(dim, heads=heads)
            self.norm2 = nn.LayerNorm(dim)
            self.ff = FeedForward(dim, hidden_dim)

        def forward(self, x):
            x = x + self.attn(self.norm1(x))
            x = x + self.ff(self.norm2(x))
            return x

    print("dit")

    class DiffusionTransformer(nn.Module):
        def __init__(self, seq_len, dim=64, depth=1, heads=1, hidden_dim=128, num_classes=2):
            super().__init__()
            self.embedding = nn.Linear(4, dim)
            self.pos_enc = PositionalEncoding(dim, max_len=seq_len)
            self.blocks = nn.ModuleList([
                TransformerBlock(dim, heads, hidden_dim) for _ in range(depth)
            ])
            self.classifier = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, num_classes)
            )

        def forward(self, x):
            x = self.embedding(x.float())  # (B, seq_len, dim)
            x = self.pos_enc(x)  # Add positional encoding
            for block in self.blocks:
                x = block(x)
            x = x.mean(dim=1)  # Global average pooling
            return self.classifier(x)  # (B, num_classes)

    class DNA_HGNN(nn.Module):
        def __init__(self, in_ch=4, hidden_ch=16, num_classes=2, dropout=0.1):
            super(DNA_HGNN, self).__init__()
            self.dropout = dropout
            self.hgc1 = HGNN_conv(in_ch, hidden_ch)
            self.hgc2 = HGNN_conv(hidden_ch, num_classes)

        def forward(self, x, G):
            B, N, feat_dim = x.size()
            out = []
            for i in range(B):  # Each graph processed separately
                xi = x[i]
                Gi = G[i]
                xi = F.relu(self.hgc1(xi, Gi))
                xi = F.dropout(xi, self.dropout, training=self.training)
                xi = self.hgc2(xi, Gi)
                out.append(xi)
            return torch.stack(out, dim=0)  # (B, 10, num_classes)

    class AttentionFusion(nn.Module):
        def __init__(self, dim):
            super(AttentionFusion, self).__init__()
            self.weight_hgnn = nn.Parameter(torch.tensor(0.5))
            self.weight_trans = nn.Parameter(torch.tensor(0.5))

        def forward(self, hgnn_feat, trans_feat):
            fused = self.weight_hgnn * hgnn_feat + self.weight_trans * trans_feat
            return fused

    class FusionGeneExpressionModel(nn.Module):
        def __init__(self, seq_len, hgnn_in, hgnn_hidden, hgnn_classes,
                     transformer_dim=64, fusion_dim=64, num_classes=2):
            super(FusionGeneExpressionModel, self).__init__()

            self.hgnn = DNA_HGNN(in_ch=hgnn_in, hidden_ch=hgnn_hidden, num_classes=hgnn_classes)
            self.transformer = DiffusionTransformer(seq_len=seq_len, dim=transformer_dim, num_classes=num_classes)

            self.hgnn_proj = nn.Linear(hgnn_classes, fusion_dim)
            self.transformer_proj = nn.Linear(3, fusion_dim)

            self.fusion = AttentionFusion(fusion_dim)
            self.classifier = nn.Linear(fusion_dim, num_classes)

        def forward(self, x_seq, x_graph, G, timesteps=None):
            trans_feat = self.transformer(x_seq)  # (B, transformer_dim)
            trans_feat = self.transformer_proj(trans_feat)  # (B, fusion_dim)

            hgnn_feat = self.hgnn(x_graph, G)  # (B, 10, hgnn_classes)
            hgnn_feat = hgnn_feat.mean(dim=1)  # (B, hgnn_classes)
            hgnn_feat = self.hgnn_proj(hgnn_feat)  # (B, fusion_dim)

            fused = self.fusion(hgnn_feat, trans_feat)  # (B, fusion_dim)
            return self.classifier(fused)  # (B, num_classes) (B, num_classes)

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.model_selection import train_test_split

    print("Reading data...")
    x_train_seq=x_train  # (1000, 10, 2400)
    x_test_seq=x_test
    y_train=y_train
    y_test=y_test
    x_train_graph = x_train_seq.reshape((x_train_seq.shape[0], 10, -1))
    x_test_graph = x_test_seq.reshape((x_test_seq.shape[0], 10, -1))
    n_data=ep
    # Compute precomputed G (graph adjacency matrix) using cosine similarity
    print("Precomputing G_small...")
    G_train = []
    G_test=[]
    for i in range(len(x_train_graph)):
        g_feat = torch.tensor(x_train_graph[i], dtype=torch.float32)
        G = F.cosine_similarity(g_feat.unsqueeze(1), g_feat.unsqueeze(0), dim=-1)
        G_train.append(G)

    for i in range(len(x_test_graph)):
        g_feat = torch.tensor(x_test_graph[i], dtype=torch.float32)
        G = F.cosine_similarity(g_feat.unsqueeze(1), g_feat.unsqueeze(0), dim=-1)
        G_test.append(G)

    G_train = torch.stack(G_train)  # (1000, 10, 10)
    G_test=torch.stack(G_test)
    print("G_small precomputed.")




    # Dataset class
    class GeneExpressionDataset(Dataset):
        def __init__(self, seq_data, graph_feat, labels, G_all):
            self.seq_data = seq_data
            self.graph_feat = graph_feat
            self.labels = labels
            self.G_all = G_all

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            seq = torch.tensor(self.seq_data[idx], dtype=torch.float32)
            graph_feat = torch.tensor(self.graph_feat[idx], dtype=torch.float32)
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            G_small = self.G_all[idx]
            return seq, graph_feat, label, G_small

    # Collate function
    def collate_fn(batch):
        seqs, graph_feats, labels, G_smalls = zip(*batch)
        return (
            torch.stack(seqs),  # (B, 6000, 4)
            torch.stack(graph_feats),  # (B, 10, 2400)
            torch.tensor(labels),  # (B,)
            torch.stack(G_smalls)  # (B, 10, 10)
        )

    # DataLoaders
    train_dataset = GeneExpressionDataset(x_train_seq, x_train_graph, y_train, G_train)
    test_dataset = GeneExpressionDataset(x_test_seq, x_test_graph, y_test, G_test)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    # Model init (use your model definition here)
    model = FusionGeneExpressionModel(
        seq_len=6000,
        hgnn_in=x_train_graph.shape[2],  # 2400
        hgnn_hidden=128,
        hgnn_classes=128,
        transformer_dim=128,
        num_classes=3
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    e1=epochs
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)


# Training Loop

    total_epochs=500


    all_y_true = []
    all_y_pred = []

    # Fixed global checkpoint epochs (not percentage-dependent)
    checkpoint_epochs = [100, 200, 300, 400, 500]
    checkpoint_results = {}

    # Define directory for this percentage run
    checkpoint_dir = f"checkpoints/{DB}/{int(percentage)}percent"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Resume from latest available checkpoint within this percentage
    start_epoch = 0
    for ep in reversed(checkpoint_epochs):
        ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_epoch{ep}.pth")
        if os.path.exists(ckpt_path):
            print(f" Resuming from checkpoint: {ckpt_path}")
            checkpoint = torch.load(ckpt_path)
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            start_epoch = ep
            break

    for epoch in range(start_epoch, total_epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        all_y_true.clear()
        all_y_pred.clear()

        for x_seq, x_graph, y, G_small in train_loader:
            x_seq, x_graph, y, G_small = (
                x_seq.to(device),
                x_graph.to(device),
                y.to(device),
                G_small.to(device),
            )

            optimizer.zero_grad()
            outputs = model(x_seq, x_graph, G_small)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)

            all_y_true.extend(y.cpu().numpy())
            all_y_pred.extend(predicted.cpu().numpy())

        acc = 100 * correct / total
        print(f"Epoch {epoch + 1} | Train Accuracy: {acc:.2f}% | Loss: {total_loss / len(train_loader):.4f}")

        # Save metrics and checkpoint at fixed epochs
        if (epoch + 1) in checkpoint_epochs:
            metrics = main_est_parameters(all_y_true, all_y_pred)
            checkpoint_results[f"epoch_{epoch + 1}"] = metrics

            # Save metrics
            metrics_path = f"Analysis1/Performance_Analysis/{DB}/metrics_{int(percentage)}percent_epoch{epoch + 1}.npy"
            os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
            np.save(metrics_path, metrics)
            print(f" Metrics saved to {metrics_path}")

            # Save checkpoint
            ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_epoch{epoch + 1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict()
            }, ckpt_path)
            print(f" Checkpoint saved to {ckpt_path}")

    print(f"\n[✅] Training complete for {percentage}% phase (Total {total_epochs} epochs).")
    return checkpoint_results
