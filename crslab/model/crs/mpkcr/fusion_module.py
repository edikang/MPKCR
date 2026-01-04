import torch
import torch.nn as nn
import torch.nn.functional as F


class PreferenceFusion(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.linear = nn.Linear(emb_dim, 1)

    def forward(self, session_emb, knowledge_emb, capsule_emb):
        pooled_list = [torch.mean(e, dim=0) for e in [session_emb, knowledge_emb, capsule_emb]]
        stacked = torch.stack(pooled_list, dim=0)  # [3, emb_dim]
        scores = self.linear(stacked)  # [3, 1]
        weights = F.softmax(scores, dim=0)  # [3, 1]
        fused = (weights * stacked).sum(dim=0)  # [emb_dim]
        return fused






    # def __init__(self, emb_dim):
    #     super().__init__()
    #     # 粒内注意力（保留细粒度信息）
    #     self.intra_attn = nn.MultiheadAttention(emb_dim, num_heads=4)
    #
    #     # 粒间注意力
    #     self.inter_attn = nn.MultiheadAttention(emb_dim, num_heads=4)
    #
    # def forward(self, session_emb, kg_emb, caps_emb):
    #     """
    #     输入形状均为 [seq_len, emb_dim]
    #     """
    #     # 粒内特征增强
    #     session_enhanced, _ = self.intra_attn(session_emb, session_emb, session_emb)
    #     kg_enhanced, _ = self.intra_attn(kg_emb, kg_emb, kg_emb)
    #     caps_enhanced, _ = self.intra_attn(caps_emb, caps_emb, caps_emb)
    #
    #     # 拼接多粒度特征
    #     all_features = torch.cat([session_enhanced, kg_enhanced, caps_enhanced], dim=0)  # [total_len, emb_dim]
    #
    #     # 粒间注意力融合
    #     fused, _ = self.inter_attn(all_features, all_features, all_features)
    #     return fused.mean(dim=0)  # [emb_dim]


class ContextGatedFusion(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.gate_layer = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.Sigmoid()
        )

    def forward(self, fused_preference, context_embedding):
        context_mean = torch.mean(context_embedding, dim=0)  # [emb_dim]
        concat = torch.cat([fused_preference, context_mean], dim=-1)  # [2*emb_dim]
        gate = self.gate_layer(concat)  # [emb_dim]
        user_repr = gate * fused_preference + (1 - gate) * context_mean
        return user_repr

