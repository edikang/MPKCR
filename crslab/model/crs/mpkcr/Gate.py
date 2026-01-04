import torch
import torch.nn as nn


class Gate(nn.Module):
    def __init__(self,emd_size):
        super(Gate, self).__init__()
        self.gate = nn.Linear(emd_size * 2, emd_size)

    def forward(self, session_embedding, caps_embedding):
        # 将两个嵌入向量拼接起来
        combined_embed = torch.cat((session_embedding, caps_embedding), dim=1)

        # 通过门控网络计算融合权重
        fusion_weights = torch.sigmoid(self.gate(combined_embed))

        # 使用融合权重对嵌入向量进行加权融合
        fusion_embed = fusion_weights * session_embedding + (1 - fusion_weights) * caps_embedding

        return fusion_embed
