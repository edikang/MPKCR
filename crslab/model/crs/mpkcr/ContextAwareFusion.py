import torch
import torch.nn as nn



class ContextAwareFusion(nn.Module):
    """上下文感知的多粒度融合模块"""

    def __init__(self, emb_dim, num_heads=4):
        super().__init__()
        # 多粒度注意力机制
        self.multi_gra_attn = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=num_heads)

        # 门控融合组件
        self.gate_layer = nn.Sequential(
            nn.Linear(emb_dim * 3, emb_dim),
            nn.Sigmoid()
        )

        # 上下文增强投影
        self.context_proj = nn.Linear(emb_dim, emb_dim)

    def forward(self, session_emb, kg_emb, context_emb, factor_emb=None):
        """
        输入：
            session_emb: [seq_len, emb_dim] 会话超图嵌入
            kg_emb: [kg_nodes, emb_dim] 知识超图嵌入
            context_emb: [ctx_len, emb_dim] 对话上下文实体嵌入
            factor_emb: [num_caps, factor_dim] 胶囊网络因子嵌入
        输出：
            fused_emb: [emb_dim] 融合后的用户表示
        """

        # 步骤1：上下文感知的特征增强
        if context_emb is not None and len(context_emb) > 0:
            # 上下文特征投影
            context_emb = context_emb.to(context_emb.device)
            ctx_proj = self.context_proj(context_emb.mean(dim=0))

            # 会话特征增强
            session_emb += ctx_proj.unsqueeze(0)

            # 知识特征增强
            kg_emb += ctx_proj.unsqueeze(0)

        # 步骤2：构建多粒度特征矩阵
        main_features = []

        # 项目级特征（会话超图）
        if session_emb.size(0) > 0:
            item_emb = torch.mean(session_emb, dim=0)  # [emb_dim]
            main_features.append(item_emb)

        # 因子级特征（胶囊网络）
        if factor_emb is not None:
            factor_emb = factor_emb.squeeze(0)
            factor_agg = factor_emb.mean(dim=0)  # [emb_dim]
            main_features.append(factor_agg)

        # 知识级特征（知识超图）
        if kg_emb.size(0) > 0:
            kg_agg = torch.mean(kg_emb, dim=0)  # [emb_dim]
            main_features.append(kg_agg)

        # 组合主特征 [3, emb_dim]
        main_feature_matrix = torch.stack(main_features, dim=0)
        # 步骤3：多粒度交叉注意力
        if context_emb is not None and len(context_emb) > 0:
            # 使用上下文作为query
            attn_output, _ = self.multi_gra_attn(
                query=main_feature_matrix.unsqueeze(1),
                key=main_feature_matrix.unsqueeze(1),
                value=main_feature_matrix.unsqueeze(1)
            )
            attended_features = attn_output.mean(dim=0)  # [1, emb_dim]
        else:
            # 自注意力模式
            attn_output, _ = self.multi_gra_attn(
                query=main_feature_matrix.unsqueeze(1),
                key=main_feature_matrix.unsqueeze(1),
                value=main_feature_matrix.unsqueeze(1)
            )
            attended_features = attn_output.mean(dim=0)  # [1, emb_dim]

        # 步骤4：门控残差融合
        gate_input = torch.cat([
            main_feature_matrix.mean(dim=0),  # 主特征均值
            attended_features.squeeze(0),  # 注意力特征
            ctx_proj if context_emb is not None else torch.zeros_like(item_emb)
        ], dim=0)

        fusion_gate = self.gate_layer(gate_input)
        final_emb = fusion_gate * main_feature_matrix.mean(dim=0) + (1 - fusion_gate) * attended_features.squeeze(0)

        return final_emb