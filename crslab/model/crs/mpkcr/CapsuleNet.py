import torch
import torch.nn as nn
import torch.nn.functional as F
class CapsuleLayer(nn.Module):
    def __init__(self, input_dim, num_capsules, capsule_dim,routings=5):
        super(CapsuleLayer, self).__init__()
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.input_dim = input_dim
        self.routings = routings

        # Initialize weights
        # self.W = nn.Parameter(torch.randn(1, num_capsules,capsule_dim,input_dim))

        # 使用xavier_uniform初始化权重
        # self.W = nn.Parameter(torch.empty(1, num_capsules, capsule_dim, input_dim))
        # nn.init.xavier_uniform_(self.W.data)
        self.W = nn.Parameter(torch.empty(num_capsules, input_dim,capsule_dim))
        # nn.init.xavier_uniform_(self.W.data)

        # 使用更好的初始化方法
        nn.init.kaiming_normal_(self.W, mode='fan_out', nonlinearity='relu')

        # 添加正则化
        self.dropout = nn.Dropout(0.3)


    def forward(self, x):
        """
                输入：
                    x: [batch_size, seq_len, input_dim]
                输出：
                    capsule_output: [batch_size, num_capsules, capsule_dim]
                """
        batch_size, seq_len, _ = x.size()

        x = self.dropout(x)

        # 投影到胶囊空间 [batch, seq, num_caps, cap_dim]
        u_hat = torch.einsum('b s d, c d e -> b s c e', x, self.W)

        # 初始化路由logits
        b = torch.zeros(batch_size, seq_len, self.num_capsules, 1, device=x.device)

        # 动态路由过程
        for i in range(self.routings):
            # 计算耦合系数 [batch, seq, num_caps, 1]
            c = F.softmax(b, dim=2)

            # 加权求和 [batch, num_caps, cap_dim]
            s = (c * u_hat).sum(dim=1, keepdim=True)  # [b,1,c,e]

            # 压缩激活
            v = self.squash(s)

            # 更新路由协议(最后一步不需要)
            if i < self.routings - 1:
                agreement = (u_hat * v).sum(dim=-1, keepdim=True)  # [b,s,c,1]
                b += agreement
            return v.squeeze(1)  # [b,c,e]
    def squash(self, s,epsilon=1e-8):
        # squared_norm = torch.sum(s ** 2, dim=-1, keepdim=True)
        # safe_norm = torch.sqrt(squared_norm + epsilon)
        # scale = squared_norm / (1 + squared_norm) / safe_norm
        # v = scale * s
        """改进的squash函数，增加数值稳定性"""
        squared_norm = torch.sum(s ** 2, dim=-1, keepdim=True)
        # 限制最大范数，防止数值不稳定
        squared_norm = torch.clamp(squared_norm, max=50, min=epsilon)
        safe_norm = torch.sqrt(squared_norm)
        scale = squared_norm / (1 + squared_norm + epsilon)
        v = scale * s / (safe_norm + epsilon)
        return v
# 构建胶囊网络模型
class CapsuleNet(nn.Module):
    def __init__(self, input_dim, num_capsules, capsule_dim):
        super(CapsuleNet, self).__init__()
        self.capsule_layer = CapsuleLayer(input_dim,num_capsules,capsule_dim)

    def forward(self, x):
        return self.capsule_layer(x)