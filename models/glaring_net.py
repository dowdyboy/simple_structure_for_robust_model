import torch
import torch.nn as nn
import torch.nn.functional as F

from dowdyboy_lib.model_util import frozen_module, unfrozen_module


class GlaringDetectorHeadV4(nn.Module):

    def __init__(self, feat_channels, feat_index, num_layers, drop_rate=0.1,):
        super(GlaringDetectorHeadV4, self).__init__()
        self.feat_index = feat_index
        self.emb_conv_list = nn.ModuleList([
            nn.Conv2d(feat_channels[0], feat_channels[1], kernel_size=3, stride=2, padding=1),
        ])
        self.pos_emb = nn.Parameter(
            torch.zeros(1, 256 + 1, 128)
        )
        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, 128)
        )
        self.drop_after_pos = nn.Dropout(p=drop_rate)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                GlaringDetectorHeadV3EncoderLayer(128, 4, 128 * 4, drop_rate, )
            )
        self.final_norm = nn.LayerNorm(128, eps=1e-6)
        self.cls_fc = nn.Linear(128, 2)

    def _embedding(self, feat_list):
        emb = self.emb_conv_list[0](feat_list[0])
        emb = emb.view(emb.size(0), emb.size(1), -1).permute(0, 2, 1)
        return emb

    def forward(self, pre_feat_list, ):
        feat_list = []
        for i in range(len(pre_feat_list)):
            if i in self.feat_index:
                feat_list.append(pre_feat_list[i])
        emb = self._embedding(feat_list)
        emb = torch.cat([self.cls_token.expand(emb.size(0), -1, -1), emb], dim=1)
        x = emb + self.pos_emb
        x = self.drop_after_pos(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == len(self.layers) - 1:
                x = self.final_norm(x)
        out_feat, _ = x[:, 0, :], x[:, 1:, :]
        out = self.cls_fc(out_feat)
        return out, out_feat





class GlaringDetectorHeadV3MultiheadAttention(nn.Module):

    def __init__(self, embed_dim, num_head, drop_rate, ):
        super(GlaringDetectorHeadV3MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_head = num_head
        self.head_dim = embed_dim // num_head
        self.scale = self.head_dim ** -0.5  # SA矩阵的缩放
        self.qkv = nn.Linear(self.embed_dim, self.embed_dim * 3, bias=True)
        self.att_drop = nn.Dropout(p=drop_rate)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.proj_drop = nn.Dropout(p=drop_rate)

    def forward(self, x):
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2, -1)) * self.scale
        att = att.softmax(dim=-1)
        att = self.att_drop(att)

        x = (att @ v).transpose(1, 2).reshape(B, N, self.embed_dim)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class GlaringDetectorHeadV3FFN(nn.Module):

    def __init__(self, embed_dim, feedforward_channels, ffn_drop=0.,):
        super(GlaringDetectorHeadV3FFN, self).__init__()
        self.layers = nn.Sequential(*[
            nn.Linear(embed_dim, feedforward_channels, ),
            nn.GELU(),
            nn.Dropout(p=ffn_drop),
            nn.Linear(feedforward_channels, embed_dim, ),
            nn.Dropout(p=ffn_drop),
        ])

    def forward(self, x):
        return self.layers(x)


class GlaringDetectorHeadV3EncoderLayer(nn.Module):

    def __init__(self, embed_dim, num_head, feedforward_channels, drop_rate=0.):
        super(GlaringDetectorHeadV3EncoderLayer, self).__init__()
        self.embed_dim = embed_dim
        self.norm_1 = nn.LayerNorm(embed_dim)
        self.att = GlaringDetectorHeadV3MultiheadAttention(embed_dim, num_head, drop_rate, )
        self.norm_2 = nn.LayerNorm(embed_dim)
        self.ffn = GlaringDetectorHeadV3FFN(embed_dim, feedforward_channels, drop_rate, )

    def forward(self, x):
        x = x + self.att(self.norm_1(x))
        x = self.ffn(self.norm_2(x)) + x
        return x


# 尝试将中间特征图作为词向量，应用于transformer
class GlaringDetectorHeadV3(nn.Module):

    def __init__(self, feat_channels, feat_index, num_layers, drop_rate=0.1):
        super(GlaringDetectorHeadV3, self).__init__()
        self.feat_index = feat_index
        self.emb_conv_list = nn.ModuleList([
            nn.Conv2d(feat_channels[0], feat_channels[1], kernel_size=3, stride=2, padding=1),
            nn.Conv2d(feat_channels[1], feat_channels[1], kernel_size=1, stride=1, padding=0),
            nn.ConvTranspose2d(feat_channels[2], feat_channels[1], kernel_size=2, stride=2, padding=0),
            nn.Sequential(*[
                nn.ConvTranspose2d(feat_channels[3], feat_channels[2], kernel_size=2, stride=2, padding=0),
                nn.ConvTranspose2d(feat_channels[2], feat_channels[1], kernel_size=2, stride=2, padding=0),
            ])
        ])
        self.pos_emb = nn.Parameter(
            torch.zeros(1, 512 + 1, 256)
        )
        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, 256)
        )
        self.drop_after_pos = nn.Dropout(p=drop_rate)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                GlaringDetectorHeadV3EncoderLayer(256, 4, 256 * 4, drop_rate, )
            )
        self.final_norm = nn.LayerNorm(256, eps=1e-6)
        self.cls_fc = nn.Linear(256, 2)

    def _embedding(self, feat_list):
        emb_list = []
        for i in range(len(feat_list)):
            emb_list.append(self.emb_conv_list[i](feat_list[i]))
        emb = torch.cat(emb_list, dim=1)
        emb = emb.view(emb.size(0), emb.size(1), -1)
        return emb

    def forward(self, pre_feat_list, ):
        feat_list = []
        for i in range(len(pre_feat_list)):
            if i in self.feat_index:
                feat_list.append(pre_feat_list[i])
        emb = self._embedding(feat_list)
        emb = torch.cat([self.cls_token.expand(emb.size(0), -1, -1), emb], dim=1)
        x = emb + self.pos_emb
        x = self.drop_after_pos(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == len(self.layers) - 1:
                x = self.final_norm(x)
        out_feat, _ = x[:, 0, :], x[:, 1:, :]
        out = self.cls_fc(out_feat)
        return out, out_feat


# 尝试使用点卷积，学习通道维度特征
class GlaringDetectorHeadV2(nn.Module):

    # 32 16 8 4
    # 64 128 256 512
    def __init__(self, feat_channels, feat_index, feat_len):
        super(GlaringDetectorHeadV2, self).__init__()
        self.feat_index = feat_index
        self.dot_conv_list = nn.ModuleList([nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0) for c in feat_channels])
        self.dense_1 = nn.Linear(feat_len, feat_len // 2)
        self.dense_2 = nn.Linear(feat_len // 2, feat_len // 4)
        self.dense_cls = nn.Linear(feat_len // 4, 2)

    def forward(self, pre_feat_list, ):
        feat_list = []
        for i in range(len(pre_feat_list)):
            if i in self.feat_index:
                feat_list.append(pre_feat_list[i])
        dot_feat_list = []
        for i, feat in enumerate(feat_list):
            dot_feat_list.append(self.dot_conv_list[i](feat))

        avg_feat_list = [F.avg_pool2d(feat, kernel_size=feat.size(-1), ).view(feat.size(0), -1) for feat in dot_feat_list]
        avg_feat = torch.cat(avg_feat_list, dim=1)
        out = self.dense_1(avg_feat)
        # out = F.tanh(out)
        out = F.relu(out)
        out_feat = self.dense_2(out)
        # out = F.tanh(out_feat)
        out = F.relu(out_feat)
        out = self.dense_cls(out)
        return out, out_feat


class GlaringDetectorHead(nn.Module):

    def __init__(self, feat_len, feat_index):
        super(GlaringDetectorHead, self).__init__()
        self.dense_1 = nn.Linear(feat_len, feat_len // 2)
        self.dense_2 = nn.Linear(feat_len // 2, feat_len // 4)
        self.dense_cls = nn.Linear(feat_len // 4, 2)
        self.feat_index = feat_index

    def forward(self, pre_feat_list, ):
        feat_list = []
        for i in range(len(pre_feat_list)):
            if i in self.feat_index:
                feat_list.append(pre_feat_list[i])

        avg_feat_list = [F.avg_pool2d(feat, kernel_size=feat.size(-1), ).view(feat.size(0), -1) for feat in feat_list]
        avg_feat = torch.cat(avg_feat_list, dim=1)
        out = self.dense_1(avg_feat)
        # out = F.tanh(out)
        out = F.relu(out)
        out_feat = self.dense_2(out)
        # out = F.tanh(out_feat)
        out = F.relu(out_feat)
        out = self.dense_cls(out)
        return out, out_feat


class GlaringNet(nn.Module):

    def __init__(self, pre_net, train_net, head):
        super(GlaringNet, self).__init__()
        self.pre_net = pre_net
        self.train_net = train_net
        self.head = head
        frozen_module(self.pre_net)

    def frozen_head(self, is_frozen):
        if is_frozen:
            frozen_module(self.head)
        else:
            unfrozen_module(self.head)

    def forward_for_head(self, x):
        _, pre_feat_list = self.pre_net.feature_list(x)
        out, out_feat = self.head(pre_feat_list)
        return out, out_feat

    def forward_for_train_net(self, x):
        out, train_feat_list = self.train_net.feature_list(x)
        _, out_feat = self.head(train_feat_list)
        return out, out_feat

    def forward(self, x):
        raise NotImplementedError()

