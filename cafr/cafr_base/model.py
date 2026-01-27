import torch
import timm
import numpy as np
import torch.nn as nn
from cafr_base import helper
from torchsummary import summary
from .backbones.common import rank_print, load_model, get_standard_transform, collate
import math
from einops import rearrange
import torch.nn.functional as F
import torchvision.transforms.functional as ttf

class RadioGeM(nn.Module):  
    """
    Args:
        pl (_type_): _description_
    """

    def __init__(self,
                 # ---- Backbone 主干网络
                 model_name='radio_gem_cafr',
                 backbone_arch='radio_v2.5-h',
                 pretrained=True,
                 vitdet_window_size=None,
                 adaptor_names=None,
                 pathch_size=16,
                 torchhub_repo='NVlabs/RADIO',
                 layers_to_freeze=1,
                 layers_to_crop=[],
                 layer1=30,
                 use_cls=False,
                 norm_descs=True,

                 # ---- Aggregator 聚合方法
                 agg_arch='GeM', 
                 agg_config={},
                 ):
        super(RadioGeM, self).__init__()
        self.pretrained = pretrained  # 是否预训练
        self.layers_to_freeze = layers_to_freeze  # 冻结网络层名称
        self.layers_to_crop = layers_to_crop  # layers_to_crop=[4],  # 4 crops the last resnet layer, 3 crops the 3rd, ...etc
        self.layer1 = layer1
        self.use_cls = use_cls
        self.pathch_size = pathch_size
        self.norm_descs = norm_descs
        self.agg_config = agg_config  # 聚合方法参数
        # self.save_hyperparameters()  # write hyperparams into a file
        self.model_name = model_name

        self.batch_acc = []  # we will keep track of the % of trivial pairs/triplets at the loss level
        self.backbone, self.preprocessor, self.info = load_model(backbone_arch, vitdet_window_size=vitdet_window_size,
                                                                 adaptor_names=adaptor_names,
                                                                 torchhub_repo=torchhub_repo)
        # for name, param in self.backbone.named_parameters():
        for i in range(0, self.layer1 + 1):
            self.backbone.blocks[i].requires_grad_(False)
        self.aggregator = helper.get_aggregator(agg_arch, agg_config)

    # the forward pass of the lightning model
    def forward(self, x):
        feature = self.backbone(x)
        part_feat = self.resize_feature(x, feature)
        glob_feat = self.aggregator(part_feat)
        return part_feat, glob_feat

    def resize_feature(self, x, feature):
        if x.shape[-2] != x.shape[-1]:
            num_rows = x.shape[-2] // self.patch_size
            num_cols = x.shape[-1] // self.patch_size
        else:
            num_rows = int(round(math.sqrt(feature[1].shape[1])))
            num_cols = num_rows
        all_feat = rearrange(feature[1], 'b (h w) c -> b c h w', h=num_rows, w=num_cols).float()
        return all_feat


class RelativePositionEncoder(nn.Module):
    """相对位置编码模块（支持多层级）"""

    def __init__(self, embed_dim, max_size=32):
        super().__init__()
        self.max_size = max_size
        self.embedding = nn.Embedding(2 * max_size + 1, embed_dim)  # 覆盖[-max_size, max_size]范围

    def forward(self, feature_map):
        """
        输入: feature_map - [B, C, H, W]
        输出: position_encoded_key - [B, H*W, C]
        """
        B, C, H, W = feature_map.shape

        # 生成相对坐标网格
        h_coords = torch.arange(H, device=feature_map.device).float() - H // 2  # [-H/2, ..., H/2]
        w_coords = torch.arange(W, device=feature_map.device).float() - W // 2
        h_grid, w_grid = torch.meshgrid(h_coords, w_coords, indexing='ij')  # [H, W]

        # 将坐标限制在[-max_size, max_size]范围内并转换为整数索引

        h_idx = torch.clamp(h_grid, -self.max_size, self.max_size).long() + self.max_size
        w_idx = torch.clamp(w_grid, -self.max_size, self.max_size).long() + self.max_size
        pos_embed = self.embedding(h_idx) + self.embedding(w_idx)  # 合并行列编码
        # 调整维度并广播到batch维度 [B, H*W, C]
        pos_embed = pos_embed.view(1, H * W, C).expand(B, -1, -1)

        return pos_embed


class MultiCrossAttention(nn.Module):
    """动态交叉注意力模块（处理单个层级）"""

    def __init__(self, embed_dim=256, global_dim=1024, max_size=16, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        # # 位置编码器
        self.pos_encoder = RelativePositionEncoder(embed_dim, max_size=max_size)

    def generate_grid(self, H, W):
        # 预生成坐标网格（原图尺度）
        x = (torch.arange(W) + 0.5)
        y = (torch.arange(H) + 0.5)
        grid = torch.stack(torch.meshgrid(x, y, indexing='xy'), dim=-1)
        # 扩展维度以适应批次输入 [1, H, W, 2]
        grid = grid.unsqueeze(0)
        return grid  # [1, H, W, 2]

    def forward(self, street_feat, sat_feat):
        """
        输入:
            street_feat: 街景/BEV特征 [B, C]
            sat_feat: 卫星特征 [B, C, H, W]
        输出:
            regressed_pos: 回归位置 [B, 2] (归一化坐标)
            attention_weights: 注意力权重 [B, H*W]
        """
        B, C, H, W = sat_feat.shape
        device = sat_feat.device
        # # 2. 添加相对位置编码到Key
        pos_embed = self.pos_encoder(sat_feat)  # [B, H*W, C]
        street_feat = street_feat / street_feat.norm(dim=1, keepdim=True)

        sat_feat1 = sat_feat.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        sat_feat1 = sat_feat1 + pos_embed
        sat_feat1 = sat_feat1 / sat_feat1.norm(dim=2, keepdim=True)
        street_feat1 = street_feat.unsqueeze(1)
        scores = torch.matmul(street_feat1, sat_feat1.transpose(-2, -1))
        attn_weights = scores.squeeze(1)
        temperature = 5e-3  # 温度参数
        attn_weights = torch.nn.functional.softmax(attn_weights / temperature,
                                                   dim=1)
        attn_weights = attn_weights.unsqueeze(1)

        grid = self.generate_grid(H, W)  # [H*W, 2]
        grid = grid.repeat(B, 1, 1, 1).view(B, H * W, 2)  # 复制批次次数[B, H, W, 2]
        grid = grid.to(device)
        # 注意力加权坐标
        regressed_pos = torch.einsum('bh,bhc->bc', attn_weights.squeeze(1), grid)
        return regressed_pos, attn_weights

class ContextAwareFeatureRefinement(nn.Module):
    """完整的位置约束模块（多层级）"""

    def __init__(self, embed_dim=256, global_dim=1024, max_size=16, num_heads=8, levels=3, dropout=0.1):
        super().__init__()
        self.levels = levels

        # 多层级注意力模块
        self.attentions = nn.ModuleList([
            MultiCrossAttention(embed_dim, global_dim, max_size, num_heads, dropout)
            for _ in range(levels)
        ])

        # 层级下采样器
        self.downsamplers = nn.ModuleList([
            nn.AvgPool2d(kernel_size=2 ** (1 + i), stride=2 ** (1 + i))
            for i in range(levels - 1)
        ])

    def forward(self, street_feat, sat_feat):
        """
        输入:
            street_feat: 街景/BEV特征 [B, C]
            sat_feat: 原始卫星特征 [B, C, H, W]
        输出:
            all_positions: 各层级回归位置列表 [[B, 2], ...]
            all_weights: 各层级注意力权重列表 [[B, H_i*W_i], ...]
        """
        all_positions = []
        all_weights = []

        for i in range(self.levels):
            # 下采样卫星特征
            if i == 0:
                downsampled = torch.clone(sat_feat)
            else:
                downsampled = self.downsamplers[i - 1](sat_feat)
            _, _, H, W = downsampled.shape

            # 执行动态交叉注意力
            pos, weights = self.attentions[i](street_feat, downsampled)
            all_positions.append(pos)
            all_weights.append(weights)  # 恢复为2D权重图

        return all_positions, all_weights

class SalientRegionSeparationLoss(nn.Module):
    def __init__(self, margin=0.6, reduction='mean'):
        super().__init__()
        self.margin = margin  # 间隔阈值alpha
        self.reduction = reduction  # 损失聚合方式（mean或sum）

    def forward(self, pred_weights, true_mask):
        """
        Args:
            pred_weights: 预测注意力矩阵 [B, H, W]
            true_mask: 真实二值掩膜 [B, H, W]（0或1）
        Returns:
            loss: 标量损失值
        """
        batch_size = pred_weights.shape[0]
        total_loss = 0.0
        data_max = torch.max(pred_weights.view(batch_size, -1), dim=1)[0]
        data_max = data_max.view(batch_size, 1, 1)
        pred_weights = torch.div(pred_weights, data_max)
        for b in range(batch_size):
            # 提取当前样本的正负区域预测值
            pos = pred_weights[b][true_mask[b] > 0.5]  # [N_pos]
            neg = pred_weights[b][true_mask[b] < 0.5]  # [N_neg]
            if len(pos) == 0 or len(neg) == 0:
                continue  # 跳过无正/负样本的情况
            self.margin = 0.5
            # 生成所有正负对（避免显式循环）
            pos = pos.unsqueeze(1)  # [N_pos, 1]
            neg = neg.unsqueeze(0)  # [1, N_neg]
            diff = self.margin - (pos - neg)  # [N_pos, N_neg]

            # 计算单个样本的损失
            sample_loss = torch.sum(F.relu(diff)) / (pos.shape[0] * neg.shape[1] + 1e-8)
            total_loss += sample_loss

        # 按批次聚合损失
        if self.reduction == 'mean':
            return total_loss / batch_size
        else:
            return total_loss


class RadioModel(nn.Module):
    def __init__(self,
                 model_name='radio_gem_cafr',
                 pretrained_path=None,
                 backbone_arch='radio_v2.5-h',
                 pretrained=True,
                 img_size=256,
                 pathch_size=16,
                 # Aggregator 聚合方法
                 agg_arch='GeM',
                 agg_config={},
                 pos_config={},
                 layer=30
                 ):
        super(RadioModel, self).__init__()
        self.img_size = img_size
        if 'radio' in backbone_arch:
            self.model = RadioGeM(backbone_arch=backbone_arch, agg_arch=agg_arch, pathch_size=pathch_size,
                                  agg_config=agg_config, layer1=layer)
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # 2. position prior guided
        self.cafr = ContextAwareFeatureRefinement(**pos_config)
        self.srs = SalientRegionSeparationLoss()
    def get_config(self):
        std = list(self.model.preprocessor.norm_std.cpu().numpy().reshape(1, 3)[0])
        meam = list(self.model.preprocessor.norm_mean.cpu().numpy().reshape(1, 3)[0])
        data_config = {'mean': meam, 'std': std}
        return data_config

    def set_grad_checkpointing(self, enable=True):
        self.model.set_grad_checkpointing(enable)

    def forward(self, img1, img2=None, positions=None, weight=None):
        # img1:query,img2:ref

        if img2 is not None:
            part_feat1, glob_feat1 = self.model(img1)
            part_feat2, glob_feat2 = self.model(img2)
            bs, c, h, w = img1.shape
            # positions是查询图像在参考图像上的真实位置 [x,y]，x向右，y向下
            device = img1.device
            if len(positions):
                scales = [16, 32, 64,128]
                atten_alpha = (np.array([4, 2, 1,0.5])*0.75).tolist()
                positions_gt = []
                weights_gt = []
                for scale in scales:  # 归一化到wxh网格
                    positions_gt.append(positions / scale)
                    weights_gt.append(
                        ttf.resize(weight, (h // scale, w // scale), interpolation=ttf.InterpolationMode.NEAREST).to(
                            device))
                positions_pred, weights = self.apcm(glob_feat1, part_feat2)
                cwr_loss = 0.0
                srs_los = 0.0
                for i in range(len(positions_pred)):
                    cwr_loss += torch.sqrt(F.mse_loss(positions_pred[i].float(), positions_gt[i].float()).float())
                    srs_los += atten_alpha[i] * self.srs(weights[i].view(bs, h // scales[i], w // scales[i]),
                                                                  weights_gt[i])
                cafr_loss = cwr_loss + srs_los
                return glob_feat1, glob_feat2, cafr_loss, weights[0], positions_pred
            else:
                return glob_feat1, glob_feat2, torch.tensor([0, 0], device='cuda', dtype=float)
        else:
            part_feat1, glob_feat1 = self.model(img1)

            return part_feat1, glob_feat1


