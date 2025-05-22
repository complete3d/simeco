import torch
import torch.nn as nn
from functools import partial
from timm.models.layers import trunc_normal_
from .build import MODELS
from models.Transformer_utils import *
from models.vec_layers import *
from utils.misc import *
from pointnet2_ops import pointnet2_utils



class VecSelfAttnBlockApi(nn.Module):
    def __init__(
        self, 
        dim: int, 
        num_heads: int, 
        mlp_ratio: float = 4.0, 
        qkv_bias: bool = False, 
        drop: float = 0.0, 
        attn_drop: float = 0.0, 
        init_values=None,
        scale_factor=None,
        drop_path: float = 0.0, 
        act_layer=None,  
        norm_layer=nn.LayerNorm, 
        block_style: str = "vnattn-vngraph",  
        combine_style: str = "concat",
        k: int = 10, 
        n_group: int = 2,
        mode: str = "so3",
        bias_epsilon: float = 1e-6
    ):
        super().__init__()

        if combine_style != "concat":
            raise ValueError(f"Unsupported combine_style: {combine_style}, only 'concat' is allowed.")

        self.norm1 = VecLayerNorm(dim, mode="sim3", eps=bias_epsilon)
        self.norm2 = VecLayerNorm(dim, mode="sim3", eps=bias_epsilon) 
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values is not None else nn.Identity()
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values is not None else nn.Identity()
        self.scale_factor = scale_factor

        if act_layer is None:
            act_layer = nn.LeakyReLU(negative_slope=0.2) 

        self.mlp = nn.Sequential(
            VecLinear(v_in=dim, v_out=int(dim * mlp_ratio), mode="so3", bias_epsilon = bias_epsilon),
            VecActivation(int(dim * mlp_ratio), act_func=act_layer, mode="so3", bias_epsilon = bias_epsilon),
            VecLinear(v_in=int(dim * mlp_ratio), v_out=dim, mode="so3", bias_epsilon = bias_epsilon),
        )      

        block_tokens = block_style.split('-')
        if not (0 < len(block_tokens) <= 2):
            raise ValueError(f"Invalid block_style: {block_style}.")

        self.attn = None
        self.local_attn = None

        for block_token in block_tokens:
            if block_token == "vnattn":
                self.attn = VNAttention(dim, num_heads=num_heads, mode= "so3", bias_epsilon=bias_epsilon)
            elif block_token == "vngraph":
                self.local_attn = VNDynamicGraphAttention(dim, k=k, bias_epsilon=bias_epsilon)
            else:
                raise ValueError(f"Unexpected block_token: {block_token}. Supported: 'vnattn', 'vngraph'.")

        self.block_length = len(block_tokens)

        if self.attn and self.local_attn:
            self.merge_map = VecLinear(v_in=dim * 2, v_out=dim, mode="so3", bias_epsilon=bias_epsilon)

    def forward(self, x: torch.Tensor, pos: torch.Tensor, idx: torch.Tensor = None) -> torch.Tensor:
        attn_features = []

        norm_x = self.norm1(x)

        if self.attn:
            attn_features.append(self.attn(norm_x))
        if self.local_attn:
            attn_features.append(self.local_attn(norm_x, pos, idx=idx))

        if len(attn_features) == 1:
            x = x + self.ls1(transform_restore(x, attn_features[0], self.scale_factor))
        elif len(attn_features) == 2:
            merged_features = torch.cat(attn_features, dim=-2)
            merged_features = self.merge_map(merged_features.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            x = x + self.ls1(transform_restore(x, merged_features, self.scale_factor))
        else:
            raise RuntimeError(f"Unexpected number of attention features: {len(attn_features)}")

        x = x + self.ls2(transform_restore(x, self.mlp(self.norm2(x).permute(0, 2, 3, 1)).permute(0, 3, 1, 2), self.scale_factor))
        return x

class VecCrossAttnBlockApi(nn.Module):
    def __init__(
        self, 
        dim: int, 
        num_heads: int, 
        mlp_ratio: float = 4.0, 
        qkv_bias: bool = False, 
        drop: float = 0.0, 
        attn_drop: float = 0.0, 
        init_values=None,
        scale_factor=None,
        drop_path: float = 0.0, 
        act_layer=None, 
        norm_layer=nn.LayerNorm, 
        self_attn_block_style: str = "attn-deform", 
        self_attn_combine_style: str = "concat",
        cross_attn_block_style: str = "attn-deform", 
        cross_attn_combine_style: str = "concat",
        k: int = 10, 
        n_group: int = 2,
        mode: str = "so3",
        bias_epsilon: float = 1e-6
    ):
        super().__init__()        

        if act_layer is None:
            act_layer = nn.LeakyReLU(negative_slope=0.2)

        self.norm1 = VecLayerNorm(dim, mode="sim3", eps=bias_epsilon)
        self.norm2 = VecLayerNorm(dim, mode="sim3", eps=bias_epsilon)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values is not None else nn.Identity()
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values is not None else nn.Identity()
        self.scale_factor = scale_factor

        self.mlp = nn.Sequential(
            VecLinear(v_in=dim, v_out=int(dim * mlp_ratio), mode="so3", bias_epsilon=bias_epsilon),
            VecActivation(int(dim * mlp_ratio), act_func=act_layer, mode="so3", bias_epsilon=bias_epsilon),
            VecLinear(v_in=int(dim * mlp_ratio), v_out=dim, mode="so3", bias_epsilon=bias_epsilon),
        )

        self.self_attn, self.local_self_attn, self.self_attn_merge_map = self.build_attention(
            block_style=self_attn_block_style, 
            combine_style=self_attn_combine_style, 
            dim=dim, 
            num_heads=num_heads, 
            k=k,
            bias_epsilon=bias_epsilon
        )

        self.cross_attn, self.local_cross_attn, self.cross_attn_merge_map = self.build_attention(
            block_style=cross_attn_block_style, 
            combine_style=cross_attn_combine_style, 
            dim=dim, 
            num_heads=num_heads, 
            k=k,
            is_cross=True,
            bias_epsilon=bias_epsilon
        )

        self.norm_q = VecLayerNorm(dim, mode="sim3", eps=bias_epsilon)
        self.norm_v = VecLayerNorm(dim, mode="sim3", eps=bias_epsilon)

    def build_attention(self, block_style, combine_style, dim, num_heads, k, is_cross=False, bias_epsilon=1e-6):
        block_tokens = block_style.split('-')
        if not (0 < len(block_tokens) <= 2):
            raise ValueError(f"Invalid block_style: {block_style}. Expected 'attn', 'deform', or both.")

        attn = None
        local_attn = None

        for block_token in block_tokens:
            if block_token == "vnattn":
                attn = VNAttention(dim, num_heads=num_heads, mode = "so3") if not is_cross else VNCrossAttention(dim, dim, num_heads=num_heads, mode = "so3", bias_epsilon=bias_epsilon)
            elif block_token == "vngraph":
                local_attn = VNDynamicGraphAttention(dim, k=k, bias_epsilon= bias_epsilon) 
            else:
                raise ValueError(f"Unexpected block_token: {block_token}. Supported: 'vnattn', 'vngraph'.")

        merge_map = None
        if attn and local_attn and combine_style == "concat":
            merge_map = VecLinear(v_in=dim * 2, v_out=dim, mode="so3", bias_epsilon=bias_epsilon)

        return attn, local_attn, merge_map

    def apply_attention(self, x, pos, idx, attn, local_attn, merge_map, layer_scale , scale_factor = None,mask=None, is_cross=False, v=None, v_pos=None):
        attn_features = []
        norm_x = self.norm1(x) if not is_cross else self.norm_q(x)
        norm_v = self.norm_v(v) if is_cross else None

        if attn:
            attn_features.append(attn(norm_x, mask=mask) if not is_cross else attn(norm_x, norm_v))
        if local_attn:
            attn_features.append(
                local_attn(norm_x, pos, idx=idx) if not is_cross else 
                local_attn(q=norm_x, v=norm_v, q_pos=pos, v_pos=v_pos, idx=idx)
            )

        if len(attn_features) == 1:
            return x + layer_scale(transform_restore(x, attn_features[0], scale_factor))
        elif len(attn_features) == 2 and merge_map:
            merged_features = torch.cat(attn_features, dim=-2)
            merged_features = merge_map(merged_features.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            return x + layer_scale(transform_restore(x, merged_features, scale_factor))
        else:
            raise RuntimeError(f"Unexpected number of attention features: {len(attn_features)}")

    def forward(self, q, v, q_pos, v_pos, self_attn_idx=None, cross_attn_idx=None, denoise_length=None):

        mask = None
        if denoise_length is not None:
            query_len = q.size(1)
            mask = torch.zeros(query_len, query_len).to(q.device)
            mask[:-denoise_length, -denoise_length:] = 1.

        
        q = self.apply_attention(
            x=q, pos=q_pos, idx=self_attn_idx, 
            attn=self.self_attn, local_attn=self.local_self_attn, 
            merge_map=self.self_attn_merge_map, mask=mask,
            layer_scale=self.ls1, scale_factor=self.scale_factor
        ) 

       
        q = self.apply_attention(
            x=q, v=v, pos=q_pos, v_pos=v_pos, idx=cross_attn_idx, 
            attn=self.cross_attn, local_attn=self.local_cross_attn, 
            merge_map=self.cross_attn_merge_map, is_cross=True,
            layer_scale=self.ls2, scale_factor=self.scale_factor
        )

        
        q = q + self.ls2(transform_restore(q, self.mlp(self.norm2(q).permute(0, 2, 3, 1)).permute(0, 3, 1, 2), self.scale_factor))
        return q

######################################## Entry ########################################  

class TransformerEncoder(nn.Module):
    """ Transformer Encoder without hierarchical structure
    """
    def __init__(self, embed_dim=256, depth=4, num_heads=4, mlp_ratio=4., qkv_bias=False, init_values=None,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0., act_layer=nn.GELU(), norm_layer=nn.LayerNorm,
        block_style_list=['vnattn-vngraph'], combine_style='concat', k=10, n_group=2, mode="so3",bias_epsilon=1e-6,scale_factor=None):
        super().__init__()
        self.k = k
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(VecSelfAttnBlockApi(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, init_values=init_values,
                drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
                act_layer=act_layer, norm_layer=norm_layer,
                block_style=block_style_list[i], combine_style=combine_style, k=k, n_group=n_group, mode=mode, bias_epsilon=bias_epsilon, scale_factor=scale_factor
            ))

    def forward(self, x, pos):
        idx = idx = knn_point(self.k, pos, pos)
        for _, block in enumerate(self.blocks):
            x = block(x, pos, idx=idx) 
        return x

class TransformerDecoder(nn.Module):
    """ Transformer Decoder without hierarchical structure
    """
    def __init__(self, embed_dim=256, depth=4, num_heads=4, mlp_ratio=4., qkv_bias=False, init_values=None,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0., act_layer=nn.GELU(), norm_layer=nn.LayerNorm,
        self_attn_block_style_list=['attn-deform'], self_attn_combine_style='concat',
        cross_attn_block_style_list=['attn-deform'], cross_attn_combine_style='concat',
        k=10, n_group=2, mode="so3",bias_epsilon=1e-6, scale_factor=None):
        super().__init__()
        self.k = k
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(VecCrossAttnBlockApi(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, init_values=init_values,
                drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
                act_layer=act_layer, norm_layer=norm_layer,
                self_attn_block_style=self_attn_block_style_list[i], self_attn_combine_style=self_attn_combine_style,
                cross_attn_block_style=cross_attn_block_style_list[i], cross_attn_combine_style=cross_attn_combine_style,
                k=k, n_group=n_group, mode=mode, bias_epsilon=bias_epsilon, scale_factor=scale_factor
            ))

    def forward(self, q, v, q_pos, v_pos, denoise_length=None):
        if denoise_length is None:
            self_attn_idx = knn_point(self.k, q_pos, q_pos)
        else:
            self_attn_idx = None
        cross_attn_idx = knn_point(self.k, v_pos, q_pos)
        for _, block in enumerate(self.blocks):
            q = block(q, v, q_pos, v_pos, self_attn_idx=self_attn_idx, cross_attn_idx=cross_attn_idx, denoise_length=denoise_length)
        return q

class PointTransformerEncoder(nn.Module):

    def __init__(
            self, embed_dim=256, depth=12, num_heads=4, mlp_ratio=4., qkv_bias=True, init_values=None,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
            norm_layer=None, act_layer=None,
            block_style_list=['attn-deform'], combine_style='concat',
            k=10, n_group=2, mode="so3",bias_epsilon=1e-6, scale_factor=None
        ):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.LeakyReLU(negative_slope=0.2, inplace=False)
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        assert len(block_style_list) == depth
        self.blocks = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth = depth,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            init_values=init_values,
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate = dpr,
            norm_layer=norm_layer, 
            act_layer=act_layer,
            block_style_list=block_style_list,
            combine_style=combine_style,
            k=k,
            n_group=n_group,
            mode=mode,
            bias_epsilon=bias_epsilon,
            scale_factor=scale_factor)
        self.norm = norm_layer(embed_dim) 
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, VecLinear):
            trunc_normal_(m.weight, std=.02)

    def forward(self, x, pos):
        x = self.blocks(x, pos)
        return x

class PointTransformerDecoder(nn.Module):
    def __init__(
            self, embed_dim=256, depth=12, num_heads=4, mlp_ratio=4., qkv_bias=True, init_values=None,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
            norm_layer=None, act_layer=None,
            self_attn_block_style_list=['attn-deform'], self_attn_combine_style='concat',
            cross_attn_block_style_list=['attn-deform'], cross_attn_combine_style='concat',
            k=10, n_group=2, mode="so3",bias_epsilon=1e-6, scale_factor=None
        ):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.LeakyReLU(negative_slope=0.2, inplace=False)
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        assert len(self_attn_block_style_list) == len(cross_attn_block_style_list) == depth
        self.blocks = TransformerDecoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth = depth,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            init_values=init_values,
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate = dpr,
            norm_layer=norm_layer, 
            act_layer=act_layer,
            self_attn_block_style_list=self_attn_block_style_list, 
            self_attn_combine_style=self_attn_combine_style,
            cross_attn_block_style_list=cross_attn_block_style_list, 
            cross_attn_combine_style=cross_attn_combine_style,
            k=k, 
            n_group=n_group,
            mode=mode,
            bias_epsilon=bias_epsilon,
            scale_factor=scale_factor
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, VecLinear):
            trunc_normal_(m.weight, std=.02)

    def forward(self, q, v, q_pos, v_pos, denoise_length=None):
        q = self.blocks(q, v, q_pos, v_pos, denoise_length=denoise_length)
        return q

class PointTransformerEncoderEntry(PointTransformerEncoder):
    def __init__(self, config, **kwargs):
        super().__init__(**dict(config))

class PointTransformerDecoderEntry(PointTransformerDecoder):
    def __init__(self, config, **kwargs):
        super().__init__(**dict(config))



class VecDGCNN(nn.Module):
    def __init__(self, k = 16, bias_epsilon=1e-6):

        super().__init__()
        self.k = k
        
        act_func = nn.LeakyReLU(negative_slope=0.2, inplace=False)
        self.conv1 = VecLinearNormalizeActivate(2, 32, mode="sim3", act_func=act_func, bias_epsilon=bias_epsilon)
        self.conv2 = VecLinearNormalizeActivate(64, 64, mode="sim3", act_func=act_func, bias_epsilon=bias_epsilon)
        self.conv3 = VecLinearNormalizeActivate(128, 64, mode="sim3", act_func=act_func, bias_epsilon=bias_epsilon)
        self.conv4 = VecLinearNormalizeActivate(128, 128, mode="sim3", act_func=act_func, bias_epsilon=bias_epsilon)

        self.pool1 = VecMaxPool(32, mode="sim3", bias_epsilon=bias_epsilon)
        self.pool2 = VecMaxPool(64, mode="sim3", bias_epsilon=bias_epsilon)
        self.pool3 = VecMaxPool(64, mode="sim3", bias_epsilon=bias_epsilon)
        self.pool4 = VecMaxPool(128, mode="sim3", bias_epsilon=bias_epsilon)
    
        self.num_features = 128
        
    @staticmethod
    def fps_downsample(coor, x, num_group):

        xyz = coor.transpose(1, 2).contiguous()  # [B, N, 3]
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, num_group)  # [B, num_group]

        B, C, D, N = x.shape  # D = 3
        x_reshaped = x.view(B, C * D, N)  
        combined_x = torch.cat([coor, x_reshaped], dim=1)  # [B, 3 + C*3, N]
        new_combined_x = pointnet2_utils.gather_operation(combined_x, fps_idx)  # [B, 3 + C*3, num_group]
        new_coor = new_combined_x[:, :3, :]  # [B, 3, num_group]
        new_x = new_combined_x[:, 3:, :].view(B, C, D, num_group)  # [B, C, 3, num_group]

        return new_coor, new_x


    def get_graph_feature(self, coor_q, x_q, coor_k, x_k):

        bias = x_k.mean(dim=-1, keepdim=True)
        x_q = x_q  - bias
        x_k = x_k  - bias
        k = self.k
        batch_size = x_k.size(0) # bs, 1, 3, N
        num_points_k = x_k.size(-1)
        num_points_q = x_q.size(-1)
        with torch.no_grad():
            idx = knn_point(k, coor_k.transpose(-1, -2).contiguous(), coor_q.transpose(-1, -2).contiguous()) # B G M
            idx = idx.transpose(-1, -2).contiguous()
            assert idx.shape[1] == k
            idx_base = torch.arange(0, batch_size, device=x_q.device).view(-1, 1, 1) * num_points_k
            idx = idx + idx_base
            idx = idx.view(-1)
        
           
        num_dims = x_k.size(1)
        feature = x_k.permute(0, 3, 1, 2).contiguous().view(batch_size * num_points_k, num_dims, -1)[idx, :, :]
        feature = feature.view(batch_size, k, num_points_q, num_dims, -1).permute(0, 3, 4,2, 1).contiguous()
        x_q = x_q.view(batch_size, num_dims, 3, num_points_q, 1).expand(-1, -1, -1, -1, k)
        feature = torch.cat((feature - x_q, x_q), dim=1)
        
        feature = feature + torch.cat((bias, bias), dim=1).unsqueeze(-1)
        return feature
 

    def forward(self, x, num):
        coor = x.transpose(-1, -2).contiguous()
        x = x.transpose(-1, -2).unsqueeze(1).contiguous() 
   
        f = self.get_graph_feature(coor, x, coor, x)
        f = self.conv1(f)
        f = self.pool1(f)
        
        coor_q, f_q = self.fps_downsample(coor, f, num_group=num[0])
        f = self.get_graph_feature(coor_q, f_q, coor, f)
        
        f = self.conv2(f)
        f = self.pool2(f)
        coor = coor_q

        f= self.get_graph_feature(coor, f, coor, f)
        f = self.conv3(f)
        f = self.pool3(f)

        coor_q, f_q = self.fps_downsample(coor, f, num_group=num[1])
        f= self.get_graph_feature(coor_q, f_q, coor, f)
        f = self.conv4(f)
        f = self.pool4(f)
        
        coor = coor_q.squeeze(1).transpose(2, 1)
        
        return coor, f


class SimpleRebuildFCLayer(nn.Module):
    def __init__(self, input_dims, step, hidden_dim=512, bias_epsilon=1e-6):
        super().__init__()
        self.input_dims = input_dims
        self.step = step
        
        self.pool = VecMaxPool(input_dims//2, mode="sim3", bias_epsilon=bias_epsilon)

        self.layer = nn.Sequential(
            VecLinear(v_in=input_dims, v_out=hidden_dim, mode="so3", bias_epsilon=bias_epsilon),
            VecActivation(
                hidden_dim, act_func=nn.LeakyReLU(negative_slope=0.2), mode="so3", bias_epsilon=bias_epsilon
            ),
            VecLinear(v_in=hidden_dim, v_out=step, mode="so3", bias_epsilon=bias_epsilon),
        )

    def forward(self, rec_feature):
        batch_size = rec_feature.size(0)

        g_feature = self.pool(rec_feature.permute(0,2,3,1))
        token_feature = rec_feature

        patch_feature = torch.cat(
            [
                g_feature.unsqueeze(1).expand(-1, token_feature.size(1), -1, -1),
                token_feature,
            ],
            dim=-2,
        )

        patch_feature = patch_feature - patch_feature.mean(dim=1, keepdim=True)
        rebuild_pc = (
            self.layer(patch_feature.permute(0, 2, 3, 1))
            .permute(0, 3, 1, 2)
            .reshape(batch_size, -1, self.step, 3)
        )
        assert rebuild_pc.size(1) == rec_feature.size(1)

        rebuild_pc = rebuild_pc
        return rebuild_pc



class QueryRanking(nn.Module):
    def __init__(self, mode = "sim3", bias_epsilon=1e-6):
        super().__init__()
        self.fc_inv = VecLinear(1, 256, mode="so3", bias_epsilon=bias_epsilon)
        self.fc_O = VecLinear(1024, 256, mode="sim3", bias_epsilon=bias_epsilon)
        self.fc_bias = VecLinear(1024, 256, mode="sim3", bias_epsilon=bias_epsilon)
        self.query_ranking = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
        
    def forward(self, global_feature, x):

        bias = self.fc_bias(global_feature) 
        z_so3 = self.fc_O(global_feature).unsqueeze(-1) - bias.unsqueeze(-1)
        x = x - x.mean(dim=1, keepdim=True)
        v_inv_per_point = self.fc_inv(x.unsqueeze(-2).permute(0, 2, 3, 1)) 

        
        v_inv_per_point = (v_inv_per_point * z_so3).sum(-2).permute(0, 2, 1) # b, 512, c, 1
        v_inv_per_point = v_inv_per_point / (v_inv_per_point.norm(dim=-1, keepdim=True) + 1e-6)
        x = self.query_ranking(v_inv_per_point)

        idx = torch.argsort(x, dim=1, descending=True) 
        
        return x

class PCTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        encoder_config = config.encoder_config
        decoder_config = config.decoder_config
        self.center_num  = getattr(config, 'center_num', [512, 128])
        self.encoder_type = config.encoder_type
        self.denoise_length = getattr(config, 'denoise_length', 64)
        self.query_selection = getattr(config, 'query_selection', True)
        self.mode = getattr(config, 'mode', "sim3")
        self.bias_epsilon = getattr(config, 'bias_epsilon', 1e-6)
        
        assert self.encoder_type in ['vecgraph'], f'unexpected encoder_type {self.encoder_type}'
        act_func = nn.LeakyReLU(negative_slope=0.2, inplace=False)
        in_chans = 3
        self.num_query = query_num = config.num_query
        global_feature_dim = config.global_feature_dim

        # base encoder
        if self.encoder_type == 'vecgraph':
            self.grouper = VecDGCNN(bias_epsilon=self.bias_epsilon)
        else:
            raise NotImplementedError(f'encoder_type {self.encoder_type} not implemented')
        self.pos_embed = nn.Sequential(
            VecLinear(v_in=1, v_out=128, mode="so3", bias_epsilon=self.bias_epsilon),
            VecActivation(128, act_func=act_func, mode="so3", bias_epsilon=self.bias_epsilon),
            VecLinear(v_in=128, v_out=encoder_config.embed_dim, mode="so3", bias_epsilon=self.bias_epsilon),
        )  
        
        self.input_proj = nn.Sequential(
            VecLinear(v_in=self.grouper.num_features, v_out=512, mode="sim3", bias_epsilon=self.bias_epsilon),
            VecActivation(512, act_func=act_func, mode="sim3", bias_epsilon=self.bias_epsilon),
            VecLinear(v_in=512, v_out=encoder_config.embed_dim, mode="sim3", bias_epsilon=self.bias_epsilon),
        )
        
        
        # Coarse Level 1 : Encoder
        self.encoder = PointTransformerEncoderEntry(encoder_config)
        
        self.increase_dim = nn.Sequential(
            VecLinear(v_in=encoder_config.embed_dim, v_out=1024, mode="sim3", bias_epsilon=self.bias_epsilon),
            VecActivation(1024, act_func=act_func, mode="sim3", bias_epsilon=self.bias_epsilon),
            VecLinear(v_in=1024, v_out=global_feature_dim, mode="sim3", bias_epsilon=self.bias_epsilon),
        )
        self.pool = VecMaxPool(global_feature_dim, mode="sim3", bias_epsilon=self.bias_epsilon)
        
        # query generator
        self.coarse_pred = nn.Sequential(
            VecLinear(v_in=global_feature_dim, v_out=1024, mode="sim3", bias_epsilon=self.bias_epsilon),
            VecActivation(1024, act_func=act_func, mode="sim3", bias_epsilon=self.bias_epsilon),
            VecLinear(v_in=1024, v_out=query_num, mode="sim3", bias_epsilon=self.bias_epsilon),
        )
        
        self.mlp_query = nn.Sequential(
            VecLinear(v_in=global_feature_dim + 1, v_out=1024, mode="sim3", bias_epsilon=self.bias_epsilon),
            VecActivation(1024, act_func=act_func, mode="sim3", bias_epsilon=self.bias_epsilon),
            VecLinear(v_in=1024, v_out=1024, mode="sim3", bias_epsilon=self.bias_epsilon),
            VecActivation(1024, act_func=act_func, mode="sim3", bias_epsilon=self.bias_epsilon),
            VecLinear(
                v_in=1024, v_out=decoder_config.embed_dim, mode="sim3", bias_epsilon=self.bias_epsilon
            )
        )
        
        if decoder_config.embed_dim == encoder_config.embed_dim:
            self.mem_link = nn.Identity()
        else:
            self.mem_link = VecLinear(
                    v_in=encoder_config.embed_dim,
                    v_out=decoder_config.embed_dim,
                    mode="sim3",
                    bias_epsilon=self.bias_epsilon,
                )
    
        # Coarse Level 2 : Decoder
        self.decoder = PointTransformerDecoderEntry(decoder_config)
 
        self.query_ranking = QueryRanking(self.mode, self.bias_epsilon)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, xyz):
        bs = xyz.size(0)
        
        coor, f = self.grouper(xyz, self.center_num) # b n c
        
        x = self.input_proj(f) 
        
        coor_mean = coor.mean(dim=1, keepdim=True)
        pe = self.pos_embed(coor.permute(0, 2, 1).unsqueeze(1).contiguous()-coor_mean.permute(0, 2, 1).unsqueeze(1).contiguous())
        x = (x + pe).permute(0, 3, 1, 2).contiguous() 
        x = self.encoder(x, coor)
        
        
        global_feature = self.increase_dim(x.permute(0, 2, 3, 1).contiguous()) 
        global_feature = self.pool(global_feature)
        coarse = self.coarse_pred(global_feature)
       

        if self.query_selection:
            coarse_inp = fps(xyz.contiguous(), self.num_query // 2)  # B 128 3
            coarse = torch.cat([coarse, coarse_inp], dim=1)  # B 224+128 3
            query_ranking = self.query_ranking(global_feature, coarse)  # b n 1
            idx = torch.argsort(query_ranking, dim=1, descending=True)  # b n 1
            coarse = torch.gather(
                coarse, 1, idx[:, : self.num_query].expand(-1, -1, coarse.size(-1))
            )

        mem = self.mem_link(x.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()

        if self.training:
            if self.denoise_length > 0:
                picked_points = fps(xyz.contiguous(), self.denoise_length)
                picked_points = jitter_points(picked_points)
                coarse = torch.cat([coarse, picked_points], dim=1)  # B 256+64 3?
                denoise_length = self.denoise_length
            else:
                denoise_length = None

            q = self.mlp_query(
                torch.cat(
                    [
                        global_feature.unsqueeze(1).expand(-1, coarse.size(1), -1, -1),
                        coarse.unsqueeze(2),
                    ],
                    dim=-2,
                )
                .permute(0, 2, 3, 1)
                .contiguous()
            )  # b n c
            q = q.permute(0, 3, 1, 2).contiguous()
            
            q = self.decoder(
                q=q, v=mem, q_pos=coarse, v_pos=coor, denoise_length=denoise_length
            )
            
            return q, coarse, self.denoise_length
        else:
            # produce query
            q = self.mlp_query(
                torch.cat(
                    [
                        global_feature.unsqueeze(1).expand(-1, coarse.size(1), -1, -1),
                        coarse.unsqueeze(2),
                    ],
                    dim=-2,
                )
                .permute(0, 2, 3, 1)
                .contiguous()
            )  # b n c
            q = q.permute(0, 3, 1, 2).contiguous()
            

            q = self.decoder(
                q=q, v=mem, q_pos=coarse, v_pos=coor
            )
            return q, coarse, 0

@MODELS.register_module()
class SIMECO(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.trans_dim = config.decoder_config.embed_dim
        self.num_query = config.num_query
        self.num_points = getattr(config, 'num_points', None)
        self.bias_epsilon = getattr(config, 'bias_epsilon', 1e-6)

        self.decoder_type = config.decoder_type
        assert self.decoder_type in ['fold', 'fc'], f'unexpected decoder_type {self.decoder_type}'
        self.encoder_type = config.encoder_type
        self.fold_step = 8
        self.base_model = PCTransformer(config)
        self.denoise_length = getattr(config, 'denoise_length', 64)
        

        if self.num_points is not None:
            self.factor = self.num_points // self.num_query
            assert self.num_points % self.num_query == 0
            self.decode_head = SimpleRebuildFCLayer(self.trans_dim * 2, step=self.num_points // self.num_query, bias_epsilon=self.bias_epsilon)
        else:
            self.factor = self.fold_step**2
            self.decode_head = SimpleRebuildFCLayer(self.trans_dim * 2, step=self.fold_step**2, bias_epsilon=self.bias_epsilon)
        
        self.increase_dim = nn.Sequential(
            VecLinear(self.trans_dim, 1024, mode="sim3", bias_epsilon=self.bias_epsilon),
            VecActivation(1024, act_func=nn.LeakyReLU(negative_slope=0.2), mode="sim3", bias_epsilon=self.bias_epsilon),
            VecLinear(1024, 1024, mode="sim3", bias_epsilon=self.bias_epsilon),
        )
        self.pool = VecMaxPool(1024, mode="sim3", bias_epsilon=self.bias_epsilon)
        
        self.reduce_map = nn.Sequential(
            VecLinear(self.trans_dim + 1025, self.trans_dim, mode="sim3", bias_epsilon=self.bias_epsilon)) 

    def get_loss(self, ret, gt, epoch=1):
        from extensions.chamfer_dist import ChamferDistanceL1
        self.loss_func = ChamferDistanceL1()
        pred_coarse, denoised_coarse, denoised_fine, pred_fine = ret
        
        assert pred_fine.size(1) == gt.size(1)

        # denoise loss
        idx = knn_point(self.factor, gt, denoised_coarse) # B n k 
        denoised_target = index_points(gt, idx) # B n k 3 
        denoised_target = denoised_target.reshape(gt.size(0), -1, 3)
        assert denoised_target.size(1) == denoised_fine.size(1)
        loss_denoised = self.loss_func(denoised_fine, denoised_target)
        loss_denoised = loss_denoised * 0.5

        # recon loss
        loss_coarse = self.loss_func(pred_coarse, gt)
        loss_fine = self.loss_func(pred_fine, gt)
        loss_recon = loss_coarse + loss_fine

        return loss_denoised, loss_recon
    
    def forward(self, xyz):
        
        q, coarse_point_cloud, denoise_length = self.base_model(xyz) 
            
        B, M, C, _ = q.shape
        
        global_feature = self.increase_dim(q.permute(0, 2, 3, 1).contiguous())
        
        global_feature = self.pool(global_feature)
        
        rebuild_feature = torch.cat(
            [
                global_feature.unsqueeze(-3).expand(-1, M, -1, -1),
                q,
                coarse_point_cloud.unsqueeze(-2),
            ],
            dim=-2,
        ) 
            
        rebuild_feature = self.reduce_map(rebuild_feature.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
        
        
        relative_xyz = self.decode_head(rebuild_feature)
        rebuild_points = relative_xyz + coarse_point_cloud.unsqueeze(-2)  # B M S 3

        if self.training:
            pred_fine = rebuild_points[:, :-denoise_length].reshape(B, -1, 3).contiguous()
            pred_coarse = coarse_point_cloud[:, :-denoise_length].contiguous()

            denoised_fine = rebuild_points[:, -denoise_length:].reshape(B, -1, 3).contiguous()
            denoised_coarse = coarse_point_cloud[:, -denoise_length:].contiguous()
            assert pred_fine.size(1) == self.num_query * self.factor
            assert pred_coarse.size(1) == self.num_query

            ret = (pred_coarse, denoised_coarse, denoised_fine, pred_fine)
            return ret
        else:
            assert denoise_length == 0
            rebuild_points = rebuild_points.reshape(B, -1, 3).contiguous()  # B N 3

            assert rebuild_points.size(1) == self.num_query * self.factor
            assert coarse_point_cloud.size(1) == self.num_query
            
            ret = (coarse_point_cloud, rebuild_points)
            
            return ret
