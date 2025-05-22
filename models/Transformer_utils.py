import torch
import torch.nn as nn
import einops
from models.vec_layers import *

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm;
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist   

def index_points(points, idx): 
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    # import pdb; pdb.set_trace()
    # if not vec:
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

 
class VNAttention(nn.Module):
    def __init__(self, dim, num_heads=8, flash=False, l2_dist=False, mode='so3', bias_epsilon=1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.to_q = VecLinear(dim, dim, mode="so3", bias_epsilon=bias_epsilon)
        self.to_k = VecLinear(dim, dim, mode="so3", bias_epsilon=bias_epsilon)
        self.to_v = VecLinear(dim, dim, mode="so3", bias_epsilon=bias_epsilon)
        self.attend = Attend(flash = flash, l2_dist = l2_dist, mode=mode)
        self.proj = VecLinear(dim, dim, mode="so3", bias_epsilon=bias_epsilon)
        self.mode = mode
        
    def forward(self, x, mask=None):
        x = x.permute(0, 2, 3, 1)
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x) # b, c, 3, n
        q, k, v = map(lambda t:  einops.rearrange(t, 'b (h d) c n-> b h n (d c)', h = self.num_heads), (q, k, v))
       
        if mask is not None:
            mask = mask[0].unsqueeze(0).expand(x.size(0), -1)
        x = self.attend(q, k, v, mask = mask)
        x = einops.rearrange(x, 'b h n (d c) -> b (h d) c n', c = 3)
        x = self.proj(x)
        x = x.permute(0, 3, 1, 2)
        return x

class VNCrossAttention(nn.Module):
    def __init__(self, dim, out_dim, num_heads=8, flash=False, l2_dist_attn=False, mode='so3', bias_epsilon=1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.out_dim = out_dim
        self.mode = mode
        self.q_map = VecLinear(v_in=dim, v_out=out_dim, mode="so3", bias_epsilon=bias_epsilon)
        self.k_map = VecLinear(v_in=dim, v_out=out_dim, mode="so3", bias_epsilon=bias_epsilon)
        self.v_map = VecLinear(v_in=dim, v_out=out_dim, mode="so3", bias_epsilon=bias_epsilon)
        self.proj = VecLinear(v_in=out_dim, v_out=out_dim, mode="so3", bias_epsilon=bias_epsilon)
        self.attend = Attend(flash = flash, l2_dist = l2_dist_attn, mode=mode)

    def forward(self, q, v):
        q = q.permute(0, 2, 3, 1)
        v = v.permute(0, 2, 3, 1)

        q, k, v = self.q_map(q), self.k_map(v), self.v_map(v)
        q, k, v = map(lambda t: einops.rearrange(t, 'b (h d) c n-> b h n (d c)', h = self.num_heads), (q, k, v))
        
        x = self.attend(q, k, v) 
        x = einops.rearrange(x, 'b h n (d c) -> b (h d) c n', c = 3)
        x = self.proj(x)
        x = x.permute(0, 3, 1, 2)
        return x

class VNDynamicGraphAttention(nn.Module):
    def __init__(
        self,
        dim,
        k = 10,
        bias_epsilon = 1e-6,
        ):
        super().__init__()
        
        self.k = k  # To be controlled
        self.dim = dim 
        self.knn_map = nn.Sequential(VecLinear(dim * 2, dim, mode="so3", bias_epsilon=bias_epsilon),
                                     VecActivation(dim, act_func= nn.LeakyReLU(negative_slope=0.2, inplace=False), mode='so3', bias_epsilon=bias_epsilon))
        self.pool = VecMaxPool(dim, bias_epsilon=bias_epsilon)


    def forward(self, q, q_pos, v=None, v_pos=None, idx=None, denoise_length=None):
   
        if denoise_length is None:
            if v is None:
                v = q
            if v_pos is None:
                v_pos = q_pos
                
            assert len(v_pos.shape) == 3 and v_pos.size(-1) == 3, f'[ERROR] Got an unexpected shape for v_pos, expect it to be B N 3, but got {v_pos.shape}'
            assert len(q_pos.shape) == 3 and q_pos.size(-1) == 3, f'[ERROR] Got an unexpected shape for q_pos, expect it to be B N 3, but got {q_pos.shape}'
            assert q.size(-2) == v.size(-2) == self.dim
            
            B, N, C, _ = q.shape
            
            # first query a neighborhood for one query token
            if idx is None:
                idx = knn_point(self.k, v_pos, q_pos) # B N k 
            assert idx.size(-1) == self.k
            
            q = q.unsqueeze(2).expand(-1, -1, self.k, -1, -1)
            local_v = index_points(v, idx)    
             # B N k C, 3
            feature = torch.cat((local_v - q, q), dim=-2) # B N k C, 3
            feature = feature.permute(0, 3, 4, 1, 2)
            out = self.knn_map(feature)
            out = self.pool(out).permute(0, 3, 1, 2)
            

            assert out.size(0) == B
            assert out.size(1) == N
            assert out.size(2) == C
        else:
            assert idx is None, f'we need online index calculation when denoise_length is set, denoise_length {denoise_length}'
            assert v is None, f'mask for denoise_length is only consider in self-attention, but v is given'
            assert v_pos is None, f'mask for denoise_length is only consider in self-attention, but v_pos is given'

            v = q
            v_pos = q_pos
            # given N token and pos
            assert len(v_pos.shape) == 3 and v_pos.size(-1) == 3, f'[ERROR] Got an unexpected shape for v_pos, expect it to be B N 3, but got {v_pos.shape}'
            assert len(q_pos.shape) == 3 and q_pos.size(-1) == 3, f'[ERROR] Got an unexpected shape for q_pos, expect it to be B N 3, but got {q_pos.shape}'
            assert q.size(-2) == v.size(-2) == self.dim
            B, N, C, _ = q.shape
            
            idx = knn_point(self.k, v_pos[:, :-denoise_length], q_pos[:, :-denoise_length]) # B N_r k 
            assert idx.size(-1) == self.k
            local_v_r = index_points(v[:, :-denoise_length], idx)     # B N_r k C, 3 
 
            idx = knn_point(self.k, v_pos, q_pos[:, -denoise_length:]) # B N_n k 
            assert idx.size(-1) == self.k
            assert idx.size(1) == denoise_length
            
            local_v_n = index_points(v, idx)    # B N_n k C, 3
            local_v = torch.cat([local_v_r, local_v_n], dim=1)
            
            q = q.unsqueeze(2).expand(-1, -1, self.k, -1, -1)
            feature = torch.cat((local_v - q, q), dim=-2) # B N k C, 3
            feature = feature.permute(0, 3, 4, 1, 2)
            out = self.knn_map(feature)
            out = self.pool(out).permute(0, 3, 1, 2)

            assert out.size(0) == B
            assert out.size(1) == N
            assert out.size(2) == C
        return out