import torch
import torch.nn as nn
import einops

from .vec_layers import *


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
    """
    Vector Neuron Attention module that performs self-attention on 3D point cloud features
    while maintaining equivariance to 3D rotations.
    """
    def __init__(self, dim, num_heads=8, flash=False, l2_dist=False, mode='so3', bias_epsilon=1e-6):
        super().__init__()
        self.num_heads = num_heads
        # Linear transformations for query, key, and value projections
        # Using VecLinear to maintain SO(3) equivariance
        self.to_q = VecLinear(dim, dim, mode="so3", bias_epsilon=bias_epsilon)
        self.to_k = VecLinear(dim, dim, mode="so3", bias_epsilon=bias_epsilon)
        self.to_v = VecLinear(dim, dim, mode="so3", bias_epsilon=bias_epsilon)
        
        # Attention mechanism with optional flash attention and L2 distance
        self.attend = Attend(flash = flash, l2_dist = l2_dist)
        
        # Output projection layer
        self.proj = VecLinear(dim, dim, mode="so3", bias_epsilon=bias_epsilon)
        self.mode = mode
        
    def forward(self, x, mask=None):
        """
        Forward pass of vector neuron attention
        
        Args:
            x: Input tensor of shape [B, N, C, 3] where:
               - B: batch size
               - N: number of points
               - C: feature channels
               - 3: spatial dimensions (x, y, z)
            mask: Optional attention mask
            
        Returns:
            Output tensor of same shape as input [B, N, C, 3]
            
        Einstein notation used in comments:
        b - batch
        n - points
        h - heads
        d - feature dimension (channels)
        c - coordinate dimension (3 for 3d space)
        """
       
        # Permute to [B, C, 3, N] for VecLinear operations
        x = x.permute(0, 2, 3, 1)
        
        # Compute query, key, and value projections
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)  # [B, C, 3, N]
        
        # Reshape for multi-head attention: [B, H, N, (D*C)] where H=num_heads, D=C/H
        q, k, v = map(lambda t:  einops.rearrange(t, 'b (h d) c n-> b h n (d c)', h = self.num_heads), (q, k, v))
       
        # Handle optional attention mask
        if mask is not None:
            # Expand mask to match batch size
            mask = mask[0].unsqueeze(0).expand(x.size(0), -1)
            
        # Apply attention mechanism
        x = self.attend(q, k, v, mask = mask)
        
        # Reshape back to [B, C, 3, N] format
        x = einops.rearrange(x, 'b h n (d c) -> b (h d) c n', c = 3)
        
        # Apply output projection
        x = self.proj(x)
        
        # Permute back to original format [B, N, C, 3]
        x = x.permute(0, 3, 1, 2)
        return x


class VNCrossAttention(nn.Module):
    """
    Vector Neuron Cross Attention module that performs cross-attention between two sets 
    of 3D point cloud features while maintaining equivariance to 3D rotations.
    
    Unlike self-attention, cross-attention uses one set of features (q) to query 
    another set of features (v) for information aggregation.
    """
    def __init__(self, dim, out_dim, num_heads=8, flash=False, l2_dist_attn=False, mode='so3', bias_epsilon=1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.out_dim = out_dim
        self.mode = mode

        # Linear transformations for query, key, and value projections
        # Query comes from first input, key and value come from second input
        self.q_map = VecLinear(v_in=dim, v_out=out_dim, mode="so3", bias_epsilon=bias_epsilon)
        self.k_map = VecLinear(v_in=dim, v_out=out_dim, mode="so3", bias_epsilon=bias_epsilon)
        self.v_map = VecLinear(v_in=dim, v_out=out_dim, mode="so3", bias_epsilon=bias_epsilon)
        
        # Output projection layer
        self.proj = VecLinear(v_in=out_dim, v_out=out_dim, mode="so3", bias_epsilon=bias_epsilon)
        
        # Attention mechanism with optional flash attention and L2 distance
        self.attend = Attend(flash = flash, l2_dist = l2_dist_attn)

    def forward(self, q, v):
        """
        Forward pass of vector neuron cross attention
        
        Args:
            q: Query tensor of shape [B, N_q, C, 3] - features to be updated
            v: Key/Value tensor of shape [B, N_v, C, 3] - features to attend to
            
        Returns:
            Output tensor of shape [B, N_q, out_dim, 3]
            
        Einstein notation:
        b - batch
        n - points 
        h - heads
        d - feature dimension
        c - coordinate dimension (3)
        """
        # Permute to [B, C, 3, N] for VecLinear operations
        q = q.permute(0, 2, 3, 1)  # [B, C, 3, N_q]
        v = v.permute(0, 2, 3, 1)  # [B, C, 3, N_v]

        # Compute query from first input, key and value from second input
        q, k, v = self.q_map(q), self.k_map(v), self.v_map(v)  # All [B, out_dim, 3, N]
        
        # Reshape for multi-head attention: [B, H, N, (D*C)]
        q, k, v = map(lambda t: einops.rearrange(t, 'b (h d) c n-> b h n (d c)', h = self.num_heads), (q, k, v))
        
        # Apply cross-attention mechanism
        x = self.attend(q, k, v)  # [B, H, N_q, (D*C)]
        
        # Reshape back to [B, out_dim, 3, N_q] format
        x = einops.rearrange(x, 'b h n (d c) -> b (h d) c n', c = 3)
        
        # Apply output projection
        x = self.proj(x)
        
        # Permute back to original format [B, N_q, out_dim, 3]
        x = x.permute(0, 3, 1, 2)
        return x

class VNDynamicGraphAttention(nn.Module):
    """
    Vector Neuron Dynamic Graph Attention module that performs attention on 3D point clouds
    using k-nearest neighbor graphs. This module maintains SO(3) equivariance while allowing
    dynamic neighborhood construction based on spatial proximity.
    
    The key innovation is using KNN to dynamically construct local neighborhoods and then
    applying vector neuron operations to aggregate information from these neighborhoods.
    """
    def __init__(
        self,
        dim,
        k = 10,
        bias_epsilon = 1e-6,
        ):
        super().__init__()
        
        self.k = k  # Number of nearest neighbors to consider
        self.dim = dim  # Feature dimension
        
        # Neural network for processing concatenated neighbor features
        # Takes input of size dim*2 (neighbor + query features) and outputs dim features
        self.knn_map = nn.Sequential(
            VecLinear(dim * 2, dim, mode="so3", bias_epsilon=bias_epsilon),
            VecActivation(dim, act_func=nn.LeakyReLU(negative_slope=0.2, inplace=False), 
                         mode='so3', bias_epsilon=bias_epsilon)
        )
        
        # Max pooling operation to aggregate information from k neighbors
        self.pool = VecMaxPool(dim, bias_epsilon=bias_epsilon)

    def forward(self, q, q_pos, v=None, v_pos=None, idx=None, denoise_length=None):
        """
        Forward pass of dynamic graph attention
        
        Args:
            q: Query features [B, N, C, 3] - features to be updated
            q_pos: Query positions [B, N, 3] - 3D positions for query points
            v: Value features [B, M, C, 3] - features to attend to (optional, defaults to q)
            v_pos: Value positions [B, M, 3] - 3D positions for value points (optional, defaults to q_pos)
            idx: Precomputed neighbor indices [B, N, k] (optional, computed if None)
            denoise_length: Number of points at end that need denoising (optional)
            
        Returns:
            out: Updated features [B, N, C, 3] after graph attention
        """
   
        if denoise_length is None:
            # Standard mode: process all points uniformly
            
            # Use self-attention if v and v_pos are not provided
            if v is None:
                v = q
            if v_pos is None:
                v_pos = q_pos
                
            # Validate input shapes
            assert len(v_pos.shape) == 3 and v_pos.size(-1) == 3, f'[ERROR] Got an unexpected shape for v_pos, expect it to be B N 3, but got {v_pos.shape}'
            assert len(q_pos.shape) == 3 and q_pos.size(-1) == 3, f'[ERROR] Got an unexpected shape for q_pos, expect it to be B N 3, but got {q_pos.shape}'
            assert q.size(-2) == v.size(-2) == self.dim
            
            B, N, C, _ = q.shape
            
            # Find k nearest neighbors for each query point
            if idx is None:
                idx = knn_point(self.k, v_pos, q_pos)  # [B, N, k]
            assert idx.size(-1) == self.k
            
            # Expand query features to match neighborhood size
            q = q.unsqueeze(2).expand(-1, -1, self.k, -1, -1)  # [B, N, k, C, 3]
            
            # Get neighbor features using computed indices
            local_v = index_points(v, idx)  # [B, N, k, C, 3]
            
            # Concatenate relative position features (local_v - q) with query features
            # This creates edge features that capture both local geometry and query context
            feature = torch.cat((local_v - q, q), dim=-2)  # [B, N, k, 2*C, 3]
            
            # Permute for processing: [B, 2*C, 3, N, k]
            feature = feature.permute(0, 3, 4, 1, 2)
            
            # Apply neural network to process neighbor features
            out = self.knn_map(feature)  # [B, C, 3, N, k]
            
            # Pool across neighbors and permute back to original format
            out = self.pool(out).permute(0, 3, 1, 2)  # [B, N, C, 3]

            # Validate output shapes
            assert out.size(0) == B
            assert out.size(1) == N
            assert out.size(2) == C
            
        else:
            # Denoising mode: handle reconstruction and denoising tasks differently
            
            # Ensure we're in self-attention mode for denoising
            assert idx is None, f'we need online index calculation when denoise_length is set, denoise_length {denoise_length}'
            assert v is None, f'mask for denoise_length is only consider in self-attention, but v is given'
            assert v_pos is None, f'mask for denoise_length is only consider in self-attention, but v_pos is given'

            # Set up self-attention
            v = q
            v_pos = q_pos
            
            # Validate input shapes
            assert len(v_pos.shape) == 3 and v_pos.size(-1) == 3, f'[ERROR] Got an unexpected shape for v_pos, expect it to be B N 3, but got {v_pos.shape}'
            assert len(q_pos.shape) == 3 and q_pos.size(-1) == 3, f'[ERROR] Got an unexpected shape for q_pos, expect it to be B N 3, but got {q_pos.shape}'
            assert q.size(-2) == v.size(-2) == self.dim
            B, N, C, _ = q.shape

            # For reconstruction task: find neighbors within clean points only
            # This prevents clean points from being influenced by noisy points
            idx = knn_point(self.k, v_pos[:, :-denoise_length], q_pos[:, :-denoise_length])  # [B, N_clean, k]
            assert idx.size(-1) == self.k
            local_v_r = index_points(v[:, :-denoise_length], idx)  # [B, N_clean, k, C, 3]
 
            # For denoising task: find neighbors from all points (clean + noisy)
            # This allows noisy points to attend to both clean and other noisy points
            idx = knn_point(self.k, v_pos, q_pos[:, -denoise_length:])  # [B, N_noisy, k]
            assert idx.size(-1) == self.k
            assert idx.size(1) == denoise_length
            
            local_v_n = index_points(v, idx)  # [B, N_noisy, k, C, 3]

            # Concatenate reconstruction and denoising neighborhoods
            local_v = torch.cat([local_v_r, local_v_n], dim=1)  # [B, N, k, C, 3]
            
            # Expand query features to match neighborhood size
            q = q.unsqueeze(2).expand(-1, -1, self.k, -1, -1)  # [B, N, k, C, 3]
            
            # Create edge features by concatenating relative positions with query features
            feature = torch.cat((local_v - q, q), dim=-2)  # [B, N, k, 2*C, 3]
            
            # Permute and process through neural network
            feature = feature.permute(0, 3, 4, 1, 2)  # [B, 2*C, 3, N, k]
            out = self.knn_map(feature)  # [B, C, 3, N, k]
            
            # Pool and reshape to final output format
            out = self.pool(out).permute(0, 3, 1, 2)  # [B, N, C, 3]

            # Validate output shapes
            assert out.size(0) == B
            assert out.size(1) == N
            assert out.size(2) == C
            
        return out


class LayerScale(nn.Module):
    """Layer scaling module."""
    
    def __init__(self, dim: int, init_values: float = 1e-5, inplace: bool = False):
        """Initialize layer scaling module.
        
        Args:
            dim: Input dimension
            init_values: Initial scaling values
            inplace: Whether to perform in-place scaling
        """
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor, shape [B, N, C]
            
        Returns:
            Scaled tensor, shape [B, N, C]
        """
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

