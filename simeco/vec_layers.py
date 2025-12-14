import torch
from torch import nn, einsum
import torch.nn.functional as F

import math
import logging
from functools import wraps
from collections import namedtuple
from einops import rearrange, reduce
from packaging import version

# Configuration for Flash Attention variants
FlashAttentionConfig = namedtuple('FlashAttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])

# Utility function to check if a value is not None
def exists(val):
    return val is not None

# Decorator to ensure a function is only called once
def once(fn):
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

# Print function that only executes once
print_once = once(print)

def channel_equi_vec_normalize(x):
    """
    Channel-wise equivariant vector normalization.
    
    Normalizes vectors to unit length while maintaining relative magnitudes
    across channels through channel-wise normalization of the norms.
    
    Args:
        x (torch.Tensor): Input tensor of shape [B, C, 3, ...] where:
            - B: batch size
            - C: number of channels
            - 3: vector dimension (x, y, z components)
            - ...: optional additional dimensions
            
    Returns:
        torch.Tensor: Normalized vectors with same shape as input
    """
    assert x.ndim >= 3, "x shape [B,C,3,...]"
    
    # Normalize each vector to unit length (preserves direction)
    x_dir = F.normalize(x, dim=2)  # Shape: [B, C, 3, ...]
    
    # Get vector magnitudes and normalize across channels
    x_norm = x.norm(dim=2, keepdim=True)  # Shape: [B, C, 1, ...]
    x_normalized_norm = F.normalize(x_norm, dim=1)  # Normalize across channel dimension
    
    # Reconstruct vectors with normalized magnitudes
    y = x_dir * x_normalized_norm
    return y

def transform_restore(x, y, scale_factor=None):
    """
    Restore scale information from reference tensor x to target tensor y.
    
    Args:
        x (torch.Tensor): Reference tensor of shape [B, N, C, 3, ...]
        y (torch.Tensor): Target tensor to scale, same shape as x
        scale_factor (float, optional): Additional scaling factor. Default: None.
        
    Returns:
        torch.Tensor: Scaled tensor y with restored scale information
    """
    if scale_factor is not None:
        assert x.ndim == y.ndim, "x and y must have same number of dimensions"
        
        # Compute scale from reference tensor x
        # Center vectors, then compute mean norm across channels and spatial dimensions
        x_centered = x - x.mean(dim=2, keepdim=True)  # Center around channel mean
        scale = x_centered.mean(dim=1, keepdim=False).norm(dim=-1).mean(dim=-1, keepdim=False)  # [B]
        
        # Apply scale restoration to y
        y = y * scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * scale_factor
    
    return y


class VecLinear(nn.Module):
    """
    Vector Linear Layer for SO(3), SE(3), and Sim(3) equivariant neural networks.
    
    Implements a linear transformation that maintains rotational, translational, 
    and/or scale equivariance for 3D vector and scalar features.
    
    Args:
        v_in (int): Number of input vector channels.
        v_out (int): Number of output vector channels.
        s_in (int, optional): Number of input scalar channels. Default: 0.
        s_out (int, optional): Number of output scalar channels. Default: 0.
        s2v_normalized_scale (bool, optional): Whether to normalize scalar-to-vector scaling. Default: True.
        mode (str, optional): Equivariance mode, either "so3", "se3", or "sim3". Default: "sim3".
        device (torch.device, optional): Device for tensors. Default: None.
        dtype (torch.dtype, optional): Data type for tensors. Default: None.
        vs_dir_learnable (bool, optional): Whether vector-to-scalar direction is learnable. Default: True.
        cross (bool, optional): Whether to use cross product operations. Default: False.
        bias_epsilon (float, optional): Small epsilon value for bias normalization. Default: 1e-6.
    """

    def __init__(
        self,
        v_in: int,
        v_out: int,
        s_in=0,
        s_out=0,
        s2v_normalized_scale=True,
        mode="sim3",
        device=None,
        dtype=None,
        vs_dir_learnable=True,
        cross=False,
        bias_epsilon=1e-6,
    ) -> None:
        mode = mode.lower()
        assert mode in ["so3", "se3", "sim3"], "mode must be so3, se3 or sim3"
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.v_in = v_in
        self.v_out = v_out
        self.s_in = s_in
        self.s_out = s_out
        self.mode = mode

        assert self.s_out + self.v_out > 0, "vec, scalar output both zero"

        # SE(3) and Sim(3) both require centering operations
        self.se3_flag = (mode == "se3" or mode == "sim3")
        if self.se3_flag:
            assert v_in > 1, "se3/sim3 layers must have at least two input layers"

        # Main vector weight
        if self.v_out > 0:
            self.weight = nn.Parameter(
                torch.empty((v_out, v_in - 1 if self.se3_flag else v_in), **factory_kwargs)
            )
            self.reset_parameters()

        # Scalar-to-vector fusion
        if self.s_in > 0 and self.v_out > 0:
            self.sv_linear = nn.Linear(s_in, v_out)
            self.s2v_normalized_scale_flag = s2v_normalized_scale

        # Vector-to-scalar projection
        if self.s_out > 0:
            self.vs_dir_learnable = vs_dir_learnable
            assert self.vs_dir_learnable, "non-learnable direction is numerically unstable"
            self.vs_dir_linear = VecLinear(v_in, v_in, mode="so3")
            self.vs_linear = nn.Linear(v_in, s_out)

        # Scalar-to-scalar transformation
        if self.s_in > 0 and self.s_out > 0:
            self.ss_linear = nn.Linear(s_in, s_out)

        # Cross product enhancement
        self.cross_flag = cross
        if self.v_out > 0 and self.cross_flag:
            self.v_out_cross = VecLinear(v_in, v_out, mode=mode, cross=False)
            self.v_out_cross_fc = VecLinear(v_out * 2, v_out, mode=mode, cross=False)

        # Bias parameters
        self.bias = nn.Parameter(torch.randn(v_out))
        self.bias_epsilon = bias_epsilon

    @torch.no_grad()
    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.se3_flag:
            self.weight.data += 1.0 / self.v_in

    def forward(self, v_input: torch.Tensor, s_input=None):
        """
        Forward pass for vector linear transformation.
        
        Args:
            v_input (torch.Tensor): Input vector features of shape (B, C, 3, ...)
            s_input (torch.Tensor, optional): Input scalar features of shape (B, C, ...)
            
        Returns:
            torch.Tensor or tuple: Vector output, or (vector_output, scalar_output) if s_out > 0
        """
        v_shape = v_input.shape
        assert v_shape[2] == 3, "not vector neuron"

        # Vector transformation
        if self.v_out > 0:
            if self.se3_flag:
                W = torch.cat([self.weight, 1.0 - self.weight.sum(-1, keepdim=True)], -1).contiguous()
            else:
                W = self.weight
            v_output = F.linear(v_input.transpose(1, -1), W).transpose(-1, 1)
        else:
            v_output = None

        # Scalar-to-vector fusion
        if self.s_in > 0:
            assert s_input is not None, "missing scalar input"
            if self.v_out > 0:
                s2v_invariant_scale = self.sv_linear(s_input.transpose(1, -1)).transpose(-1, 1)
                if self.s2v_normalized_scale_flag:
                    s2v_invariant_scale = F.normalize(s2v_invariant_scale, dim=1)
                
                if self.se3_flag:
                    v_new_mean = v_output.mean(dim=1, keepdim=True)
                    v_output = (v_output - v_new_mean) * s2v_invariant_scale.unsqueeze(2) + v_new_mean
                else:
                    v_output = v_output * s2v_invariant_scale.unsqueeze(2)

        # Cross product enhancement
        if self.v_out > 0 and self.cross_flag:
            v_out_dual = self.v_out_cross(v_input)
            if self.se3_flag:
                v_out_dual_o = v_out_dual.mean(dim=1, keepdim=True)
                v_output_o = v_output.mean(dim=1, keepdim=True)
                v_cross = torch.cross(
                    channel_equi_vec_normalize(v_out_dual - v_out_dual_o),
                    v_output - v_output_o,
                    dim=2,
                )
            else:
                v_cross = torch.cross(channel_equi_vec_normalize(v_out_dual), v_output, dim=2)
            v_cross = v_cross + v_output
            v_output = self.v_out_cross_fc(torch.cat([v_cross, v_output], dim=1))

        # Vector-to-scalar projection
        if self.s_out > 0:
            v_sR = v_input - v_input.mean(dim=1, keepdim=True) if self.se3_flag else v_input
            v_sR_dual_dir = F.normalize(self.vs_dir_linear(v_sR), dim=2)
            s_from_v = F.normalize((v_sR * v_sR_dual_dir).sum(dim=2), dim=1)
            s_from_v = self.vs_linear(s_from_v.transpose(-1, 1)).transpose(-1, 1)
            
            if self.s_in > 0:
                s_from_s = self.ss_linear(s_input.transpose(-1, 1)).transpose(-1, 1)
                s_output = s_from_s + s_from_v
            else:
                s_output = s_from_v
            return v_output, s_output
        else:
            # Add normalized bias
            bias = F.normalize(self.bias, dim=-1) * self.bias_epsilon
            bias = bias.view(1, -1, *([1] * (v_output.ndim - 2)))
            v_output = v_output + bias
            return v_output


class VecActivation(nn.Module):
    """
    Vector Activation Layer for SO(3), SE(3), and Sim(3) equivariant neural networks.
    
    Applies non-linear activation functions while preserving equivariance properties.
    The activation is applied only to the component of the input vector that is parallel
    to a learned direction, while the orthogonal component remains unchanged.
    
    Args:
        in_features (int): Number of input vector channels.
        act_func: Activation function to apply (e.g., nn.ReLU()).
        shared_nonlinearity (bool, optional): Whether to share nonlinearity across channels. Default: False.
        mode (str, optional): Equivariance mode, either "so3", "se3", or "sim3". Default: "sim3".
        normalization (nn.Module, optional): Normalization layer (breaks scale equivariance). Default: None.
        cross (bool, optional): Whether to use cross product operations. Default: False.
        bias_epsilon (float, optional): Small epsilon value for bias normalization. Default: 1e-6.
    """
    
    def __init__(
        self,
        in_features,
        act_func,
        shared_nonlinearity=False,        
        mode="sim3",
        normalization=None,
        cross=False,
        bias_epsilon=1e-6,
    ) -> None:
        super().__init__()

        mode = mode.lower()
        assert mode in ["so3", "se3", "sim3"], "mode must be so3, se3 or sim3"
        self.se3_flag = mode == "se3" or mode == "sim3"
        self.shared_nonlinearity_flag = shared_nonlinearity
        self.act_func = act_func

        # Determine output channels for direction prediction
        nonlinear_out = 1 if self.shared_nonlinearity_flag else in_features
        
        # Linear layer to predict activation direction
        self.lin_dir = VecLinear(in_features, nonlinear_out, mode=mode, cross=cross, bias_epsilon=bias_epsilon)
        
        # For SE(3)/Sim(3): additional layer to predict translation origin
        if self.se3_flag:
            self.lin_ori = VecLinear(in_features, nonlinear_out, mode=mode, cross=cross, bias_epsilon=bias_epsilon)
        
        # Optional normalization (breaks scale equivariance)
        self.normalization = normalization
        if self.normalization is not None:
            logging.warning("Warning! Set Batchnorm True, not Scale Equivariant")

    def forward(self, x):
        """
        Forward pass for vector activation.
        
        Args:
            x (torch.Tensor): Input vector features of shape (B, C, 3, ...)
            
        Returns:
            torch.Tensor: Activated vector features with same shape as input
        """
        assert x.shape[2] == 3, "not vector neuron"
        
        # Get input vectors and activation direction
        q = x
        k = self.lin_dir(x)  # Learned activation direction
        
        # For SE(3)/Sim(3): center the vectors
        if self.se3_flag:
            o = self.lin_ori(x)  # Learned translation origin
            q = q - o  # Center input vectors
            k = k - o  # Center direction vectors

        # Apply normalization if specified (breaks scale equivariance)
        if self.normalization is not None:
            q_dir = F.normalize(q, dim=2)  # Unit direction
            q_len = q.norm(dim=2)  # Vector magnitudes
            q_len_normalized = self.normalization(q_len)  # Normalize magnitudes
            q = q_dir * q_len_normalized.unsqueeze(2)  # Reconstruct vectors

        # Apply activation only to parallel component
        k_dir = F.normalize(k, dim=2)  # Normalize activation direction
        q_para_len = (q * k_dir).sum(dim=2, keepdim=True)  # Parallel component length
        q_orthogonal = q - q_para_len * k_dir  # Orthogonal component (unchanged)
        
        # Apply activation to parallel component length
        acted_len = self.act_func(q_para_len)
        
        # Reconstruct vector with activated parallel component
        q_acted = q_orthogonal + k_dir * acted_len
        
        # For SE(3)/Sim(3): add back the translation origin
        if self.se3_flag:
            q_acted = q_acted + o
            
        return q_acted


class VecLinearNormalizeActivate(nn.Module):
    """
    Combined Vector Linear, Normalization, and Activation Layer.
    
    This module combines a vector linear transformation with optional normalization
    and activation functions, supporting both vector and scalar features while
    maintaining SO(3), SE(3), or Sim(3) equivariance.
    
    Args:
        in_features (int): Number of input vector channels.
        out_features (int): Number of output vector channels.
        act_func: Activation function to apply.
        s_in_features (int, optional): Number of input scalar channels. Default: 0.
        s_out_features (int, optional): Number of output scalar channels. Default: 0.
        shared_nonlinearity (bool, optional): Whether to share nonlinearity across channels. Default: False.
        normalization (nn.Module, optional): Normalization layer for vectors. Default: None.
        mode (str, optional): Equivariance mode ("so3", "se3", or "sim3"). Default: "sim3".
        s_normalization (nn.Module, optional): Normalization layer for scalars. Default: None.
        vs_dir_learnable (bool, optional): Whether vector-to-scalar direction is learnable. Default: True.
        cross (bool, optional): Whether to use cross product operations. Default: False.
        bias_epsilon (float, optional): Small epsilon value for bias normalization. Default: 1e-6.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        act_func,
        s_in_features=0,
        s_out_features=0,
        shared_nonlinearity=False,
        normalization=None,
        mode="sim3",
        s_normalization=None,
        vs_dir_learnable=True,
        cross=False,
        bias_epsilon=1e-6,
    ) -> None:
        super().__init__()

        mode = mode.lower()
        assert mode in ["so3", "se3", "sim3"], "mode must be so3, se3 or sim3"
        
        # Check if scalar output is required
        self.scalar_out_flag = s_out_features > 0
        
        # Vector linear transformation layer
        self.lin = VecLinear(
            in_features,
            out_features,
            s_in_features,
            s_out_features,
            mode=mode,
            vs_dir_learnable=vs_dir_learnable,
            cross=cross,
            bias_epsilon=bias_epsilon,
        )
        
        # Vector activation layer
        self.act = VecActivation(
            out_features, 
            act_func, 
            shared_nonlinearity, 
            mode, 
            normalization, 
            cross=cross, 
            bias_epsilon=bias_epsilon
        )
        
        # Scalar normalization and activation
        self.s_normalization = s_normalization
        self.act_func = act_func

    def forward(self, v, s=None):
        """
        Forward pass combining linear transformation, normalization, and activation.
        
        Args:
            v (torch.Tensor): Input vector features of shape (B, C, 3, ...)
            s (torch.Tensor, optional): Input scalar features of shape (B, C, ...)
            
        Returns:
            torch.Tensor or tuple: 
                - If s_out_features > 0: (activated_vectors, activated_scalars)
                - Otherwise: activated_vectors
        """
        if self.scalar_out_flag:  # Hybrid vector-scalar mode
            # Apply linear transformation to both vectors and scalars
            v_out, s_out = self.lin(v, s)
            
            # Apply vector activation
            v_act = self.act(v_out)
            
            # Apply scalar normalization if specified
            if self.s_normalization is not None:
                s_out = self.s_normalization(s_out)
            
            # Apply scalar activation
            s_act = self.act_func(s_out)
            
            return v_act, s_act
        else:  # Vector-only mode
            # Apply linear transformation (may still use scalar input for modulation)
            v_out = self.lin(v, s)
            
            # Apply vector activation
            v_act = self.act(v_out)
            
            return v_act


class VecMaxPool(nn.Module):
    """
    Vector Max Pooling Layer for SO(3), SE(3), and Sim(3) equivariant neural networks.
    
    Performs max pooling on vector features while maintaining equivariance properties.
    The pooling is performed along a learned direction, selecting the point that has
    the maximum projection along that direction.
    
    Args:
        in_channels (int): Number of input vector channels.
        mode (str, optional): Equivariance mode ("so3", "se3", or "sim3"). Default: "so3".
        bias_epsilon (float, optional): Small epsilon value for bias normalization. Default: 1e-6.
    """
    
    def __init__(self, in_channels, mode="so3", bias_epsilon=1e-6):
        super(VecMaxPool, self).__init__()
        mode = mode.lower()
        assert mode in ["so3", "se3", "sim3"], "mode must be so3, se3 or sim3"
        
        # SE(3) and Sim(3) require centering operations
        self.se3_flag = mode == "se3" or mode == "sim3"
        
        # Linear layer to learn the pooling direction
        self.map_to_dir = VecLinear(in_channels, in_channels, mode=mode, bias_epsilon=bias_epsilon)
        
        # For SE(3)/Sim(3): additional layer to learn translation origin
        if self.se3_flag:
            self.lin_ori = VecLinear(in_channels, in_channels, mode=mode, bias_epsilon=bias_epsilon)
        
    def forward(self, x):
        """
        Forward pass for vector max pooling.
        
        Args:
            x (torch.Tensor): Input vector features of shape [B, N_feat, 3, npoint, ...]
            
        Returns:
            torch.Tensor: Max pooled features with last dimension reduced
        """
        # Learn the pooling direction
        d = self.map_to_dir(x)
        
        # For SE(3)/Sim(3): center vectors around learned origin
        if self.se3_flag:
            o = self.lin_ori(x)  # Learned translation origin
            d = d - o  # Center direction vectors
            k = x - o  # Center input vectors
        else:
            k = x  # Use input vectors directly for SO(3)
        
        # Compute dot product between centered vectors and direction
        dotprod = (k * d).sum(2, keepdim=True)  # [B, N_feat, 1, npoint, ...]
        
        # Find indices of maximum projections along the last dimension
        idx = dotprod.max(dim=-1, keepdim=False)[1]  # [B, N_feat, 1, ...]
        
        # Create index tuple for advanced indexing to select max elements
        index_tuple = torch.meshgrid([torch.arange(j) for j in x.size()[:-1]]) + (idx,)
        
        # Select the vectors with maximum projection
        x_max = x[index_tuple]
        
        return x_max


class VecLayerNorm(nn.Module):
    """
    Vector Layer Normalization for SO(3), SE(3), and Sim(3) equivariant neural networks.
    
    Applies layer normalization to vector features while maintaining rotational equivariance.
    The normalization is performed on the vector magnitudes, while the directions are
    preserved through proper centering and scaling operations.
    
    Args:
        dim (int): Number of vector channels to normalize.
        eps (float, optional): Small epsilon value to prevent division by zero. Default: 1e-6.
    """
    
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.ln = nn.LayerNorm(dim, eps=self.eps)
        
    def forward(self, x):
        """
        Forward pass for vector layer normalization.
        
        Args:
            x (torch.Tensor): Input vector features of shape (B, N, C, 3)
            
        Returns:
            torch.Tensor: Normalized vector features with same shape as input
        """
        # Center vectors by subtracting mean position (translation equivariance)
        o = x.mean(dim=2, keepdim=True)  # (B, N, 1, 3) - center of mass
        d = x - o  # (B, N, C, 3) - centered vectors
        
        # Compute vector magnitudes for normalization
        norms = d.norm(dim=-1)  # (B, N, C) - vector lengths
        
        # Normalize directions while preventing division by zero
        d = d / rearrange(norms.clamp(min=self.eps), '... -> ... 1')  # (B, N, C, 3)
        
        # Apply layer normalization to the magnitudes
        ln_out = self.ln(norms)  # (B, N, C) - normalized magnitudes
        
        # Reconstruct vectors with normalized magnitudes
        return d * rearrange(ln_out, '... -> ... 1')  # (B, N, C, 3)


class Attend(nn.Module):
    """
    Attention module supporting standard attention, L2 distance-based attention, 
    and PyTorch 2.0+ Flash Attention for improved memory efficiency.
    
    Args:
        dropout (float): Dropout probability for attention weights. Default: 0.
        flash (bool): Whether to use Flash Attention (requires PyTorch 2.0+). Default: False.
        l2_dist (bool): Whether to use L2 distance instead of dot product similarity. Default: False.
    """
    
    def __init__(
        self,
        dropout=0.,
        flash=False,
        l2_dist=False,
    ):
        super().__init__()
        assert not (flash and l2_dist), 'flash attention is not compatible with l2 distance'
        
        self.l2_dist = l2_dist
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.flash = flash

        # Validate PyTorch version for flash attention
        if flash:
            assert not (version.parse(torch.__version__) < version.parse('2.0.0')), \
                'flash attention requires PyTorch 2.0 or above'

        # Configure flash attention for different devices
        self.cpu_config = FlashAttentionConfig(True, True, True)
        self.cuda_config = None

        if torch.cuda.is_available() and flash:
            device_properties = torch.cuda.get_device_properties(torch.device('cuda'))
            if device_properties.major == 8 and device_properties.minor == 0:
                print_once('A100 GPU detected, using flash attention')
                self.cuda_config = FlashAttentionConfig(True, False, False)
            else:
                print_once('Non-A100 GPU detected, using math or mem efficient attention')
                self.cuda_config = FlashAttentionConfig(False, True, True)

    def flash_attn(self, q, k, v, mask=None):
        """Flash attention implementation using PyTorch's scaled_dot_product_attention."""
        _, heads, q_len, _ = q.shape
        is_cuda = q.is_cuda

        # Expand mask to compatible shape if it exists
        if exists(mask):
            mask = mask.expand(-1, heads, q_len, -1)

        # Select appropriate config based on device
        config = self.cuda_config if is_cuda else self.cpu_config

        # Apply flash attention with appropriate configuration
        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.
            )
        return out

    def forward(self, q, k, v, mask=None):
        """
        Forward pass for attention computation.
        
        Args:
            q (torch.Tensor): Query tensor of shape (batch, heads, seq_len, dim)
            k (torch.Tensor): Key tensor of shape (batch, heads, seq_len, dim)  
            v (torch.Tensor): Value tensor of shape (batch, heads, seq_len, dim)
            mask (torch.Tensor, optional): Attention mask
            
        Returns:
            torch.Tensor: Attention output of shape (batch, heads, seq_len, dim)
        """
        scale = q.shape[-1] ** -0.5
        
        # Expand mask dimensions if needed
        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')

        # Use flash attention if enabled
        if self.flash:
            return self.flash_attn(q, k, v, mask=mask)

        # Standard attention computation
        sim = einsum(f"b h i d, b h j d -> b h i j", q, k) * scale

        # Apply L2 distance modification if enabled
        if self.l2_dist:
            q_squared = reduce(q ** 2, 'b h i d -> b h i 1', 'sum')
            k_squared = reduce(k ** 2, 'b h j d -> b h 1 j', 'sum')
            sim = sim * 2 - q_squared - k_squared

        # Apply attention mask
        if exists(mask):
            sim = sim.masked_fill(mask > 0, -torch.finfo(sim.dtype).max)

        # Compute attention weights and apply dropout
        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        
        # Store attention weights for visualization/analysis
        self.last_attn = attn[0, 0]

        # Apply attention to values
        out = einsum(f"b h i j, b h j d -> b h i d", attn, v)
        return out