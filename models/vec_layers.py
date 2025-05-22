import math
import logging
from functools import wraps
from collections import namedtuple
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce

FlashAttentionConfig = namedtuple('FlashAttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])

# helpers

def exists(val):
    return val is not None

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

print_once = once(print)


def channel_equi_vec_normalize(x):
    # B,C,3,...
    assert x.ndim >= 3, "x shape [B,C,3,...]"
    x_dir = F.normalize(x, dim=2)
    x_norm = x.norm(dim=2, keepdim=True)
    x_normalized_norm = F.normalize(x_norm, dim=1)# normalize across C
    y = x_dir * x_normalized_norm
    return y


def transform_restore(x, y, scale_factor=None):  
    # B,C,3,...; B,C,3,...
    if scale_factor is not None:
        assert x.ndim == y.ndim, "x shape [B,N,C,3,...]; y shape [B,N,C,3,...]"
        
        scale = (x - x.mean(dim=2, keepdim=True)).mean(dim=1, keepdim= False).norm(dim=-1).mean(dim=-1, keepdim=False)
        y = y * scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)*scale_factor
    
    return y
    

class VecLinear(nn.Module):
    r"""
    from pytorch Linear
    Can be SO3 or sim3
    Can have hybrid feature
    The input scalar feature must be invariant
    valid mode: V,h->V,h; V,h->V; V->V,h; V->V; V,h->h
    """

    v_in: int
    v_out: int
    s_in: int
    s_out: int
    weight: torch.Tensor

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
        hyper=False,
        bias_epsilon = 1e-6,
    ) -> None:
        mode = mode.lower()
        assert mode in ["so3", "sim3"], "mode must be so3 or sim3"
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.v_in = v_in
        self.v_out = v_out
        self.s_in = s_in
        self.s_out = s_out

        self.hyper_flag = hyper

        assert self.s_out + self.v_out > 0, "vec, scalar output both zero"

        self.sim3_flag = mode == "sim3"
        if self.sim3_flag:
            assert v_in > 1, "sim3 layers must have at least two input layers"

        if self.v_out > 0:
            self.weight = nn.Parameter(
                torch.empty(
                    (v_out, v_in - 1 if self.sim3_flag else v_in), **factory_kwargs
                )  # if use sim3 mode, should constrain the weight to have sum 1.0
            )  # This is the main weight of the vector, due to historical reason, for old checkpoint, not rename this
            self.reset_parameters()

        if (
            self.s_in > 0 and self.v_out > 0
        ):  # if has scalar input, must have a path to fuse to vector
            if self.hyper_flag:
                self.sv_linear = nn.Linear(s_in, int((v_out // 9) * 9))
            else:
                self.sv_linear = nn.Linear(s_in, v_out)
            self.s2v_normalized_scale_flag = s2v_normalized_scale

        if self.s_out > 0:  # if has scalar output, must has vector to scalar path
            self.vs_dir_learnable = vs_dir_learnable
            assert (
                self.vs_dir_learnable
            ), "because non-learnable is not stable numerically, not allowed now"
            if self.vs_dir_learnable:
                self.vs_dir_linear = VecLinear(v_in, v_in, mode="so3")  # TODO: can just have 1 dir
            self.vs_linear = nn.Linear(v_in, s_out)
        if self.s_in > 0 and self.s_out > 0:  # when have s in and s out, has ss path
            self.ss_linear = nn.Linear(s_in, s_out)

        self.cross_flag = cross
        if self.v_out > 0 and self.cross_flag:
            self.v_out_cross = VecLinear(v_in, v_out, mode=mode, cross=False)
            self.v_out_cross_fc = VecLinear(v_out * 2, v_out, mode=mode, cross=False)
            
        
        self.bias = nn.Parameter(torch.randn(v_out))
        self.bias_epsilon = bias_epsilon

    @torch.no_grad()
    def reset_parameters(self) -> None:
        # ! warning, now the initialization will bias to the last channel with larger weight, need better init
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # nn.init.xavier_uniform_(self.weight)
        if self.sim3_flag:
            self.weight.data += 1.0 / self.v_in

    def forward(self, v_input: torch.Tensor, s_input=None):
        # B,C,3,...; B,C,...

        # First do Vector path if output vector
        v_shape = v_input.shape
        assert v_shape[2] == 3, "not vector neuron"
        if self.v_out > 0:
            if self.sim3_flag:
                W = torch.cat(
                    [self.weight, 1.0 - self.weight.sum(-1, keepdim=True)], -1
                ).contiguous()
                # W = W / W.sum(-1, keepdim=True).clamp(min=1e-6)
            else:
                W = self.weight
            if not self.training and self.sim3_flag:
                pass
                # print("W1 mean:", W.mean().item(), "W1 std:", W.std().item())
                # print("W1 min:", W.min().item(), "W1 max:", W.max().item())
                # print("W1 sum:", W.sum(-1).mean().item())
                # print("weight sum:", self.weight.sum(-1).mean().item())
            v_output = F.linear(v_input.transpose(1, -1), W).transpose(-1, 1)  # B,C,3,...
        else:
            v_output = None

        # Optional Scalar path
        if self.s_in > 0:
            assert s_input is not None, "missing scalar input"
            s_shape = s_input.shape
            assert v_shape[3:] == s_shape[2:]
            # must do scalar to vector fusion
            if self.v_out > 0:
                if self.hyper_flag:
                    raise NotImplementedError()
                    s2v_W = self.sv_linear(s_input.transpose(1, -1)).transpose(-1, 1)
                    B, _, N = s2v_W.shape
                    s2v_W = s2v_W.reshape(B, -1, 3, 3, N)
                    head_K = s2v_W.shape[1]
                    s2v_W = (
                        s2v_W.unsqueeze(1)
                        .expand(-1, int(np.ceil(self.v_out / head_K)), -1, -1, -1, -1)
                        .reshape(B, -1, 3, 3, N)
                    )[:, : self.v_out]
                    s2v_W = s2v_W + torch.eye(3).to(s2v_W.device)[None, None, :, :, None]
                    if self.sim3_flag:  # need to scale the rotation part, exclude the center
                        v_new_mean = v_output.mean(dim=1, keepdim=True)
                        v_output = (
                            torch.einsum("bcjn, bcjin->bcin", v_output - v_new_mean, s2v_W)
                            + v_new_mean
                        )
                    else:
                        v_output = torch.einsum("bcjn, bcjin->bcin", v_output, s2v_W)
                else:
                    s2v_invariant_scale = self.sv_linear(s_input.transpose(1, -1)).transpose(-1, 1)
                    if self.s2v_normalized_scale_flag:
                        s2v_invariant_scale = F.normalize(s2v_invariant_scale, dim=1)
                    if self.sim3_flag:  # need to scale the rotation part, exclude the center
                        v_new_mean = v_output.mean(dim=1, keepdim=True)
                        v_output = (v_output - v_new_mean) * s2v_invariant_scale.unsqueeze(
                            2
                        ) + v_new_mean
                    else:
                        v_output = v_output * s2v_invariant_scale.unsqueeze(2)
                    # now v_new done

        if self.v_out > 0 and self.cross_flag:
            # do cross production
            v_out_dual = self.v_out_cross(v_input)
            if self.sim3_flag:
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

        if self.s_out > 0:
            # must have the vector to scalar path
            v_sR = v_input - v_input.mean(dim=1, keepdim=True) if self.sim3_flag else v_input
            if self.vs_dir_learnable:
                v_sR_dual_dir = F.normalize(self.vs_dir_linear(v_sR), dim=2)
            else:
                v_sR_dual_dir = F.normalize(v_sR.mean(dim=1, keepdim=True), dim=2)
            s_from_v = F.normalize((v_sR * v_sR_dual_dir).sum(dim=2), dim=1)  # B,C,...
            s_from_v = self.vs_linear(s_from_v.transpose(-1, 1)).transpose(-1, 1)
            if self.s_in > 0:
                s_from_s = self.ss_linear(s_input.transpose(-1, 1)).transpose(-1, 1)
                s_output = s_from_s + s_from_v
            else:
                s_output = s_from_v
            return v_output, s_output
        else:
            # import pdb; pdb.set_trace()
            bias = F.normalize(self.bias, dim=-1) * self.bias_epsilon
            bias = bias.view(1, -1, *([1] * (v_output.ndim - 2)))  # 自动扩展
            v_output = v_output + bias

            return v_output


class VecActivation(nn.Module):
    # Also integrate a batch normalization before the actual activation
    # Order: 1.) centered [opt] 2.) normalization in norm [opt] 3.) act 4.) add center [opt]
    def __init__(
        self,
        in_features,
        act_func,
        shared_nonlinearity=False,        
        mode="sim3",
        normalization=None,
        cross=False,
        bias_epsilon = 1e-6,
    ) -> None:
        super().__init__()

        mode = mode.lower()
        assert mode in ["so3", "sim3"], "mode must be so3 or sim3"
        self.sim3_flag = mode == "sim3"
        self.shared_nonlinearity_flag = shared_nonlinearity
        self.act_func = act_func

        nonlinear_out = 1 if self.shared_nonlinearity_flag else in_features
        self.lin_dir = VecLinear(in_features, nonlinear_out, mode=mode, cross=cross, bias_epsilon = bias_epsilon)
        if self.sim3_flag:
            self.lin_ori = VecLinear(in_features, nonlinear_out, mode=mode, cross=cross, bias_epsilon = bias_epsilon)
        self.normalization = normalization
        if self.normalization is not None:
            logging.warning("Warning! Set Batchnorm True, not Scale Equivariant")

    def forward(self, x):
        # B,C,3,...
        # warning, there won't be shape check before send to passed in normalization in side this layer
        assert x.shape[2] == 3, "not vector neuron"
        q = x
        k = self.lin_dir(x)
        if self.sim3_flag:
            o = self.lin_ori(x)
            q = q - o
            k = k - o

        # normalization if set
        if self.normalization is not None:
            q_dir = F.normalize(q, dim=2)
            q_len = q.norm(dim=2)  # ! note: the shape into BN is [B,C,...]
            # ! Warning! when set the normalization, not scale equivariant!
            q_len_normalized = self.normalization(q_len)
            q = q_dir * q_len_normalized.unsqueeze(2)

        # actual non-linearity on the parallel component length
        k_dir = F.normalize(k, dim=2)
        q_para_len = (q * k_dir).sum(dim=2, keepdim=True)
        q_orthogonal = q - q_para_len * k_dir
        acted_len = self.act_func(q_para_len)
        q_acted = q_orthogonal + k_dir * acted_len
        if self.sim3_flag:
            q_acted = q_acted + o
        return q_acted


class VecLinearNormalizeActivate(nn.Module):
    # support vector scalar hybrid operation
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
        bias_epsilon = 1e-6,
    ) -> None:
        super().__init__()

        self.scalar_out_flag = s_out_features > 0
        self.lin = VecLinear(
            in_features,
            out_features,
            s_in_features,
            s_out_features,
            mode=mode,
            vs_dir_learnable=vs_dir_learnable,
            cross=cross,
            bias_epsilon = bias_epsilon,
        )
        self.act = VecActivation(
            out_features, act_func, shared_nonlinearity, mode, normalization, cross=cross, bias_epsilon = bias_epsilon
        )
        self.s_normalization = s_normalization
        self.act_func = act_func
        

    def forward(self, v, s=None):
        if self.scalar_out_flag:  # hybrid mode
            v_out, s_out = self.lin(v, s)
            v_act = self.act(v_out)
            if self.s_normalization is not None:
                s_out = self.s_normalization(s_out)
            s_act = self.act_func(s_out)
            return v_act, s_act
        else:
            v_out = self.lin(v, s)
            v_act = self.act(v_out)
            return v_act


class VecMaxPool(nn.Module):
    def __init__(self, in_channels, mode = "so3", bias_epsilon = 1e-6):
        super(VecMaxPool, self).__init__()
        # self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)
        self.map_to_dir = VecLinear(in_channels, in_channels, mode=mode, bias_epsilon = bias_epsilon)
        if mode == "sim3":
            self.lin_ori = VecLinear(in_channels, in_channels, mode=mode)
        self.mode = mode
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, npoint, ...]
        --------
        x_max: max the last dim of x
        '''
        # d = self.map_to_dir(x.transpose(1,-1)).transpose(1,-1)
        d = self.map_to_dir(x)
        if self.mode == "sim3":
            o = self.lin_ori(x)
            d = d - o
            k = x - o
        else:
            k = x
        dotprod = (k*d).sum(2, keepdims=True)
        idx = dotprod.max(dim=-1, keepdim=False)[1]
        index_tuple = torch.meshgrid([torch.arange(j) for j in x.size()[:-1]]) + (idx,)
        x_max = x[index_tuple]
        return x_max


class VecLayerNorm(nn.Module):
    def __init__(self, dim, mode="sim3", eps = 1e-6):
        super().__init__()
        self.eps = eps
        self.ln = nn.LayerNorm(dim, eps = eps)
        self.mode = mode
        
    def forward(self, x):
        # if self.mode == "sim3": 
        d = x 
        o = x.mean(dim=2, keepdim=True)# b, n, 1, 3
        d = d - o
        norms = d.norm(dim = -1) # b, n,c
        d = d / rearrange(norms.clamp(min = self.eps), '... -> ... 1')
        ln_out = self.ln(norms)
        return d * rearrange(ln_out, '... -> ... 1')


class Attend(nn.Module):
    def __init__(
        self,
        dropout = 0.,
        flash = False,
        l2_dist = False,
        mode = 'so3',
        dim = None  # if mode == 'sim3' then dim must be specified
    ):
        super().__init__()
        assert not (flash and l2_dist), 'flash attention is not compatible with l2 distance'
        self.l2_dist = l2_dist
        self.mode = mode

        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        self.flash = flash
        # assert not (flash and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        # determine efficient attention configs for cuda and cpu

        self.cpu_config = FlashAttentionConfig(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

        if device_properties.major == 8 and device_properties.minor == 0:
            print_once('A100 GPU detected, using flash attention if input tensor is on cuda')
            self.cuda_config = FlashAttentionConfig(True, False, False)
        else:
            print_once('Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda')
            self.cuda_config = FlashAttentionConfig(False, True, True)

    def flash_attn(self, q, k, v, mask = None):
        _, heads, q_len, _, k_len, is_cuda, device = *q.shape, k.shape[-2], q.is_cuda, q.device

        # Check if mask exists and expand to compatible shape
        # The mask is B L, so it would have to be expanded to B H N L

        if exists(mask):
            mask = mask.expand(-1, heads, q_len, -1)

        # Check if there is a compatible device for flash attention

        config = self.cuda_config if is_cuda else self.cpu_config

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, softmax_scale

        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask = mask,
                dropout_p = self.dropout if self.training else 0.
            )

        return out

    def forward(self, q, k, v, mask = None):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """
        q_len, k_len, device = q.shape[-2], k.shape[-2], q.device

        scale = q.shape[-1] ** -0.5
        
        if exists(mask) and mask.ndim != 4:
            mask = rearrange(mask, 'b j -> b 1 1 j')

        if self.flash:
            return self.flash_attn(q, k, v, mask = mask)

        # similarity
        q_norm = q
        k_norm = k
       
        sim = einsum(f"b h i d, b h j d -> b h i j", q_norm, k_norm) * scale

        # l2 distance

        if self.l2_dist:
            q_squared = reduce(q_norm ** 2, 'b h i d -> b h i 1', 'sum')
            k_squared = reduce(k_norm ** 2, 'b h j d -> b h 1 j', 'sum')
            sim = sim * 2 - q_squared - k_squared

        # key padding mask

        if exists(mask):
            sim = sim.masked_fill(mask > 0, -torch.finfo(sim.dtype).max)

        # attention
        # import pdb; pdb.set_trace()
        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        self.last_attn = attn[0, 0]
        # aggregate values

        out = einsum(f"b h i j, b h j d -> b h i d", attn, v)

        return out

