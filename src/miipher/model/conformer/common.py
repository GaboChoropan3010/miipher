import torch
import torchaudio
import torch.nn as nn

from torch import Tensor
from torch import sin, pow
from typing import Optional
from torch.nn import Linear
from torch.nn import Parameter
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding

import math
import librosa
import numpy as np
from scipy.signal import get_window

from einops import rearrange, repeat, reduce

# Implementation adapted from https://github.com/EdwardDixon/snake under the MIT license.
#   LICENSE is in incl_licenses directory.

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d


def generate_mask_with_prob(shape, mask_prob, device):
    seq = shape[-1]
    rand = torch.randn(shape, device = device)
    rand[:, 0] = -torch.finfo(rand.dtype).max
    num_mask = min(int(seq * mask_prob), seq - 1)
    indices = rand.topk(num_mask, dim = -1).indices
    mask = ~torch.zeros(shape, device = device).scatter(1, indices, 1.).bool()
    return mask


class Snake(nn.Module):
    '''
    Implementation of a sine-based periodic activation function
    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter
    References:
        - This activation function is from this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    Examples:
        >>> a1 = snake(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    '''
    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
        '''
        Initialization.
        INPUT:
            - in_features: shape of the input
            - alpha: trainable parameter
            alpha is initialized to 1 by default, higher values = higher-frequency.
            alpha will be trained along with the rest of your model.
        '''
        super(Snake, self).__init__()
        self.in_features = in_features

        # initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale: # log scale alphas initialized to zeros
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
        else: # linear scale alphas initialized to ones
            self.alpha = Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        '''
        Forward pass of the function.
        Applies the function to the input elementwise.
        Snake ∶= x + 1/a * sin^2 (xa)
        '''
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1) # line up with x to [B, C, T]
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
        x = x + (1.0 / (alpha + self.no_div_by_zero)) * pow(sin(x * alpha), 2)

        return x

class SnakeBeta(nn.Module):
    '''
    A modified Snake function which uses separate parameters for the magnitude of the periodic components
    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter that controls frequency
        - beta - trainable parameter that controls magnitude
    References:
        - This activation function is a modified version based on this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    Examples:
        >>> a1 = snakebeta(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    '''
    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
        '''
        Initialization.
        INPUT:
            - in_features: shape of the input
            - alpha - trainable parameter that controls frequency
            - beta - trainable parameter that controls magnitude
            alpha is initialized to 1 by default, higher values = higher-frequency.
            beta is initialized to 1 by default, higher values = higher-magnitude.
            alpha will be trained along with the rest of your model.
        '''
        super(SnakeBeta, self).__init__()
        self.in_features = in_features

        # initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale: # log scale alphas initialized to zeros
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
            self.beta = Parameter(torch.zeros(in_features) * alpha)
        else: # linear scale alphas initialized to ones
            self.alpha = Parameter(torch.ones(in_features) * alpha)
            self.beta = Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        '''
        Forward pass of the function.
        Applies the function to the input elementwise.
        SnakeBeta ∶= x + 1/b * sin^2 (xa)
        '''
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1) # line up with x to [B, C, T]
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        x = x + (1.0 / (beta + self.no_div_by_zero)) * pow(sin(x * alpha), 2)

        return x
    

class WeightNorm(nn.Module):
    append_g = '_g'
    append_v = '_v'

    def __init__(self, module, weights):
        super(WeightNorm, self).__init__()
        self.module = module
        self.weights = weights
        self._reset()

    def _reset(self):
        for name_w in self.weights:
            w = getattr(self.module, name_w)

            # construct g,v such that w = g/||v|| * v
            dim = len(w.size())
            g = torch.sqrt(torch.sum(w ** 2, dim=list(range(1, dim)), keepdim=True) + 1e-16)
            v = w / g
            g = nn.Parameter(g.view((-1,)).data)
            v = nn.Parameter(v.data)
            name_g = name_w + self.append_g
            name_v = name_w + self.append_v
            
            # remove w from parameter list
            del self.module._parameters[name_w]

            # add g and v as new parameters
            self.module.register_parameter(name_g, g)
            self.module.register_parameter(name_v, v)

    def _setweights(self):
        for name_w in self.weights:
            name_g = name_w + self.append_g
            name_v = name_w + self.append_v
            g = getattr(self.module, name_g)
            v = getattr(self.module, name_v)
            dim = len(v.size())
            norm_v = torch.sqrt(torch.sum(v ** 2, dim=list(range(1, dim)), keepdim=True) + 1e-16)
            w = v * (g.view_as(norm_v) / norm_v)
            setattr(self.module, name_w, w)

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)
    
class Transpose(nn.Module):
    def __init__(self, shape: tuple):
        super(Transpose, self).__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return x.transpose(*self.shape)
    
class SimpleMultiheadAttention(nn.Module):
    def __init__(self, 
                 d_input: int, 
                 d_model: int, 
                 num_heads: int, 
                 dropout_p: float = 0.1, 
                 use_pos=True, 
                 use_pos_scale=True,  
                 pos_type='relative', 
                 causal=True,):
        super(SimpleMultiheadAttention, self).__init__()

        self.causal = causal
        self.pos_type = pos_type
        self.use_pos = use_pos

        self.query_proj = Linear(d_input, d_model)
        self.key_proj = Linear(d_input, d_model)
        self.value_proj = Linear(d_input, d_model)
        self.out_proj = Linear(d_model, d_model)
        if use_pos:
            
            if pos_type == 'relative':
                self.pos_encode = RelativePositionBias(dim=d_input, heads=num_heads) # Use relative positional coding
            elif pos_type == 'rotary':
                self.rotary_emb = RotaryEmbedding(dim = 32)
            elif pos_type == 'absolute':
                self.pos_encode = PositionalEncoding(d_model = d_input, max_len = 10000)
            
            if use_pos_scale:
                self.pos_scale_k = nn.Parameter(torch.Tensor([1.0]), requires_grad=True)
                self.pos_scale_q = nn.Parameter(torch.Tensor([1.0]), requires_grad=True)
            else:
                self.pos_scale_k = torch.Tensor([1.0])
                self.pos_scale_q = torch.Tensor([1.0])

            
        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.d_model = d_model


    def forward(self, query, key, value, mask: Optional[Tensor] = None): # Original
    
        batch_size, q_seq_length, channels = query.size()
        _, k_seq_length, _ = key.size()

        if self.use_pos:
            if self.pos_type == 'absolute':

                pos_embedding = self.pos_encode(q_seq_length) # shape - (1, seq_length, d_input)
                pos_embedding = pos_embedding.repeat(batch_size, 1, 1)
                query = query + self.pos_scale_q * pos_embedding
                pos_embedding_k = self.pos_encode(key.shape[-2])
                key =  key + self.pos_scale_k * pos_embedding_k
        
            elif self.pos_type == 'rotary':

                query = self.rotary_emb.rotate_queries_or_keys(query)
                key = self.rotary_emb.rotate_queries_or_keys(key)

        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        content_score = torch.matmul(query, key.transpose(2, 3)) # torch.Size([50, num_heads, seq_length, seq_length + nsy_lengthA])
        
        # Relative postional coding is appied on the matrix from dot product of q and k
        score = content_score / ((self.d_head)**0.5)
        
        if self.use_pos and self.pos_type == 'relative':   
            if k_seq_length > q_seq_length: 
                assert k_seq_length - q_seq_length >= q_seq_length # When k_seq_length is not double q_seq_length (when training), don't use relative positioning
                pos_embedding = self.pos_encode(k_seq_length - q_seq_length)[:, :q_seq_length] # When sysnthesizing, q_seq_length can be small
                if pos_embedding.shape[-2] < key.shape[-2]:
                    pos_embedding_attach = pos_embedding[:, :, :q_seq_length]
                    pos_embedding = torch.cat((pos_embedding, pos_embedding_attach), dim=-1) # 
            else:
                pos_embedding = self.pos_encode(q_seq_length)
            # elif pos_embedding.shape[-2] > key.shape
            score = score + pos_embedding

        if mask is not None and self.training: # Adding attention mask only when training
        # if mask is not None: # can have mask when validating, for debugging validtion fluctuation
            # mask = mask.unsqueeze(1)
            score.masked_fill_(~mask, -1e9)

        if self.causal:
            causal_mask = ~torch.ones((query.shape[-2], key.shape[-2]), device = query.device, dtype = torch.bool).triu(key.shape[-2] - query.shape[-2] + 1) 
            score.masked_fill_(~causal_mask, -1e9)

        attn = F.softmax(score, -1)
        context = torch.matmul(attn, value).transpose(1, 2)
        context = context.contiguous().view(batch_size, -1, self.d_model)
        output = self.out_proj(context)

        return output, attn

class FeedForwardModule(nn.Module):
    def __init__(self, model_size, expansion_factor=1, dropout=0.1):
        super(FeedForwardModule, self).__init__()
        self.sequential = nn.Sequential(
            nn.LayerNorm(model_size, eps=1e-6),
            nn.Linear(model_size, model_size * expansion_factor, bias=True),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(model_size * expansion_factor, model_size, bias=True),
            nn.Dropout(p=dropout),
        )
    
    def forward(self, inputs):
        return self.sequential(inputs)
    

class DepthwiseSeparableConv1d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            bias: bool = True,
            w_norm = False,
    ) -> None:
        super(DepthwiseSeparableConv1d, self).__init__()
        self.depthwise = nn.Conv1d(
            in_channels, 
            in_channels, 
            kernel_size, 
            stride=stride, 
            padding=padding, 
            dilation=dilation, 
            groups=in_channels,
            bias=bias,
        )
        if w_norm:
            self.depthwise = WeightNorm(self.depthwise, ["weight"])
        
        self.pointwise = nn.Conv1d(
                        in_channels, 
                        out_channels, 
                        kernel_size=1, 
                        bias=bias)
        if w_norm:
            self.pointwise = WeightNorm(self.pointwise, ["weight"])
    
    def forward(self, x):
        y = self.depthwise(x)
        y = self.pointwise(y)
        return y

class PointwiseConv1d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            padding: int = 0,
            bias: bool = True,
    ) -> None:
        super(PointwiseConv1d, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)

    
class ConformerConvModule(nn.Module):
    def __init__(self, in_channels, filters, out_channels, kernel_size=31, expansion_factor=2, dil_rate=1, dropout=0.1, momentum=0.9, batch_norm=True, causal=False):
        super(ConformerConvModule, self).__init__()
        
        assert (kernel_size - 1) % 2 == 0, "kernel_size should be a odd number for 'SAME' padding"
        assert expansion_factor == 2, "Currently, Only Supports expansion_factor 2"

        self.causal = causal
        self.padding = (kernel_size - 1) * dil_rate if causal else 'same' # Adding causal conv block - Haici
        self.sequential = nn.Sequential(
            nn.LayerNorm(in_channels, eps=1e-6),
            nn.Linear(in_channels, 2 * filters, bias=True),
            Transpose(shape=(1, 2)), # [batch, len, ch] => [batch, ch, len]
            nn.GLU(dim=1),
            Conv1d(
                in_channels=filters,
                out_channels=filters,
                kernel_size=kernel_size,
                groups=filters,
                stride=1,
                padding=self.padding,
                bias=True,
                dilation=dil_rate,
                causal=causal,
            ),
            # nn.Conv1d(
            #     in_channels=filters,
            #     out_channels=filters,
            #     kernel_size=kernel_size,
            #     groups=filters,
            #     stride=1,
            #     padding=self.padding,
            #     bias=True,
            #     dilation=dil_rate,
            # ),
            nn.BatchNorm1d(filters, momentum=1-momentum, eps=0.001) if batch_norm else Identity(), 
            nn.SiLU(),
            Transpose(shape=(1, 2)),
            nn.Linear(filters, out_channels),
            nn.Dropout(p=dropout),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        out = self.sequential(inputs)
        # if self.causal and self.padding != 0:
        #     out = out[:, :-self.padding, :]
        return out


class Conv1d(nn.Conv1d):
    def __init__(self, causal=False, **kwargs):
        super().__init__(**kwargs)
        
        self.causal = causal
        self.left_padding = kwargs['padding']
    
    def forward(self, x: Tensor) -> Tensor:

        out = super().forward(x)
        if self.causal:
            out = out[:, :, :-self.left_padding]

        return out

    
class ConformerBlock(nn.Module):
    # def __init__(self, in_channels, filters, out_channel, dil_rate=1, kernel_size=7, dropout=0.1, momentum=0.9, 
    #              num_heads=6, expansion_factors=[2,4], batch_norm=True, attn_type=None, use_pos=False, use_pos_scale=False, causal=causal):
    def __init__(self, input_size, filters, num_heads, dil_rate, dropout=0.2, expansion_factors=[4,4], attn_mask_prob=0.2, use_global_attn_mask = True, attn_mask=None, attn_type=None, **kwargs):
        super(ConformerBlock, self).__init__()

        self.causal = False
        self.cond_as_prefix = False
        self.crossattn = False
        self.attn_mask_prob = attn_mask_prob
        self.use_global_attn_mask = use_global_attn_mask
        self.transformer_rmv_conv = False

        self.ffm1 = FeedForwardModule(filters, expansion_factors[0], dropout,)
        self.att_layernorm = nn.LayerNorm(filters, eps=1e-6)

        self.att_layernorm_crs = nn.LayerNorm(filters, eps=1e-6)
        self.att_dropout_crs = nn.Dropout(p=dropout)

        # self.ffm_cond = FeedForwardModule(H.filters, H.expansion_factors[1], H.dropout,) # New change
        self.ffm_cond = FeedForwardModule(filters, expansion_factors[0], dropout,) # New change
    
        self.att_cond_layernorm = nn.LayerNorm(filters, eps=1e-6)

        if attn_type=="relative":
            self.mha = RelativeMultiHeadAttention(filters, num_heads, dropout_p=0.1)
        else:
            self.mha = SimpleMultiheadAttention(filters, filters, num_heads=num_heads, 
                                                dropout_p=0.0,  # Not using dropout in SimpleMultiheadAttention
                                                use_pos=True, 
                                                pos_type='rotary', 
                                                use_pos_scale=True, 
                                                causal=False)
            # if self.crossattn:
            #     self.cross_mha = SimpleMultiheadAttention(H.filters, H.filters, num_heads=H.num_heads, 
            #                                     dropout_p=0.0,  # Not using dropout in SimpleMultiheadAttention
            #                                     use_pos=H.cross_use_pos, 
            #                                     pos_type=H.cross_pos_type,
            #                                     use_pos_scale=True, 
            #                                     causal=False)
            
        self.att_dropout = nn.Dropout(p=dropout)
        if not self.transformer_rmv_conv:
            self.confconv = ConformerConvModule(filters, filters, filters, 
                                                kernel_size=5, 
                                                dropout=0.2,
                                                dil_rate=dil_rate,
                                                momentum=0.9,
                                                batch_norm=True,
                                                causal=False)
        self.ffm2 =  FeedForwardModule(filters, expansion_factors[1], dropout,)
        self.out_layernorm = nn.LayerNorm(filters, eps=1e-6)
        
    def forward(self, inputs: Tensor, noisy_z=None, self_attn_mask = None, cross_attn_mask = None, branch_mask = None) -> Tensor:
        
        out_tensor = inputs + 0.5 * self.ffm1(inputs)
        att_tensor = self.att_layernorm(out_tensor) #torch.Size([bt, L, dim])

        assert not self.cond_as_prefix
        # if noisy_z is not None and self.cond_as_prefix:
        #     # Attached the clean code at the end of the noisy code
        #     query = att_tensor
        #     if self.use_rel_pos and att_tensor.shape[1] // noisy_z.shape[1] == self.in_qtz_layers:
        #         noisy_z = torch.repeat_interleave(noisy_z, self.in_qtz_layers, dim=1)
        #     key = torch.cat((noisy_z, att_tensor), 1)
        #     value = torch.cat((noisy_z, att_tensor), 1)
        # else:
        query, key, value = att_tensor, att_tensor, att_tensor

        if self.use_global_attn_mask:
            mask = self_attn_mask
        else:
            mask = generate_mask_with_prob((query.shape[1], key.shape[1]), self.attn_mask_prob, query.device)

        # att_out, att_map = self.mha(att_tensor, att_tensor, att_tensor)
        att_out, self_att_map = self.mha(query, key, value, mask=mask) # self attention
        
        att_x = self.att_dropout(att_out)
        out_tensor = out_tensor + att_x
        
        cross_att_map = None
        if noisy_z is not None and self.crossattn:

            att_tensor = self.att_layernorm_crs(out_tensor)
            cond_tensor = noisy_z + 0.5 * self.ffm_cond(noisy_z)
            att_cond_tensor = self.att_cond_layernorm(cond_tensor)

            if self.use_global_attn_mask:
                mask = cross_attn_mask
            else:
                mask = generate_mask_with_prob((att_tensor.shape[1], att_cond_tensor.shape[1]), self.attn_mask_prob, query.device)
                

            att_out, cross_att_map = self.cross_mha(query=att_tensor, key=att_cond_tensor, value=att_cond_tensor, mask=mask) # Cross attention
            
            att_x = self.att_dropout_crs(att_out)
            
            out_tensor = out_tensor + att_x

        if not self.transformer_rmv_conv:
            out_tensor = out_tensor + self.confconv(out_tensor)
        out_tensor = out_tensor + 0.5 * self.ffm2(out_tensor)
        out_tensor = self.out_layernorm(out_tensor)

        return out_tensor, self_att_map, cross_att_map


class STFT(torch.nn.Module):
    """adapted from Prem Seetharaman's https://github.com/pseeth/pytorch-stft"""
    def __init__(self, 
                 filter_length=800, 
                 hop_length=200, win_length=800,
                 window='hamming'):
        super(STFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.forward_transform = None
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                                   np.imag(fourier_basis[:cutoff, :])])

        forward_basis = torch.tensor(fourier_basis[:, None, :], dtype=torch.float32)

        if window is not None:
            assert(filter_length >= win_length)
            # get window and zero center pad it to filter_length
            fft_window = get_window(window, win_length, fftbins=True)
            fft_window = librosa.util.pad_center(fft_window, size=filter_length)
            fft_window = torch.from_numpy(fft_window).float()

            # window the bases
            forward_basis *= fft_window

        self.forward_basis = forward_basis

        # self.register_buffer('forward_basis', forward_basis.float())

    def to(self, device):
        self.forward_basis = self.forward_basis.to(device)
        return self

    def forward(self, input_data):
        # num_batches = input_data.size(0)
        num_batches = 1
        num_samples = input_data.size(-1)

        # similar to librosa, reflect-pad the input
        input_data = input_data.view(num_batches, 1, num_samples)
        input_data = F.pad(
            input_data.unsqueeze(1),
            (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0),
            mode='reflect')
        input_data = input_data.squeeze(1)

        forward_transform = F.conv1d(
            input_data,
            self.forward_basis,
            stride=self.hop_length,
            padding=0)

        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        magnitude = torch.sqrt(real_part**2 + imag_part**2)

        return magnitude

class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

    def additional_to(self, device):
        return
    
class MelSpec(nn.Module):
    def __init__(self, winsz=512, hopsz=160, fftsz=512, mels=80, 
                       fmin=0, fmax=8000, sr=16000, min_db=-100, 
                       ref_db=16, center=True, log_normalized=True, 
                       noise_floor=0, log_offset=0):
        super(MelSpec, self).__init__()
        self.winsz = winsz
        self.hopsz = hopsz
        self.fftsz = fftsz
        self.mels = mels
        self.fmin = fmin
        self.fmax = fmax
        self.center = center
        self.sr = sr
        self.log_normalized = log_normalized
        # self.stft = STFT(filter_length=self.fftsz, 
        #          hop_length=self.hopsz, 
        #          win_length=self.winsz,
        #          window='hamming')
        self._mel_basis = self._build_mel_basis(sample_rate=sr, fft_size=fftsz, fmin=fmin, fmax=fmax, num_mels=mels)
        self._window = torch.tensor(np.hamming(fftsz))
        # self._log10 = np.log(10)
        self._minlv = torch.tensor(np.exp(min_db / 20.0*np.log(10.0)), dtype=torch.float32)
        self._mindb = min_db
        self._ref_db = ref_db
        self.noise_floor = noise_floor
        self.log_offset = log_offset
        print("Use noise floor:", self.noise_floor, "; log offset:", self.log_offset)
        
    def to(self, device):
        # self.stft.to(device)
        self._mel_basis = self._mel_basis.to(device)
        self._window = self._window.to(device)
        self._minlv = self._minlv.to(device)
        return self
        
    def forward(self, x):
        # out = self.stft(x)
        out = torch.stft(x, self.fftsz, win_length=self.winsz, hop_length=self.hopsz, 
                         window=self._window.to(x.device), 
                         return_complex=True, center=self.center, pad_mode="constant")
        out = torch.abs(out).float()
        out = torch.matmul(self._mel_basis.to(x.device), out)

        if self.log_normalized:
            out = self._amp2db(out)
            out = out - self._ref_db
            out = self._normalize(out)
        
        return out

    def _amp2db(self, x):
        return 20.0 * torch.log10(torch.maximum(self._minlv.to(x.device), x + self.log_offset))
        
    def _normalize(self, x, clip=True):
        x = (x - self._mindb) / (-self._mindb)
        if clip:
            return torch.clip(x, self.noise_floor, 1.0)
        else:
            return x

    def _build_mel_basis(self, sample_rate, fft_size, fmin, fmax, num_mels):
        mel_basis = librosa.filters.mel(sr=sample_rate, 
                            n_fft=fft_size,
                            fmin=fmin, fmax=fmax,
                            n_mels=num_mels)

        mel_basis = torch.tensor(mel_basis, dtype=torch.float32)
        return mel_basis


class PositionalEncoding(nn.Module):
    """
    Positional Encoding proposed in "Attention Is All You Need".
    Since transformer contains no recurrence and no convolution, in order for the model to make
    use of the order of the sequence, we must add some positional information.
    "Attention Is All You Need" use sine and cosine functions of different frequencies:
        PE_(pos, 2i)    =  sin(pos / power(10000, 2i / d_model))
        PE_(pos, 2i+1)  =  cos(pos / power(10000, 2i / d_model))
    """
    def __init__(self, d_model = 512, max_len = 10000) -> None:
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, length) -> Tensor:
        return self.pe[:, :length]


class RelativePositionBias(nn.Module):
    """ from https://arxiv.org/abs/2111.09883 """

    def __init__(
        self,
        *,
        dim,
        heads,
        layers = 3
    ):
        super().__init__()
        self.net = nn.ModuleList([])
        self.net.append(nn.Sequential(nn.Linear(1, dim), nn.SiLU()))

        for _ in range(layers - 1):
            self.net.append(nn.Sequential(nn.Linear(dim, dim), nn.SiLU()))

        self.net.append(nn.Linear(dim, heads))

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, n):
        device = self.device
        pos = torch.arange(n, device = device)
        rel_pos = (rearrange(pos, 'i -> i 1') - rearrange(pos, 'j -> 1 j'))
        rel_pos += (n - 1)

        x = torch.arange(-n + 1, n, device = device).float()
        x = rearrange(x, '... -> ... 1')

        for layer in self.net:
            x = layer(x)

        x = x[rel_pos]

        return rearrange(x, 'i j h -> h i j')