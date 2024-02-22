import os
import numpy as np
from scipy import signal
import math
import torch
from torch import Tensor
import torchaudio.transforms as T
import torchaudio.functional as F
import torchaudio
import librosa
from scipy import interpolate
from audiomentations import SevenBandParametricEQ, LowPassFilter

from .sampling import *
from .dsppy import *

import warnings
warnings.filterwarnings("ignore", message="formats: mp3 can't encode MPEG audio (layer I, II or III) to 16-bit")
import time

EPS = np.finfo(float).eps

class AudioLoader(object):
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.resamplers = {}

    def __call__(self, filename, size=None):
        # waveform: # channels x timesteps
        
        # - Read a random segment of the audio file if too long
        # - Resample to the target sample rate
        
        if size is not None:
            meta = torchaudio.info(filename)
            num_frames = int(math.ceil(float(size) / self.sample_rate) * meta.sample_rate)
            if meta.num_frames > num_frames:
                frame_offset = np.random.randint(meta.num_frames - num_frames)
                waveform, sample_rate = torchaudio.load(filename, frame_offset=frame_offset, num_frames=num_frames)
            else:
                waveform, sample_rate = torchaudio.load(filename)
        else:
            waveform, sample_rate = torchaudio.load(filename)

        if sample_rate != self.sample_rate:
            if sample_rate not in self.resamplers:
                # Approximate "kaiser_best" resampler in Librosa
                new_resampler = T.Resample(sample_rate, self.sample_rate, 
                                        lowpass_filter_width=64,
                                        rolloff=0.9475937167399596,
                                        # resampling_method="kaiser_window",
                                        resampling_method='sinc_interp_kaiser',
                                        beta=14.769656459379492,
                                        dtype=torch.float32)
                self.resamplers[sample_rate] = new_resampler
            waveform = self.resamplers[sample_rate](waveform)
        
        return waveform

class AudioSaver(object):
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate

    def __call__(self, x, filename):
        # x: Tensor either of size (timesteps, ) or (# channels x timesteps)
        if len(x.size()) == 1:
            x = x.unsqueeze(0)
        
        torchaudio.save(filename, x, self.sample_rate)

class SilenceTrim(object):
    '''
    Trim beginning and ending silence
    '''
    def __init__(self, top_db=80):
        self.top_db = top_db

    def __call__(self, x):
        return librosa.effects.trim(x, top_db=self.top_db)

    def reproduce(self, x, indices):
        # x: Tensor of size (..., timesteps)
        if indices is not None and len(indices) == 0:
            st = indices[0]
            ed = indices[-1]
            return x[..., st:ed]
        else:
            return x

class Crop1D(object):
    '''
    Randomly crop for the target size
    If circular is enabled (applicable to noise), automatically cycle the audio snippet for cropping
    '''
    def __init__(self, size, circular=False, pad_mode="constant"):
        self.size = size
        self.circular = circular
        self.pad_mode = pad_mode

    def __call__(self, x, pos):
        # x: Tensor of size (timesteps,)
        if self.circular:
            x = np.tile(x, (math.ceil((self.size + pos) / x.shape[-1]) + 1,))
        else:
            diff = pos + self.size - x.shape[-1]
            if diff > 0:
                half_pad = diff // 2
                x = np.pad(x, (half_pad, diff-half_pad), mode=self.pad_mode)

        return x[..., pos : pos+self.size]

    def random_sample(self, x):
        if self.circular:
            st = np.random.randint(0, x.shape[-1])
            return self(x, st), st
        
        diff = x.shape[-1] - self.size

        if diff > 0:
            st = np.random.randint(0, diff)
            return self(x, st), st
        
        if diff < 0:
            half_pad = (-diff) // 2
            x = np.pad(x, (half_pad, -diff-half_pad), mode=self.pad_mode)
        
        return self(x, 0), 0

class SpeedChange(object): 
    '''
    Augment for speed
    '''
    SPEED_RANGE = (0.8, 1.2)

    def __init__(self, sample_rate, speed_range=SPEED_RANGE):
        self.sample_rate = sample_rate
        self.speed_range = speed_range

    def __call__(self, x, speed):
        '''
        x: Tensor of (Timesteps, )
        Returns: same size as x
        '''
        if speed==1.0:
            return x

        return np.interp(np.arange(0, len(x), speed), np.arange(0, len(x)), x)
    
    def random_sample(self, x):
        if np.random.uniform()<0.5:
            speed = half_gauss_sampling(self.speed_range[0], 1.0, spread=2.0)
        else:
            speed = half_gauss_sampling(1.0, self.speed_range[1], spread=2.0)

        return self(x, speed), speed
    

# class DynamicVolumeNormalizer(object):
#     LEVEL_RANGE = (-30, -20)
#     NUM_PTS_RANGE = (1, 3)

#     def __init__(self, active=False, level_range=LEVEL_RANGE, num_pts_range=NUM_PTS_RANGE):
#         self.active = active
#         self.level_range = level_range
#         self.num_pts_range = num_pts_range

#     def __call__(self, x, target_levels):
#         '''
#         x: Tensor of (# Channels, Timesteps)
#         Returns: same size as x
#         '''
#         mean_target_level = np.mean(target_levels)

#         if self.active:
#             x, pre_scalar = active_volume_normalize(x, target_level=mean_target_level)
#         else:
#             x, pre_scalar = volume_normalize(x, target_level=mean_target_level)

#         gains = target_levels - mean_target_level
#         scalar = 10.0 ** (gains / 20.0)

#         share = x.shape[-1] // len(gains)
#         xp = list((np.arange(len(gains)) * share) + share // 2)
#         interp_scalar = np.interp(np.arange(x.shape[-1]), xp, scalar)

#         f = interpolate.interp1d([0] + xp + [x.shape[-1] - 1], 
#                                  [scalar[0]] + list(scalar) + [scalar[-1]],
#                                  kind='nearest')
#         interp_scalar = f(np.arange(x.shape[-1]))
#         x = x * interp_scalar

#         return x, interp_scalar * pre_scalar

#     def random_sample(self, x):
#         num = np.random.randint(self.num_pts_range[0], self.num_pts_range[1] + 1)
#         target_levels = []
#         for i in range(num):
#             target_level = uniform_sampling(self.level_range[0], self.level_range[1])
#             target_levels.append(target_level)
        
#         x, scaler = self(x, target_levels)
#         return x, scaler, target_levels


class VolumeNormalizer(object):
    '''
    Rescale volume
    '''
    LEVEL_RANGE = (-30, -20)

    def __init__(self, active=False, level_range=LEVEL_RANGE):
        self.active = active
        self.level_range = level_range

    def __call__(self, x, target_level):
        '''
        x: Tensor of (# Channels, Timesteps)
        Returns: same size as x
        '''
        if self.active:
            return active_volume_normalize(x, target_level=target_level)
        else:
            return volume_normalize(x, target_level=target_level)

    def random_sample(self, x):
        target_level = uniform_sampling(self.level_range[0], self.level_range[1])
        y, scalar = self(x, target_level)
        return y, scalar, target_level

# class GainTransition(object):
#     CENTERS = [50, 300, 1500]
#     GAIN_RANGE = (-5, 5)
#     def __init__(self, sample_rate, gain_range=GAIN_RANGE, gain_prob=0.5):
#         self.sample_rate = sample_rate
#         self.transit = GainTransition(
#                 min_gain_in_db = gain_range[0],
#                 max_gain_in_db = gain_range[1],
#                 max_duration=1,
#                 duration_unit="seconds"
#                 p = gain_prob
#             )
    
#     def __call__(self, x, params):
#         self.set_params(params)
#         y = self.transit.apply(x.astype(np.float32), self.sample_rate)        
#         return y

#     def set_params(self, params):
#         self.eq.low_shelf_filter.parameters.update(params[0])
#         for i in range(len(self.eq.peaking_filters)):
#             self.eq.peaking_filters[i].parameters.update(params[i + 1])
#         self.eq.high_shelf_filter.parameters.update(params[-1])

#     def snapshot_params(self):
#         params = []
#         params.append(self.eq.low_shelf_filter.parameters.copy())
#         for i in range(len(self.eq.peaking_filters)):
#             params.append(self.eq.peaking_filters[i].parameters.copy())
#         params.append(self.eq.high_shelf_filter.parameters.copy())
#         return params

#     def random_params(self):
#         self.eq.randomize_parameters(None, self.sample_rate)

#     def random_sample(self, x):
#         self.random_params()
#         params = self.snapshot_params()
#         return self(x, params), params

class Equalization(object):
    '''
    Augment with EQ effects
    '''
    CENTERS = [50, 300, 1500]
    MOD_RANGE = (-8, 8)
    def __init__(self, sample_rate, mod_range=MOD_RANGE, eq_prob=1.0, normalize=True):
        self.sample_rate = sample_rate
        self.eq = SevenBandParametricEQ(
                min_gain_db = mod_range[0],
                max_gain_db = mod_range[1],
                p = eq_prob
            )
        self.normalize = normalize
    
    def __call__(self, x, params):
        self.set_params(params)
        y = self.eq.apply(x.astype(np.float32), self.sample_rate)
        
        # Normalize energy of EQ
        if self.normalize:
            en_ratio = np.sqrt(np.mean(x * x) / (np.mean(y * y) + EPS))
            y = en_ratio * y
        
        return y

    def set_params(self, params):
        self.eq.low_shelf_filter.parameters.update(params[0])
        for i in range(len(self.eq.peaking_filters)):
            self.eq.peaking_filters[i].parameters.update(params[i + 1])
        self.eq.high_shelf_filter.parameters.update(params[-1])

    def snapshot_params(self):
        params = []
        params.append(self.eq.low_shelf_filter.parameters.copy())
        for i in range(len(self.eq.peaking_filters)):
            params.append(self.eq.peaking_filters[i].parameters.copy())
        params.append(self.eq.high_shelf_filter.parameters.copy())
        return params

    def random_params(self):
        self.eq.randomize_parameters(None, self.sample_rate)

    def random_sample(self, x):
        self.random_params()
        params = self.snapshot_params()
        return self(x, params), params

# # TODO: Debug NaN
# class Equalization(object):
#     CENTERS = [50, 300, 1500]
#     MOD_RANGE = (-8, 8)

#     def __init__(self, sample_rate, centers=CENTERS, mod_range=MOD_RANGE):
#         self.sample_rate = sample_rate
#         self.centers = np.array(centers)
#         if len(self.centers.shape)==1:
#             self.centers = self.centers[None, ...]
#         self.mod_range = mod_range

#         self.firs = []
#         self.iirs = []
#         for i in range(self.centers.shape[0]):
#             fir, iir = self._create_filters(self.centers[i], sample_rate)
#             self.firs.append(fir)
#             self.iirs.append(iir)

#     def _create_filters(self, bands, sample_rate, order=4, fir_len=4096):
#         ''' Create constant filters in numpy '''
#         ir = np.zeros([fir_len])
#         ir[0] = 1
#         fir = np.zeros([len(bands) + 1, fir_len])
#         iir = []
#         for j in range(len(bands)):
#             freq = bands[j] / (sample_rate / 2.0)
#             bl, al = signal.butter(order, freq, btype='low')
#             bh, ah = signal.butter(order, freq, btype='high')
#             fir[j] = signal.lfilter(bl, al, ir)
#             ir = signal.lfilter(bh, ah, ir)
#             iir.append([bl, al, bh, ah])
#         fir[-1] = ir
#         pfir = np.square(np.abs(np.fft.fft(fir, axis=1)))
#         pfir = np.real(np.fft.ifft(pfir, axis=1))
#         fir = np.concatenate((pfir[:,fir_len//2:fir_len], pfir[:,0:fir_len//2]), axis=1)
#         return fir, iir

#     def _get_equivalent_fir(self, firs, mods):
#         output = np.zeros((firs.shape[1],))
#         for j in range(0, firs.shape[0]):
#             f = firs[j] * (10.0 ** (mods[j] / 20.0))
#             output += f

#         return output

#     def _apply_fir(self, x, firs, mods):
#         ''' Apply filters in pytorch '''
#         eq_fir = self._get_equivalent_fir(firs, mods)
#         out_x = np.convolve(x, eq_fir)
#         out_x = out_x[eq_fir.shape[-1]//2:-eq_fir.shape[-1]//2+1]
#         return out_x

#     def _apply_iir(self, x, iirs, mods):
#         output = np.zeros_like(x)
#         for j in range(len(iirs)):
#             [bl, al, bh, ah] = iirs[j]
#             x1 = signal.lfilter(bl, al, x)
#             x = signal.lfilter(bh, ah, x)
#             output += x1 * (10.0 ** (mods[j]/20.0))
#         output += x * (10.0 ** (mods[-1]/20.0))
#         return output

#     def __call__(self, x, irs_mods):
#         idx, mods = irs_mods
        
#         # firs = self.firs[idx]
#         # y = self._apply_fir(x, firs, mods)
        
#         ## Use iir (rather than fir) for efficiency
#         iirs = self.iirs[idx]
#         y = self._apply_iir(x, iirs, mods)
        
#         # Normalize energy of EQ
#         en_ratio = np.sqrt(np.mean(x * x) / (np.mean(y * y) + EPS))
#         y = en_ratio * y
        
#         return y

#     def random_params(self):
#         idx = discrete_sampling([1,] * self.centers.shape[0])
#         u = (self.mod_range[0] + self.mod_range[1]) // 2
#         std = abs(self.mod_range[1] - u)

#         mods = np.random.normal(u, std, len(self.firs[idx]) + 1)
#         mods = np.clip(mods, u-std, u+std)

#         return idx, mods

#     def random_sample(self, x):
#         idx, mods = self.random_params()
#         return self(x, (idx, mods)), (idx, mods)

class Codec(object):
    'Augment with codec effect in torch'

    'Pre-defined config options'
    CONFIGS = [
        ({"format": "wav", 
            "bits_per_sample": [8, 16, 32, 64]}, 0),
        ({"format": "flac", 
            "compression": list(range(9)), 
            "bits_per_sample": [8, 16, 24]}, 0),
        ({"format": "mp3", 
            "compression": [96.01, 96.1, 96.2, 96.3, 96.4, 96.5, 96.6, 96.7, 96.8, 96.9, 
                            128.01, 128.1, 128.2, 128.3, 128.4, 128.5, 128.6, 128.7, 128.8, 128.9, 
                            192.01, 192.1, 192.2, 192.3, 192.4, 192.5, 192.6, 192.7, 192.8, 192.9, 
                            256.01, 256.1, 256.2, 256.3, 256.4, 256.5, 256.6, 256.7, 256.8, 256.9]
                            }, 1),
    ]

    def __init__(self, sample_rate, configs=CONFIGS):
        self.sample_rate = sample_rate
        self.configs = configs
    
    'Apply codec to a waveform tensor'
    def __call__(self, x: Tensor, config):
        x_ = F.apply_codec(x, sample_rate=self.sample_rate, **config)
        if config["format"]=="mp3":
            # Hard-coded padding
            x_ = x_[..., 1105:1105+x.size(-1)]
        return x_

    'Randomly select a codec based on weights, and apply it to a waveform tensor'
    def random_sample(self, x: Tensor):
        idx = discrete_sampling(list(zip(*self.configs))[1])
        config = self.configs[idx][0]
        for key in config:
            if isinstance(config[key], list):
                sub_idx = discrete_sampling([1, ]*len(config[key]))
                config[key] = config[key][sub_idx]

        return self(x, config), config

class SpeechAugmenter(object):
    '''
    Chains speech augmentation
    '''
    def __init__(self, sample_rate, size=None, speed_ramge=(0.8, 1.2), target_level=None):
        self.sample_rate = sample_rate
        self.size = size
        self.speed_ramge = speed_ramge
        self.target_level = target_level
        
        normal_cutoff = 30.0 / (0.5 * sample_rate)
        self.b, self.a = signal.butter(5, normal_cutoff, btype='high', analog=False)
        
        # self.trim = SilenceTrim()
        self.speed_change = SpeedChange(sample_rate, speed_range=speed_ramge)
        self.crop = Crop1D(size, circular=False, pad_mode="reflect")
        self.volume = VolumeNormalizer(active=True)

    def augment(self, x):
        '''
        x: list - concatenate multiple speeches sequentially
        '''
        if not isinstance(x, list):
            x = [x]
        
        speed = []
        flip = []
        new_x = []
        for item in x:
            # Remove 30Hz below
            ####################################
            ## TODO: disable silence trim for now to allow some silence snippets in training
            # item, indices = self.trim(item)
            indices = None
            ####################################
            
            item = signal.filtfilt(self.b, self.a, item)            
            
            if self.speed_ramge:
                item, speed_i = self.speed_change.random_sample(item)
            else:
                speed_i = None
            speed.append(speed_i)

            flip_i = np.random.rand() < 0.5
            if flip_i:
                item = -item
            flip.append(flip_i)
            
            ## Allow some padding in between
            item = np.pad(item, (2000, 0))
            new_x.append(item)

        x = np.concatenate(new_x)

        if self.size:
            x, pos = self.crop.random_sample(x)
        else:
            pos = None

        if self.target_level is not None:
            x, _ = self.volume(x, target_level=self.target_level)
        
        return x, {"trim": indices, "speed": speed, "crop": pos, "level": self.target_level, "flip": flip}

    def reproduce(self, x, params):
        if not isinstance(x, list):
            x = [x]
        
        new_x = []
        for i, item in enumerate(x):
            # if params.get("trim", None) is not None:
                # item = self.trim.reproduce(item, params["trim"])
                
            item = signal.filtfilt(self.b, self.a, item)
        
            if params.get("speed", [])[i] is not None:
                item = self.speed_change(item, params["speed"])
            if params.get("flip", [])[i]:
                item = -item
            
            ## Allow some padding in between
            item = np.pad(item, (2000, 0))
            new_x.append(item)
        
        x = np.concatenate(new_x)

        if params.get("crop", None) is not None:
            x = self.crop(x, params["crop"])
        if params.get("level", None) is not None:
            x, _ = self.volume(x, target_level=params["level"])
            
        return x

class IRAugmenter(object):
    '''
    Chains reverb augmentation
    '''
    def __init__(self, sample_rate, 
                 drr_prob=0.8, 
                 rt60_prob=0.8, 
                 drr_scale=(0.5, 2.0), 
                 rt60_scale=(0.5, 2.0), 
                 norm_type="peak",
                 drr_limit=(1.0, None),
                 rt60_limit=(None, None),
                 ):
        self.sample_rate = sample_rate
        self.drr_prob = drr_prob
        self.rt60_prob = rt60_prob
        self.drr_scale = drr_scale
        self.rt60_scale = rt60_scale
        self.norm_type = norm_type
        self.drr_limit = drr_limit
        self.rt60_limit = rt60_limit

        # For internal use with rt60 scaling
        self._speed_change = SpeedChange(self.sample_rate)

    def _decompose_ir(self, ir, window):
        '''
        ir: Tensor of (timesteps, )
        window: half width of direct arrival window
        '''
        x_max = np.argmax(ir)
        start_idx = max(x_max-window, 0)
        end_idx = x_max + window + 1
        ir_early = ir[start_idx:end_idx]
        ir_late = np.array(ir)
        ir_late[start_idx:end_idx] = 0.0
        return ir_early, ir_late, (start_idx, x_max, end_idx)

    def calculate_drr(self, ir):
        '''
        ir: Tensor of (timesteps, )
        '''
        window = int(2.5/1000.0 * self.sample_rate)
        ir_early, ir_late, _ = self._decompose_ir(ir, window)
        drr_db = 10.0 * np.log10(np.sum(ir_early**2) / (np.sum(ir_late**2) + EPS))
        return drr_db

    def peak_normalize(self, ir):
        '''
        ir: Tensor of (timesteps, )
        '''
        window = int(2.5/1000.0 * self.sample_rate)
        ir_early, _, _ = self._decompose_ir(ir, window)
        # hamming_window = torch.hamming_window(2 * window + 1, periodic=False) 
        # hamming_window = hamming_window[torch.clamp(window-indices[1], min=0):]
        # ir_early = hamming_window * ir_early
        direct_energy = np.sqrt(np.sum(ir_early**2)) + EPS
        ir = ir / direct_energy
        return ir
    
    def energy_normalize(self, ir):
        '''
        ir: Tensor of (timesteps, )
        '''
        energy = np.sqrt(np.sum(ir**2)) + EPS
        ir = ir / energy
        return ir

    def scale_drr(self, ir, gain):        
        '''
        Scale off direct signal component of impulse response to augment on DRR

        ir: Tensor of (timesteps, )
        all_levels: return augmentation of evenly spaced drr gains in both [l, 1] and [1, h] intervals
        gain: scale factor on direct signal component
        '''
        window = int(2.5/1000.0 * self.sample_rate)
        ir_early, ir_late, indices = self._decompose_ir(ir, window)
                
        hamm_window = np.hamming(2 * window + 1) 
        hamm_window = hamm_window[max(window-indices[1], 0):]

        win_ir_early = hamm_window * ir_early
        residual_ir_early = (1.0-hamm_window) * ir_early
        
        direct_ir_early = gain * win_ir_early

        new_ir_early = direct_ir_early + residual_ir_early

        ### Reverberation energy no larger than direct signal
        ir_final = np.array(ir_late)
        ir_final[indices[0]:indices[2]] = new_ir_early
        
        return ir_final

    def random_scale_drr(self, ir, all_levels=None, l=None, h=None):        
        '''
        Scale off direct signal component of impulse response to augment on DRR

        ir: Tensor of (timesteps, )
        all_levels: return augmentation of evenly spaced drr gains in both [l, 1] and [1, h] intervals
        l: optional, lower bound for scale factor in random sampling (we ensure the direct peak is no smaller than the late peak)
        h: optional, upper bound for scale factor in random sampling
        '''
        if l is None:
            l = self.drr_scale[0]
        if h is None:
            h = self.drr_scale[1]

        window = int(2.5/1000.0 * self.sample_rate)
        ir_early, ir_late, indices = self._decompose_ir(ir, window)
        
        direct_max = np.max(np.abs(ir_early))
        next_max = np.max(np.abs(ir_late))
        
        '''Decide aug level'''
        
        l = max(l, self.drr_limit[0] * next_max / (direct_max + EPS) + EPS)
        
        if self.drr_limit[1] is not None:
            h = min(h, self.drr_limit[1] * next_max / (direct_max + EPS) + EPS)
        
        if all_levels is not None:
            if l < 1.0:
                gain_up = expon_sampling(1.0, h, all_levels=all_levels)
                gain_down = expon_sampling(1.0, l, all_levels=all_levels)
                gain = np.concatenate([gain_up, gain_down])
            else:
                gain = expon_sampling(l, h, all_levels=all_levels)
        else:
            if l < 1.0:
                if np.random.uniform()<0.5:
                    gain = expon_sampling(1.0, h)
                else:
                    gain = expon_sampling(1.0, l)
            else:
                gain = expon_sampling(l, h)
            gain = [gain]

        new_irs = []    
        for i in range(len(gain)):
            new_irs.append(self.scale_drr(ir, gain[i]))
 
        if len(new_irs)==1:
            return new_irs[0], {"gain": gain[0]}
        else:
            return new_irs, {"gain": gain}

    def scale_rt60(self, ir, gain_early, gain_late):    
        '''
        Change speed of early and late parts of impulse response to augment on RT60

        ir: Tensor of (timesteps, )
        gain_early: speed factor on direct arrival component
        gain_late: speed factor on late component
        '''    
        window = int(2.5/1000.0 * self.sample_rate)
        ir_early, ir_late, indices = self._decompose_ir(ir, window)

        if gain_early != 1.0:
            new_ir_early = self._speed_change(np.array(ir_early), gain_early)
        else:
            new_ir_early = np.array(ir_early)
        if gain_late != 1.0:
            new_ir_late = self._speed_change(np.array(ir_late[indices[-1]:]), gain_late)
        else:
            new_ir_late = np.array(ir_late[indices[-1]:])
        new_ir = np.concatenate([new_ir_early, new_ir_late], axis=0)
        
        return new_ir

    def random_scale_rt60(self, ir, all_levels=None, l=None, h=None):
        '''
        Change speed of early and late parts of impulse response to augment on RT60

        ir: Tensor of (timesteps, )
        all_levels: return augmentation of evenly spaced rt60 gains in both [l, 1] and [1, h] intervals
        l: optional, lower bound for speed factor on direct arrival component (we ensure the direct peak is no smaller than the late peak)
        h: optional, upper bound for speed factor on late component
        ''' 
           
        if l is None:
            l = self.rt60_scale[0]

        if h is None:
            h = self.rt60_scale[1]
        
        tail_sec = float(len(ir) - np.argmax(ir)) / self.sample_rate
            
        if self.rt60_limit[1] is not None:
            # Smaller spped rate leads to longer tail, limit in seconds
            l = max(l,  tail_sec / (self.rt60_limit[1] + EPS))
        if self.rt60_limit[0] is not None:
            # Larger speed rate leads to smaller tail, limit in seconds
            h = min(h,  tail_sec / (self.rt60_limit[0] + EPS))
            
        if all_levels is not None:
            gain_late = []
            if h >= 1.0:
                gain_up = half_gauss_sampling(max(l, 1.0), h, all_levels=all_levels)
                gain_late.append(gain_up)
            if l <= 1.0:
                gain_down = half_gauss_sampling(l, min(1.0, h), all_levels=all_levels)
                gain_late.append(gain_down)
            gain_late = np.concatenate(gain_late)
            gain_early = [1.0, ] * len(gain_late)
        else:
            gain_late = []
            if h >= 1.0:
                gain_late.append(half_gauss_sampling(max(l, 1.0), h))
            if l <= 1.0:
                gain_late.append(half_gauss_sampling(l, min(1.0, h)))
            gain_late = np.random.choice(gain_late)
            
            gain_early = 1.0 ## Fix early reflection part for now
            ## TODO: gain_early = half_gauss_sampling(1.0, 1.0 + (gain_late - 1.0)/ 2.0)
            
            gain_late = [gain_late]  
            gain_early = [gain_early]

        new_irs = []
        for i in range(len(gain_early)):
            new_irs.append(self.scale_rt60(ir, gain_early[i], gain_late[i]))
 
        if len(gain_early)==1:
            return new_irs[0], {"gain_early": gain_early[0], "gain_late": gain_late[0]}
        else:
            return new_irs, {"gain_early": gain_early, "gain_late": gain_late}

    def augment(self, ir):
        # Flip IR if the peak is not positive
        if np.max(np.abs(ir)) != np.max(ir):
            ir = -ir

        if np.random.uniform() < self.drr_prob:
            ir, drr_params = self.random_scale_drr(ir)
        else:
            drr_params = None

        if np.random.uniform() < self.rt60_prob:
            ir, rt60_params = self.random_scale_rt60(ir)
        else:
            rt60_params = None
            
        if self.norm_type == "peak":
            ir = self.peak_normalize(ir)
        elif self.norm_type == "energy":
            ir = self.energy_normalize(ir)
        else:
            print("Unrecognized norm type for IR:", self.norm_type)
            os._exit(1)
            
        return ir, {"drr_params": drr_params, "rt60_params": rt60_params}

    def reproduce(self, ir, params):
        # Flip IR if the peak is not positive
        if np.max(np.abs(ir)) != np.max(ir):
            ir = -ir

        if params.get("drr_params", None) is not None:
            ir = self.scale_drr(ir, **params["drr_params"])
        
        if params.get("rt60_params", None) is not None:
            ir = self.scale_rt60(ir, **params["rt60_params"])

        if self.norm_type == "peak":
            ir = self.peak_normalize(ir)
        elif self.norm_type == "energy":
            ir = self.energy_normalize(ir)
        else:
            print("Unrecognized norm type for IR:", self.norm_type)
            os._exit(1)
            
        return ir

class NoiseAugmenter(object):
    '''
    Chains noise augmentation
    '''
    SNR_RANGE = (-10, 10)
    def __init__(self, sample_rate, size=None, snr_range=SNR_RANGE):
        self.sample_rate = sample_rate
        
        self.crop = Crop1D(size, circular=True)
        self.eq = Equalization(self.sample_rate)
        self.adder = NoiseAdder(sample_rate, snr_range=snr_range, name="noise_mixer")

    def generate_gauss_noise(self, length):
        return np.normal(0, 1, size=(length,))
        
    def augment(self, x):
        if not isinstance(x, list):
            x = [x]
        
        base = None
        params = []
        for i in range(len(x)):
            sub_x = x[i]
            sub_x, pos = self.crop.random_sample(sub_x)
            sub_x, eq_params = self.eq.random_sample(sub_x)
            flip = np.random.rand() < 0.5
            if flip:
                sub_x = -sub_x
            # Mix noises together
            if base is None:
                base = sub_x
                snr_db = 0
            else:
                base, _, snr_db= self.adder.random_mix(base, sub_x)
            params.append({"crop": pos, "eq": eq_params, "snr_db": snr_db, "flip": flip})
        
        return base, params

    # Cannot reproduce exactly due to EQ
    def reproduce(self, x, params):
        if isinstance(x, list):
            x = [x]
        
        base = None
        for i in range(len(x)):
            sub_x = x[i]
            sub_params = params[i]
            if sub_params.get("crop", None) is not None:
                sub_x = self.crop(sub_x, sub_params["crop"])
            if sub_params.get("eq", None) is not None:
                sub_x = self.eq(sub_x, sub_params["eq"])
            if sub_params.get("flip", False):
                sub_x = -sub_x
            # Mix noises together
            if base is None:
                base = sub_x
            else:
                base, _ = self.adder.mix(base, sub_x, sub_params["snr_db"])

        return base

class Clipping(object):
    CLIP_RANGE = (0.5, 1.0)

    def __init__(self, clip_range=CLIP_RANGE):
        self.clip_range = clip_range

    def __call__(self, x, rel_thresh):
        # TODO: change to gain?
        peak_val = np.max(np.abs(x))
        out_x = np.clip(x, -peak_val * rel_thresh, peak_val * rel_thresh)
        return out_x

    def random_sample(self, x):
        rel_thresh = uniform_sampling(self.clip_range[0], self.clip_range[1])
        return self(x, rel_thresh), rel_thresh

class Resampler(object):
    LOWPASS_FILTER_WIDTH = [6, 16, 32, 64]
    ROLLOFF_RANGE = (0.94, 0.99)
    RESAMPLING_METHOD = ["sinc_interpolation", "sinc_interp_kaiser"]

    def __init__(self, in_sample_rate, out_sample_rate,
                    lowpass_filter_width=LOWPASS_FILTER_WIDTH, 
                    rolloff=ROLLOFF_RANGE, 
                    resampling_method=RESAMPLING_METHOD):
        self.in_sample_rate = in_sample_rate
        self.out_sample_rate = out_sample_rate
        self.lowpass_filter_width = lowpass_filter_width
        self.rolloff = rolloff
        self.resampling_method = resampling_method

    def __call__(self, x, params):
        out_x = F.resample(x, self.in_sample_rate, self.out_sample_rate,
                                lowpass_filter_width=params["lowpass_filter_width"], 
                                rolloff=params["rolloff"],
                                resampling_method=params["resampling_method"])
        return out_x

    def random_sample(self, x):
        lowpass_filter_width = np.random.choice(self.lowpass_filter_width)
        rolloff = np.random.uniform(self.rolloff[0], self.rolloff[1])
        resampling_method = np.random.choice(self.resampling_method)

        params = {"lowpass_filter_width": lowpass_filter_width, 
                   "rolloff": rolloff, 
                   "resampling_method": resampling_method}
        return self(x, params), params


class Denoise(object):
    def __init__(self):
        return

    def __call__(self, x, params):
        import pyroomacoustics
        y = pyroomacoustics.denoise.spectral_subtraction.apply_spectral_sub(
                            x, nfft=512,
                            db_reduc=params["db_reduc"], lookback=5,
                            beta=params["beta"], alpha=params["alpha"])
        return y

    def random_sample(self, x):
        try:
            db_reduc = np.random.uniform(5, 15)
            alpha = np.random.uniform(1, 3)
            beta = np.random.uniform(10, 20)

            params = {"db_reduc": db_reduc, "alpha": alpha, "beta": beta}
            return self(x, params), params
        except Exception as e:
            return x, None

# class LowPass(object):
#     def __init__(self, sample_rate):
#         self.lp = LowPass(
#                 min_cutoff_freq = min(sample_rate // 2, 4000),
#                 max_cutoff_freq = sample_rate // 2 - 1000,
#                 p = 0.5
#             )
#         return

#     def __call__(self, x, params):
#         y = self.lp(x)
#         return y

#     def random_sample(self, x):
#         y = self.lp(x)
#         return self(x, params), params

class SndFX(object):
    SNDFX_PROBS = {
        "overdrive": 0.2,
        "phaser": 0.2,
        "compand": 0.8,
        "lowpass": 0
    }

    def __init__(self, sample_rate, sndfx_probs=SNDFX_PROBS):
        self.sample_rate = sample_rate
        self.sndfx_probs = sndfx_probs
        return

    def _gen_overdrive(self):
        return {
            "gain": np.random.uniform(0, 20), 
            "colour": np.random.uniform(0, 100)
            }

    def _gen_compand(self):
        attack = np.random.uniform(0.01, 0.1)
        decay = np.random.uniform(attack, 0.3)
        soft_knee = np.random.randint(2.0, 7.0)
        db_from = np.random.randint(-8, -1) * 10
        db_to = min(math.floor(db_from / 60) * 10, np.random.randint(db_from // 10 + 1, 0) * 10)
        threshold = db_from - np.random.randint(1, 15)
        return {
            "attack": attack,
            "decay": decay,
            "soft_knee": soft_knee,
            "db_from": db_from,
            "db_to": db_to,
            "threshold": threshold,
        }

    def _gen_phaser(self):
        return {
            "gain_in": 1.0, 
            "gain_out": 1.0, 
            "delay": np.random.uniform(1, 5), 
            "decay": np.random.uniform(0.1, 0.5), 
            "speed": np.random.uniform(0.1, 2.0)
            }
        
    def _gen_lowpass(self):
        return {
            "frequency": np.random.randint(1000, min(self.sample_rate // 2 - 2000, 15000))
        }
        
    def __call__(self, x, params):
        ######### Sndfxs

        from pysndfx import AudioEffectsChain
        fx = AudioEffectsChain()

        for item in params:
            if item[0]=="phaser":
                fx.phaser(**item[1])
            elif item[0]=="overdrive":
                fx.overdrive(**item[1])
            elif item[0]=="compand":
                fx.compand(**item[1])
            elif item[0]=="lowpass":
                fx.lowpass(**item[1])

        y = fx(x)

        return y

    def random_sample(self, x, order=None):
        sndfx_options = { 
            "overdrive": self._gen_overdrive(),
            "phaser": self._gen_phaser(),
            "compand": self._gen_compand(),
            "lowpass": self._gen_lowpass(),
        }
        
        if order is None:
            order = []
            for item in sndfx_options.keys():
                if item in self.sndfx_probs and np.random.uniform() < self.sndfx_probs[item]:
                    order.append(item)

        if len(order)==0:
            return x, []
        
        np.random.shuffle(order)

        params = []
        for item in order:
            params.append((item, sndfx_options[item]))

        return self(x, params), params


class LowPass(object):
    def __init__(self, sample_rate,):
        self.sample_rate = sample_rate
        self.lp = LowPassFilter(
                min_cutoff_freq = 1000,
                max_cutoff_freq = self.sample_rate // 2 - 2000,                
                min_rolloff = 12,
                max_rolloff = 24,
                zero_phase = True,
                p = 1.0,
            )
        
    def __call__(self, x, params):
        self.set_params(params)
        y = self.lp.apply(x.astype(np.float32), self.sample_rate)
        return y

    def set_params(self, params):
        self.lp.parameters.update(params)
        
    def snapshot_params(self):
        params = self.lp.parameters.copy()
        return params
    
    def random_params(self):
        self.lp.randomize_parameters(None, self.sample_rate)
        
    def random_sample(self, x, order=None):
        self.random_params()
        params = self.snapshot_params()
        return self(x, params), params


class PostAugmenter(object):
    '''
    Chains post-effects augmentation
    '''
    def __init__(self, sample_rate, 
                    level_range=VolumeNormalizer.LEVEL_RANGE, 
                    clip_range=Clipping.CLIP_RANGE, 
                    clip_prob=0.5,
                    sndfx_prob=0.3,
                    denoise_prob=0,
                    lowpass_prob=0,
                    sndfx_option_probs=SndFX.SNDFX_PROBS):
        # self.volume = DynamicVolumeNormalizer(level_range=level_range, 
        #                                       num_pts_range=num_pts_range)
        self.volume = VolumeNormalizer(active=False, level_range=level_range)
        self.sndfx = SndFX(sample_rate, sndfx_probs=sndfx_option_probs)
        self.denoise = Denoise()
        # TODO: Jiaqi - simulate band limitation
        self.lowpass = LowPass(sample_rate)
        self.clip = Clipping(clip_range=clip_range)
        # self.codec = Codec(sample_rate)

        self.clip_prob = clip_prob
        self.sndfx_prob = sndfx_prob
        self.denoise_prob = denoise_prob
        self.lowpass_prob = lowpass_prob

    def augment(self, x, denoise=True):
        if np.random.uniform() < self.sndfx_prob:
            x, sndfx_params = self.sndfx.random_sample(x)
        else:
            sndfx_params = None

        if denoise and np.random.uniform() < self.denoise_prob:
            x, denoise_params = self.denoise.random_sample(x)
        else:
            denoise_params = None
            
        if np.random.uniform() < self.lowpass_prob:
            x, lowpass_params = self.lowpass.random_sample(x)
        else:
            lowpass_params = None
                
        x, scalar, volume_params = self.volume.random_sample(x)

        if np.random.uniform() < self.clip_prob:
            x, clip_params = self.clip.random_sample(x)
        else:
            clip_params = None

        # x, codec_params = self.codec.random_sample(x)
        codec_params = None
        
        return x, {"volume": volume_params, 
                    "scalar": scalar, 
                    "clip": clip_params, 
                    "codec": codec_params,
                    "sndfx": sndfx_params,
                    "denoise": denoise_params,
                    "lowpass": lowpass_params,
                    }

    def reproduce(self, x, params):
        if params.get("sndfx", None) is not None:
            x = self.sndfx(x, params["sndfx"])

        if params.get("denoise", None) is not None:
            x = self.denoise(x, params["denoise"])

        if params.get("lowpass", None) is not None:
            x = self.lowpass(x, params["lowpass"])
            
        if params.get("volume", None) is not None:
            x, scalar = self.volume(x, params["volume"])
            
        if params.get("clip", None) is not None:
            x = self.clip(x, params["clip"])
        # if params.get("codec", None) is not None:
        #     x = self.codec(x, params["codec"])
        return x, scalar
    
class NoiseAdder(object):
    '''
    Mix audio with noise at a randomly sampled SNR
    '''
    SNR_RANGE =(-10, 10)

    def __init__(self, sample_rate, snr_range=SNR_RANGE, name="noise_adder"):
        self.snr_range = snr_range
        self.sample_rate = sample_rate
        self.name = name
    
    def mix(self, audio, noise, snr_db=None):
        if snr_db is None:
            snr_db = uniform_sampling(self.snr_range[0], self.snr_range[1])

        en_noise = peak_perc_energy(noise) + EPS
        en_audio = peak_perc_energy(audio)
        
        if not activity_threshold(en_audio):
            scaled_noise = noise[..., :audio.shape[-1]]
            noisy_audio = audio + scaled_noise
        else:
            mag_snr = np.sqrt(10.0 ** (-snr_db / 10.0) * en_audio / en_noise)

            scaled_noise = noise * mag_snr
            scaled_noise = scaled_noise[..., :audio.shape[-1]]
            noisy_audio = audio + scaled_noise

        return noisy_audio, scaled_noise
        
    def random_mix(self, audio, noise):
        snr_db = uniform_sampling(self.snr_range[0], self.snr_range[1])
        noisy, scaled_noise = self.mix(audio, noise=noise, snr_db=snr_db)
        return noisy, scaled_noise, {"snr_db": snr_db}
    
class AcousticMixer(object):
    '''
    Apply reverb, add noise and apply EQ, with randomly sampled augmentation paramters
    '''
    
    SNR_RANGE =(-10, 30)
    
    def __init__(self, sample_rate, 
                    eq_centers=Equalization.CENTERS, # TODO: Unused
                    eq_mod_range=Equalization.MOD_RANGE,
                    eq_prob=1.0, 
                    snr_range=SNR_RANGE,
                    master_mixing_range=None,
                    rir_decomposed=False):
        self.sample_rate = sample_rate
        self.snr_range = snr_range
        self.eq_mod_range = eq_mod_range
        self.master_mixing_range = master_mixing_range
        self.rir_decomposed = rir_decomposed
        
        self.adder = NoiseAdder(sample_rate, snr_range)

        ## TODO: replaced with audiomentations' SevenBandParametricEQ
        self.eq_prob = eq_prob
        self.eq = Equalization(sample_rate, 
                                    # eq_centers, 
                                    mod_range=eq_mod_range,
                                    normalize=not rir_decomposed)
        
    
    def _convolve_rir(self, speech, rir):
        '''
        speech: Tensor of (SPEECH timesteps)
        rir: Tensor of (RIR timesteps, )
        '''
        if self.rir_decomposed:
            pre_delay = np.argmax(rir[0])
            noisy_speech_early = signal.fftconvolve(speech, rir[0], mode="full")[pre_delay:][:speech.shape[0]]
            noisy_speech_late = signal.fftconvolve(speech, rir[1], mode="full")[pre_delay:][:speech.shape[0]]
            noisy_speech = noisy_speech_early + noisy_speech_late
        else:
            pre_delay = np.argmax(rir)
            noisy_speech = signal.fftconvolve(speech, rir, mode="full")
            noisy_speech = noisy_speech[pre_delay:]
            noisy_speech = noisy_speech[:speech.shape[0]]
        en_ratio = np.sqrt(np.mean(speech * speech) / (np.mean(noisy_speech * noisy_speech) + EPS))
        # if self.pair_mode == "unit_ir":
        #     noisy_speech = en_ratio * noisy_speech
        if self.rir_decomposed:
            return (noisy_speech, noisy_speech_early, noisy_speech_late), en_ratio
        else:
            return (noisy_speech,), en_ratio

    def mix(self, speech, rir=None, noise=None, snr_db=None, eq_params=None, master_speech=None, master_mixing=0.0):
        y = speech
        if rir is not None:
            y_reverb, en_ratio = self._convolve_rir(y, rir)
        else:
            y_reverb = (y, y, np.zeros_like(y))
            en_ratio = None

        ## TODO: replaced with audiomentations' SevenBandParametricEQ
        if eq_params is not None:
            if self.rir_decomposed:
                y_eq_early = self.eq(y_reverb[1], eq_params)
                if rir is not None:
                    y_eq_late = self.eq(y_reverb[2], eq_params)
                else:
                    y_eq_late = np.zeros_like(y_eq_early)
                y_eq_full = y_eq_early + y_eq_late
                en_ratio = np.sqrt(np.mean(y_reverb[0] * y_reverb[0]) / (np.mean(y_eq_full * y_eq_full) + EPS))
                y_eq_full = en_ratio * y_eq_full
                y_eq = (y_eq_full, y_eq_early, y_eq_late)
            else:
                y_eq_full = self.eq(y_reverb[0], eq_params)
                y_eq = (y_eq_full,)
        else:
            y_eq = y_reverb
            
        if noise is not None:
            if master_speech is None:
                if en_ratio is not None:
                    corr = -20.0 * np.log10(en_ratio + EPS) / 2.0
                else:
                    corr = 0
                # corr = 0
                z, noise_ = self.adder.mix(y_eq[0], noise, snr_db + corr)
            else:
                _, noise_ = self.adder.mix(master_speech, noise, snr_db)
                z = y_eq[0] + noise_ + master_mixing * master_speech
        else:
            z = y_eq[0]
            noise_ = None

        return z, noise_, y_reverb, y_eq

    def random_mix(self, speech, rir=None, noise=None, master_speech=None):
        snr_db = uniform_sampling(self.snr_range[0], self.snr_range[1])

        ## TODO: replaced with audiomentations' SevenBandParametricEQ
        eq_params = None
        if np.random.rand() < self.eq_prob:
            self.eq.random_params()
            eq_params = self.eq.snapshot_params()

        if self.master_mixing_range is not None:
            master_mixing = uniform_sampling(self.master_mixing_range[0], self.master_mixing_range[1])
        else:
            master_mixing = 0.0
        noisy, noise_, y_reverb, y_eq = self.mix(speech, rir=rir, noise=noise, snr_db=snr_db, eq_params=eq_params, master_speech=master_speech, master_mixing=master_mixing)
        return noisy, noise_, y_reverb, y_eq, {"snr_db": snr_db, "eq": eq_params, "master_mixing": master_mixing}

class Simulator(object):
    '''
    Full simulator
    '''
    def __init__(self, sample_rate, size, 
                        speech_speed_range=(0.8, 1.2), 
                        speech_target_level=-25,
                        drr_prob=1.0, drr_scale=(0.5, 2.0),
                        rt60_prob=1.0, rt60_scale=(0.5, 2.0),
                        eq_prob=1.0,
                        eq_centers=[50, 300, 1500], # TODO: Unused
                        eq_mod_range=(-8, 8), 
                        snr_range=(-10, 30), 
                        clip_prob=0.25, clip_range=(0.5, 1.0),
                        level_range=(-30, -20),
                        num_pts_range=(1, 3),
                        pair_mode="target_energy",
                        sndfx_prob=0.15,
                        denoise_prob=0.001,
                        lowpass_prob=0,
                        speech_mix_snr_range=(-5, 5),
                        master_mixing_range=None,
                        drr_limit=(1.0, None),
                        rt60_limit=(None, None),
                        rir_decomposed=False,
                        sndfx_option_probs=SndFX.SNDFX_PROBS,
                        **kwargs
                ):
        '''
        sample_rate: audio sample rate
        size (int, or None): if specified, it will crop or pad the audio to the target size (in number of samples). 
        speech_speed_range (list or tuple of two floats, or None): the range of random speed change for speech if specified. 
        speech_target_level (float, or None): the target speech will be normalized to the volume level if specified.
        drr_prob (float): the probability of Direct-to-Reverb-Ratio augmentation for reverb
        drr_scale (list or tuple of two floats): the range of random scaling for Direct-to-Reverb-Ratio augmentation for reverb
        rt60_prob (float): the probability of RT60 augmentation for reverb
        rt60_scale (list or tuple of two floats): the range of random scaling for RT60 augmentation for reverb
        eq_prob (float): the probability of applying random equalization
        eq_mod_range (list or tuple of two floats): the range of random mod values used in the equalization
        snr_range (list or tuple of two floats): the SNR range for randomly mixing noise
        clip_prob (float): the probability of clipping effect
        clip_range (list or tuple of two floats <= 1.0): the range of random clipping values
        level_range (list or tuple of two floats): the range of random volumes for the simulated input audio
        sndfx_prob (float): the probability of sound-effect-like degradataion in the post effects augmentation 
        denoise_prob (float): the probability of denoiser-effect-like degradataion in the post effects augmentation
        lowpass_prob (float): the probability of bandwidth limitation degradataion in the post effects augmentation
        speech_mix_snr_range: the SNR range for randomly mixing multiple tracks of speech (e.g., useful in multi-spk scenarios)
        sndfx_option_probs: the probs of individual sndfx options
        '''
        
        print("Unused simulator parameters:", kwargs)
        
        self.sample_rate = sample_rate
        self.rir_decomposed = rir_decomposed
        self.pair_mode = pair_mode
        self.speech_aug = SpeechAugmenter(sample_rate, size, 
                                        speed_ramge=speech_speed_range, 
                                        target_level=speech_target_level)
        # self.speech_volume = DynamicVolumeNormalizer(
        #                                       active=True,
        #                                       level_range=[-5, 5], 
        #                                       num_pts_range=num_pts_range)

        if pair_mode=="target_energy" or pair_mode=="unit_peak":
            norm_type = "peak"
        elif pair_mode=="unit_ir":
            norm_type = "energy"
        else:
            print("Unrecognized pair mode:", pair_mode)
            os._exit(1)

        self.speech_mixer = NoiseAdder(sample_rate, snr_range=speech_mix_snr_range, name="speech_mixer")
        self.rir_aug = IRAugmenter(sample_rate, 
                                    drr_prob=drr_prob, drr_scale=drr_scale, 
                                    rt60_prob=rt60_prob, rt60_scale=rt60_scale,
                                    norm_type=norm_type,
                                    drr_limit=drr_limit,
                                    rt60_limit=rt60_limit,)
        self.noise_aug = NoiseAugmenter(sample_rate, size)
        self.mixer = AcousticMixer(sample_rate, 
                                    eq_prob=eq_prob,
                                    eq_centers=eq_centers, # TODO: Unused
                                    eq_mod_range=eq_mod_range, 
                                    snr_range=snr_range,
                                    master_mixing_range=master_mixing_range,
                                    rir_decomposed=rir_decomposed)
        self.post_aug = PostAugmenter(sample_rate, level_range=level_range, 
                            clip_prob=clip_prob, clip_range=clip_range,
                            sndfx_prob=sndfx_prob, denoise_prob=denoise_prob, 
                            lowpass_prob=lowpass_prob,
                            sndfx_option_probs=sndfx_option_probs)

    def simulate(self, speech, rir, noise):
        # Target clean speech
        if not isinstance(speech, list):
            speech = [[speech]]
        if not isinstance(speech[0], list):
            speech = [speech]
        
        master_speech = None
        target_speech = None
        speech_params = []
        for sub_speech in speech:
            sub_target_speech, sub_speech_params = self.speech_aug.augment(sub_speech)
            # Mix speech together
            if target_speech is None:
                target_speech = sub_target_speech
                master_speech = sub_target_speech
                snr_db = 0
            else:
                target_speech, _, snr_db = self.speech_mixer.random_mix(
                                                target_speech, 
                                                sub_target_speech
                                            )
            speech_params.append({
                    "speech_params": sub_speech_params, 
                    "snr_db": snr_db
                    })
            
        # Speech with dynamic volume for input

        # speech_, _, volume_params = self.speech_volume.random_sample(target_speech)
        speech_ = target_speech
        volume_params = None
        
        speech_, _ = fix_clipped(speech_)

        if rir is not None:
            rir_, rir_params = self.rir_aug.augment(rir)
        else:
            rir_ = None
            rir_params = None
        
        if noise is not None:
            noise_, noise_params = self.noise_aug.augment(noise)
        else:
            noise_ = None
            noise_params = None
        
        if self.rir_decomposed and rir_ is not None:
            window = int(2.5/1000.0 * self.sample_rate)
            ir_early, ir_late, (start_idx, _, end_idx) = self.rir_aug._decompose_ir(rir_, window)
            ir_early_ = np.pad(ir_early, (start_idx, 0))
        
            noisy_, noise_, y_reverb, y_eq, mix_params = self.mixer.random_mix(speech_, 
                                                                           rir=(ir_early_, ir_late), 
                                                                           noise=noise_, 
                                                                           master_speech=master_speech)
        else:
            noisy_, noise_, y_reverb, y_eq, mix_params = self.mixer.random_mix(speech_, 
                                                                           rir=rir_, 
                                                                           noise=noise_, 
                                                                           master_speech=master_speech)
        denoise_enabled = active_energy(noisy_) > 0 and active_energy(speech_) > 0
        noisy, post_params = self.post_aug.augment(noisy_, denoise=denoise_enabled)
        noisy, clip_scalar = fix_clipped(noisy)
        
        if noise_ is not None:
            noise_ = clip_scalar * post_params["scalar"] * noise_
        else:
            noise_ = np.zeros_like(speech_)
            
        speech_ = clip_scalar * post_params["scalar"] * speech_

        if self.pair_mode=="target_energy":
            clean = target_speech 
        elif self.pair_mode=="unit_peak" or self.pair_mode=="unit_ir":
            clean = speech_

        # Return different stages of simulation
        
        data_dict =  {"noisy": noisy, 
                "clean": clean,
                "speech_ir": y_reverb[0] * clip_scalar * post_params["scalar"],
                "speech_ir_eq": y_eq[0] * clip_scalar * post_params["scalar"],
                "speech_ir_eq_noise": noisy_,
                "speech": speech_, 
                "noise": noise_, 
                "rir": rir_}
        
        if self.rir_decomposed:
            data_dict["speech_ir_early"] = y_reverb[1] * clip_scalar * post_params["scalar"]
            data_dict["speech_ir_late"] = y_reverb[2] * clip_scalar * post_params["scalar"]
            data_dict["speech_ir_eq_early"] = y_eq[1] * clip_scalar * post_params["scalar"]
            data_dict["speech_ir_eq_late"] = y_eq[2] * clip_scalar * post_params["scalar"]

        return data_dict, {"speech": speech_params, 
                 "volume": volume_params, 
                 "rir": rir_params, 
                 "noise": noise_params,
                 "mix": mix_params, 
                 "post": post_params}

    def reproduce(self, speech, rir, noise, params):
        if not isinstance(speech, list):
            speech = [[speech]]
        if not isinstance(speech[0], list):
            speech = [speech]

        target_speech = None
        speech_params = params["speech"]
        for i, sub_speech in enumerate(speech):
            sub_target_speech = self.speech_aug.reproduce(speech, speech_params[i]["speech_params"])
            # Mix speech together
            if target_speech is None:
                target_speech = sub_target_speech
            else:
                target_speech, _ = self.speech_mixer.mix(
                                            target_speech, 
                                            sub_target_speech,
                                            speech_params[i]["snr_db"]
                                        )

        # speech_, _ = self.speech_volume(target_speech, params["volume"])

        speech_ = target_speech
        speech_, _ = fix_clipped(speech_)
        
        if rir is not None:
            rir_ = self.rir_aug.reproduce(rir, params["rir"])
        else:
            rir_ = None
        if noise is not None:
            noise_ = self.noise_aug.reproduce(noise, params["noise"])
        else:
            noise_ = None
        noisy_, noise_, y_reverb, y_eq = self.mixer.mix(speech_, rir=rir_, noise=noise_, **params["mix"])
        noisy, scalar = self.post_aug.reproduce(noisy_, params["post"])
        noisy, clip_scalar = fix_clipped(noisy) 
        if noise_ is not None:
            noise_ = clip_scalar * scalar * noise_
        else:
            noise_ = np.zeros_like(speech_)
        speech_ = clip_scalar * scalar * speech_

        if self.pair_mode=="target_energy":
            clean = target_speech 
        elif self.pair_mode=="unit_peak" or self.pair_mode=="unit_ir":
            clean = speech_

        # Return different stages of simulation
        return {"noisy": noisy, 
                "clean": clean, 
                "speech": speech_, 
                "noise": noise_, 
                "rir": rir_}