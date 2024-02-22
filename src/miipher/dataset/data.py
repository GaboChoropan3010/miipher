import torch
from torch.utils import data
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from torch.utils.data.dataloader import default_collate

import time
import numpy as np
import math
import pickle 
import os    
import json
import copy
import matplotlib.pyplot as plt

from .asim import AudioLoader, Simulator

def my_collate(batch):
    "Puts each data field into a tensor with outer dimension batch size"
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)

class BaseDataset(data.Dataset):
    """Dataset class for the simulated data."""

    def __init__(self, name, 
                       sample_rate,
                       size):
        super().__init__()
        self.name = name
        self.sample_rate = sample_rate
        self.size = size

        self.loader = AudioLoader(sample_rate)

    def _load_audio(self, filename, channel=None, size=None):
        'Load audio in pytorch'

        waveform = self.loader(filename, size)
        if channel is None or channel >= waveform.size(dim=0):
            channel=torch.randint(waveform.size(0), (1,))[0]

        waveform = waveform[channel, ...].unsqueeze(0)
        return waveform

class SimEnhanceDataset(BaseDataset):
    """Dataset class for the simulated data."""

    def __init__(self, name, 
                       sample_rate,
                       size,
                       speech_list, 
                       noise_list=None, 
                       reverb_list=None, 
                       sim_params = {
                            "speech_speed_range": (0.9, 1.1), 
                            "speech_target_level": -25,
                            "drr_prob": 1.0, 
                            "drr_scale": (0.5, 2.0),
                            "rt60_prob": 1.0, 
                            "rt60_scale": (0.5, 2.0),
                            "eq_mod_range": (-8, 8), 
                            "snr_range": (-10, 30), 
                            "clip_prob":0.5, 
                            "clip_range":(0.5, 1.0),
                            "level_range": (-30, -20),
                            "noise_prob": 1.0,
                            "reverb_prob": 1.0,
                            "speech_prob": 1.0,
                       },
                       count_reverb = True,
                    ):
        super().__init__(name, sample_rate, size)

        self.speech_list = speech_list
        
        if noise_list is not None:
            self.noise_list = noise_list
        else:
            self.noise_list = []
            
        if reverb_list is not None:
            self.reverb_list = reverb_list
        else:
            self.reverb_list = []

        self.simulator = Simulator(sample_rate, size, **sim_params)
        self.speech_prob = sim_params.get("speech_prob", 1.0)
        self.noise_prob = sim_params.get("noise_prob", 1.0)
        self.reverb_prob = sim_params.get("reverb_prob", 1.0)
        
        self.count_reverb = count_reverb

    def get_index(self, idx):
        sp_idx = idx % len(self.speech_list)
        ir_idx = idx // len(self.speech_list)

        return sp_idx, ir_idx

    def __len__(self):
        'Denotes the total number of samples'
        if self.count_reverb:
            num_tuples = len(self.speech_list) * max(1, len(self.reverb_list))
        else:
            num_tuples = len(self.speech_list)

        return num_tuples
        # return 100
    
    def get_speech_audio(self, speech_filename=None, idx=0):
        if speech_filename is None:
            speech_filename = self.speech_list[int(idx)]

        if isinstance(speech_filename, (list, np.ndarray)):
            speech_filename = np.random.choice(speech_filename)
        
        speech_audio = self._load_audio(speech_filename, size=2 * self.size)
        return speech_audio, speech_filename

    def get_reverb_audio(self, reverb_filename=None, idx=0):
        if reverb_filename is None:
            reverb_filename = self.reverb_list[int(idx)]
        
        reverb_audio = self._load_audio(reverb_filename)[0] 
        return reverb_audio, reverb_filename

    def get_noise_audio(self, noise_filename=None, idx=None):
        # print(self.noise_list)
        # for k in self.noise_list:
        #     if 'ext_subnoise' in k:
        #         print(k)
        # print('end')
        if noise_filename is None:
            if idx is None:
                idx = np.random.randint(0, len(self.noise_list))
            noise_filename = self.noise_list[idx]
            
        noise = self._load_audio(noise_filename, size = 2 * self.size)
        return noise, noise_filename

    def _generate_pair(self, speech, reverb=None, noise=None, filename=None):
        data, sim_params = self.simulator.simulate(speech, reverb, noise)
        if np.sum(data["noisy"] ** 2) > 1e-8:
            data_book = {
                    "noisy": data["noisy"][None, ...].astype(np.float32), 
                    "clean": data["clean"][None, ...].astype(np.float32),
                    "speech": data["speech"][None, ...].astype(np.float32),
                    "filename": filename
                    }
                    
            if data["noise"] is not None:
                data_book["noise"] = data["noise"][None, ...].astype(np.float32)
            else:
                data_book["noise"] = np.zeros_like(data["noisy"])[None, ...].astype(np.float32)
            
            rir = np.zeros((2 * self.sample_rate, ))
            if data["rir"] is not None:
                rir[:len(data["rir"])] = data["rir"][:2 * self.sample_rate]
            data_book["rir"] = rir[None, ...].astype(np.float32)
            
            return data_book
        else:
            return None
    
    def __getitem__(self, index):
        'Generates one sample of data'
        # try:
        if True:
            # Select samples for speech and reverb
            # speech_idx, reverb_idx = self.get_index(index)
            
            speech_audio = []
            speech_filename = []
            total_len = 0
            
            # Purposefully allow small chance of no speech
            if np.random.rand() < self.speech_prob:
                # Keep accumulating speech until we have enough duration of speech (having some extra for augmentation)
                target_len = 2 * self.size if np.random.rand() < 0.5 else 1
                
                while total_len < target_len:
                    speech_idx = np.random.randint(len(self.speech_list))
                    speech_audio_i, speech_filename_i = self.get_speech_audio(idx=speech_idx)
                    speech_audio_i = speech_audio_i[0].numpy()
                    speech_audio.append(speech_audio_i)
                    speech_filename.append(speech_filename_i)
                    total_len += len(speech_audio_i)
            else:
                speech_audio.append(np.zeros((2 * self.size,)))
                speech_filename.append("N/A")

            # Purposefully allow small chance of no reverb
            if len(self.reverb_list) > 0 and np.random.rand() < self.reverb_prob:
                reverb_idx = np.random.randint(len(self.reverb_list))
                reverb_audio, reverb_filename = self.get_reverb_audio(idx=reverb_idx)
                
                if reverb_audio.size(-1) == 0 or torch.mean(reverb_audio ** 2.0) == 0:
                    print(reverb_filename, "is broken")
                    reverb_audio = None
                    reverb_filename = None
                else:
                    reverb_audio = reverb_audio.numpy()
            else:
                reverb_audio = None
                reverb_filename = None

            # Purposefully allow small chance of no noise
            if len(self.noise_list) > 0 and np.random.rand() < self.noise_prob:
                noise_audio = []
                # Randomly decide the number of noises to mix in
                sel_num = np.random.randint(1, 4)
                # Randomly draw noise, redraw if broken file
                while len(noise_audio) < sel_num:
                    noise_audio_, noise_filename = self.get_noise_audio()
                    if noise_audio_.size(-1) == 0 or torch.mean(noise_audio_ ** 2.0) == 0:
                        continue
                    else:
                        noise_audio.append(noise_audio_[0].numpy())               
            else:
                noise_audio = None
            
            return self._generate_pair(speech_audio, 
                                        reverb=reverb_audio, 
                                        noise=noise_audio,
                                        filename=" || ".join(speech_filename))

        # except Exception as e:
            # print("Exception:", repr(e))
        
    def print(self):
        msg =  "========== Sim Enhance Dataset: {} ==========\n".format(self.name)
        msg += "Num speeches: {}\n".format(len(self.speech_list))
        msg += "Num reverbs: {}\n".format(len(self.reverb_list))
        msg += "Num noises: {}\n".format(len(self.noise_list))
        msg += "Num elements: {}\n".format(len(self))
        msg += "================================\n"
        print(msg)
        
        
class RealEnhanceDataset(BaseDataset):
    """Dataset class for the simulated data."""

    def __init__(self, name, 
                       sample_rate,
                       noisy_list,
                       speech_list=None,
                       size=None):
        super().__init__(name, sample_rate, size=size)

        self.noisy_list = noisy_list
        if speech_list is not None:
            self.speech_list = speech_list
        else:
            self.speech_list = None
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.noisy_list)

    def get_index(self, idx):
        return idx % len(self.noisy_list)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        idx = self.get_index(index)

        noisy_fn = self.noisy_list[idx]        
        noisy_audio = self._load_audio(noisy_fn, size=None)
        noisy_audio = noisy_audio[0].numpy()
        noisy_audio = noisy_audio[:self.size]
         
        if self.speech_list is not None:
            speech_fn = self.speech_list[idx]
            speech_audio = self._load_audio(speech_fn, channel=0, size=None)
            speech_audio = speech_audio[0].numpy()
            speech_audio = speech_audio[:self.size]
            return {"noisy": noisy_audio[None, ...], "clean": speech_audio[None, ...], "filename": noisy_fn}
        else:
            return {"noisy": noisy_audio[None, ...], "filename": noisy_fn}
    
    def print(self):
        msg =  "========== Single Enhance Dataset: {} ==========\n".format(self.name)
        msg += "Num elements: {}\n".format(len(self))
        msg += "================================\n"
        print(msg)