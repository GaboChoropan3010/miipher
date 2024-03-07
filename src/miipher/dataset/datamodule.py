from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import webdataset as wds
import torch
import torchaudio
import hydra
from scipy.io.wavfile import write

import glob

from .data import SimEnhanceDataset


class MiipherDataModule(LightningDataModule):
    def __init__(self, cfg, input_phonemes) -> None:
        super().__init__()
        self.speech_ssl_processor = hydra.utils.instantiate(
            cfg.data.speech_ssl_processor.processor
        )
        self.speech_ssl_sr = cfg.data.speech_ssl_processor.sr
        self.phoneme_tokenizer = hydra.utils.instantiate(cfg.data.phoneme_tokenizer)
        # self.text2phone_dict = dict()


        # self.text2phone = hydra.utils.instantiate(
        #         self.cfg.preprocess.text2phone_model, language='eng-us'
        # )
        self.input_phonemes = input_phonemes
        self.cfg = cfg

    def get_dataset(self, task = 'train'): # **** TODO(HAICI) ***

        assert task == 'train' or task == 'val'

        if task == 'train':
            # speech_list = prepare_files([
            #     f"/sensei-fs/users/haiciy/DataProc/studiosound2-asim/datasets/speech/daps_speech_{task}.txt", # daps
            #     f"/sensei-fs/users/haiciy/DataProc/studiosound2-asim/datasets/speech/vctk_speech_{task}.txt", # vctk
            #     "/sensei-fs/users/haiciy/DataProc/studiosound2-asim/datasets/speech/libritts_filt_speech.txt", # libritts
            #     "/sensei-fs/users/haiciy/DataProc/studiosound2-asim/datasets/speech/podcasts_speech.txt" # podcast
            # ])
            # print(len(speech_list))
            
            # tr_path = '/N/project/SAIGE_shared/librittsR/LibriTTS_R/train-clean-360' 
            tr_path1 = '/data/hy17/librittsR/LibriTTS_R/train-clean-360/*/*/*.wav'
            tr_path2 = '/data/hy17/daps/train/*/*.wav'
            speech_list = glob.glob(tr_path1)+ \
                          glob.glob(tr_path2)

        else:
            tr_path1 = '/data/hy17/librittsR/LibriTTS_R/dev-clean/*/*/*.wav'
            tr_path2 = '/data/hy17/daps/valid/*.wav'
            speech_list = glob.glob(tr_path1) + \
                          glob.glob(tr_path2)
        
        # rir_list = prepare_files([
        #     f"/sensei-fs/users/haiciy/DataProc/studiosound2-asim/datasets/rir/mit_rir_{task}.txt",
        #     "/sensei-fs/users/haiciy/DataProc/studiosound2-asim/datasets/rir/echothief_rir.txt",
        #     "/sensei-fs/users/haiciy/DataProc/studiosound2-asim/datasets/rir/openslr_rir.txt"
        # ]) 
        
        rir_path = '/data/hy17/rirs/*/*/*/*/*.wav'
        rir_list = glob.glob(rir_path)
        
        # noise_list = glob.glob('/N/project/SAIGE_shared/noise/WHAM/high_res_wham/audio/*.wav')
        # noise_list = glob.glob('/data/hy17/noise/TAU_Urban_Audio/audio/*.wav')
        
        noise_list = glob.glob('/data/hy17/noise/DNS/*/*/*/*.wav') + \
      				 glob.glob('/data/hy17/noise/ISOLATED_URBAN_SOUND/*/*/*.wav') + \
					 glob.glob('/data/hy17/noise/SFS-static/*/*/*/*.wav') + \
					 glob.glob('/data/hy17/noise/TAU_Urban_Audio/*/*.wav') + \
					 glob.glob('/data/hy17/noise/WHAM/high_res_wham/audio/*.wav')
        
        
        # elif task == 'val':
        #     speech_list = prepare_files([
        #         f"/sensei-fs/users/haiciy/DataProc/studiosound2-asim/datasets/speech/daps_speech_{task}.txt", # daps
        #         f"/sensei-fs/users/haiciy/DataProc/studiosound2-asim/datasets/speech/vctk_speech_{task}.txt", # vctk
        #     ])
        #     rir_list = prepare_files([f"/sensei-fs/users/haiciy/DataProc/studiosound2-asim/datasets/rir/mit_rir_{task}.txt"]) 
        
        # noise_list = prepare_files([
        #         "/sensei-fs/users/haiciy/DataProc/studiosound2-asim/datasets/noise/basic_noise.txt",
        #         "/sensei-fs/users/haiciy/DataProc/studiosound2-asim/datasets/noise/esc50_noise.txt",
        #         "/sensei-fs/users/haiciy/DataProc/studiosound2-asim/datasets/noise/dns_noise.txt",
        #         "/sensei-fs/users/haiciy/DataProc/studiosound2-asim/datasets/noise/isolated_urban_noise.txt",
        #         "/sensei-fs/users/haiciy/DataProc/studiosound2-asim/datasets/noise/wham_noise.txt",
        #         "/sensei-fs/users/haiciy/DataProc/studiosound2-asim/datasets/noise/sfs_noise.txt",
        #         ])
        
        
        if task == 'train':
            dataset = SimEnhanceDataset(
                            name='sim', 
                            sample_rate=24000,
                            size=24000 * 3 , # 3s
                            speech_list=speech_list, 
                            noise_list=noise_list, 
                            reverb_list=rir_list, 
                            sim_params={
                                        "speech_speed_range": (0.9, 1.1), 
                                        "speech_target_level": -25,
                                        "drr_prob": 1.0,  # 0
                                        "drr_scale": (0.5, 2.0),
                                        "rt60_prob": 1.0,  # 0
                                        "rt60_scale": (0.5, 2.0),
                                        "eq_prob": 1.0, 
                                        "eq_mod_range": (-10, 10), # Jiaqi - enable greater variation of simulation
                                        "snr_range": (-10, 30), # (10, 10)
                                        "clip_prob": 0.2, 
                                        "clip_range": (0.5, 1.0),
                                        "level_range": (-30, -20),
                                        "sndfx_prob": 0.3,         # Jiaqi - enable greater variation of simulation 
                                        "denoise_prob": 0,         # Disable post effects that are computationally expensive
                                        "noise_prob": 0.999,              
                                        "reverb_prob": 0.999,             
                                        "speech_prob": 1.0,
                                        "sndfx_option_probs": {    # Jiaqi - enable greater variation of simulation
                                        "overdrive": 0.2,
                                        "phaser": 0.2,
                                        "compand": 0.8,
                                        },
                                        "lowpass_prob": 0.8,        # Jiaqi - enable greater variation of simulation
                            },
                            # count_reverb = (task == "val"),
                            count_reverb = False
                        )
        else:
            dataset = SimEnhanceDataset(
                            name='sim', 
                            sample_rate=24000,
                            size=24000 * 3 , # 3s
                            speech_list=speech_list, 
                            noise_list=noise_list, 
                            reverb_list=rir_list, 
                            sim_params={
                                        "speech_speed_range": (0.9, 1.1), 
                                        "speech_target_level": -25,
                                        "drr_prob": 0, 
                                        "drr_scale": (0.5, 2.0),
                                        "rt60_prob": 0, 
                                        "rt60_scale": (0.5, 2.0),
                                        "eq_prob": 1.0, 
                                        "eq_mod_range": (-10, 10), # Jiaqi - enable greater variation of simulation
                                        "snr_range": (10, 10), 
                                        "clip_prob": 0.2, 
                                        "clip_range": (0.5, 1.0),
                                        "level_range": (-30, -20),
                                        "sndfx_prob": 0.3,         # Jiaqi - enable greater variation of simulation 
                                        "denoise_prob": 0,         # Disable post effects that are computationally expensive
                                        "noise_prob": 0.999,              
                                        "reverb_prob": 0.999,             
                                        "speech_prob": 1.0,
                                        "sndfx_option_probs": {    # Jiaqi - enable greater variation of simulation
                                        "overdrive": 0.2,
                                        "phaser": 0.2,
                                        "compand": 0.8,
                                        },
                                        "lowpass_prob": 0.8,        # Jiaqi - enable greater variation of simulation
                            },
                            # count_reverb = (task == "val"),
                            count_reverb = False
                        )


        return dataset

    def setup(self, stage: str):
        # self.train_dataset = (
        #     wds.WebDataset(
        #         self.cfg.data.train_dataset_path,
        #         resampled=True,
        #         nodesplitter=wds.split_by_node,
        #     )
        #     .shuffle(1000)
        #     .decode(wds.torch_audio)
        #     # .decode(self.decode_phoneme_input)
        #     .repeat(2)
        #     .with_length(20000 * self.cfg.data.train_batch_size)
        # )
        # self.val_dataset = (
        #     wds.WebDataset(
        #         self.cfg.data.val_dataset_path, nodesplitter=wds.split_by_node
        #     )
        #     .decode(wds.torch_audio)
        #     # .decode(self.decode_phoneme_input)
        #     .repeat(2)
        #     .with_length(3000 * 4 // self.cfg.data.val_batch_size)
        # )
        
        self.train_dataset = self.get_dataset(task='train')
        self.val_dataset = self.get_dataset(task='val')

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.data.train_batch_size,
            collate_fn=self.collate_fn,
            num_workers=8,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.data.val_batch_size,
            collate_fn=self.collate_fn,
            num_workers=8,
        )

    @torch.no_grad()
    def collate_fn(self, batch):
        output = dict()
        degraded_wav_16ks = []
        clean_wav_16ks = []

        for sample in batch:

            clean_wav = torch.tensor(sample["clean"])
            
            sr = 24000 # **** TODO(HAICI) ***
            clean_wav_16ks.append(
                torchaudio.functional.resample(clean_wav, sr, new_freq=16000).squeeze()[:16000*20]
            )
            
            degraded_wav = torch.tensor(sample["noisy"])
            degraded_wav_16ks.append(
                torchaudio.functional.resample(
                    degraded_wav, sr, new_freq=16000
                ).squeeze()[:16000*20]
            )
        output["degraded_wav_16k"] = pad_sequence(degraded_wav_16ks, batch_first=True)
        output["degraded_wav_16k_lengths"] = torch.tensor(
            [degraded_wav_16k.size(0) for degraded_wav_16k in degraded_wav_16ks]
        )
        output["clean_ssl_input"] = self.speech_ssl_processor(
            [x.numpy() for x in clean_wav_16ks],
            return_tensors="pt",
            sampling_rate=16000,
            padding=True,
        )
        output["degraded_ssl_input"] = self.speech_ssl_processor(
            [x.numpy() for x in degraded_wav_16ks],
            return_tensors="pt",
            sampling_rate=16000,
            padding=True,
        )
        
        
        # input_phonems = self.get_phonemes_input_ids(
        #     word_segmented_text='', lang_code="eng-us"
        # )
        # print(input_phot)
        
        # output["phoneme_input_ids"] = self.phoneme_tokenizer(
        #     [b["phoneme.txt"] for b in batch], return_tensors="pt", padding=True
        # )
        
        # space_phonem_id = {'input_ids': torch.tensor([[  0, 171,   8,   2]]), 'attention_mask': torch.tensor([[1, 1, 1, 1]])}
        output["phoneme_input_ids"] = self.phoneme_tokenizer(
            [self.input_phonemes for b in batch], return_tensors="pt", padding=True
        )
        
        return output
        
        
        
    @torch.inference_mode()
    def get_phonemes_input_ids(self, word_segmented_text, lang_code):
        
        input_phonemes = self.text2phone.infer_sentence(
            word_segmented_text
        )
        # input_ids = self.phoneme_tokenizer(input_phonemes, return_tensors="pt")
        return input_phonemes
    
        # # output["phoneme_input_ids"] = self.phoneme_tokenizer(
        # #     ['' for b in batch], return_tensors="pt", padding=True
        # # )
        # return output
