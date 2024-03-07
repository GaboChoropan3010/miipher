import os
import glob
import torch
import hydra
import tempfile
import torchaudio
from argparse import ArgumentParser

import gradio as gr

from tqdm import tqdm

from torch.utils import data
from torch.utils.data import DataLoader

from omegaconf import DictConfig
from miipher.lightning_module import MiipherLightningModule
from miipher.dataset.preprocess_for_infer import PreprocessForInfer
from lightning_vocoders.models.hifigan.xvector_lightning_module import HiFiGANXvectorLightningModule

from matplotlib import pyplot as plt

    

class EvalDataset(data.Dataset):
    
    def __init__(self, path):
        self.wav_list = glob.glob(path)

    def __getitem__(self, index):
        sample_path = self.wav_list[index]
        wav,sr =torchaudio.load(wav_path)
        
        return sample_path, wav, sr
    def __len__(self):
        return len(self.wav_list)
        

@torch.inference_mode()
def main(wav_path,transcript,lang_code, note = '', re=False):
    print(wav_path)
    # print(load_model())
    # miipher, preprocessor, xvector_model, vocoder = load_model()
    wav,sr =torchaudio.load(wav_path)
    wav = wav[0].unsqueeze(0)#.to('cuda')
    batch = preprocessor.process(
        'test',
        (torch.tensor(wav),sr),
        word_segmented_text=transcript,
        lang_code=lang_code
    )
    
    miipher.feature_extractor(batch)
    (
        phone_feature,
        speaker_feature,
        degraded_ssl_feature,
        _,
    ) = miipher.feature_extractor(batch)
    cleaned_ssl_feature, _ = miipher(phone_feature,speaker_feature,degraded_ssl_feature)
    vocoder_xvector = xvector_model.encode_batch(batch['degraded_wav_16k'].view(1,-1).cpu()).squeeze(1)
    vocoder_xvector = xvector_model.encode_batch(batch['degraded_wav_16k'].view(1,-1)).squeeze(1)
    
    # vocoder_xvector = torch.zeros(vocoder_xvector.shape).to(vocoder_xvector.device)
    # print(vocoder_xvector)#torch.Size([1, 512])
    cleaned_wav = vocoder.generator_forward({"input_feature": cleaned_ssl_feature, "xvector": vocoder_xvector})[0].T
    # cleaned_wav = vocoder.generator_forward({"input_feature": cleaned_ssl_feature})[0].T   
    
    if not re:
        torchaudio.save(f'/home/hy17/Projects/EXTERNAL/miipher/miipher/output_samples/{note}.wav', cleaned_wav.view(1,-1), sample_rate=22050,format='wav')
    else:
        return cleaned_wav


if __name__ == '__main__':
    
    parser = ArgumentParser()
   
    parser.add_argument("--ckpt", type=str, required=True, default='epoch=12-step=14157.ckpt')
    parser.add_argument("--wav_path", required=True)

    args = parser.parse_args()
   
    miipher_path = args.ckpt
    print(miipher_path)
    fake()
    miipher = MiipherLightningModule.load_from_checkpoint(miipher_path,map_location='cpu')

    # ---- Load vocoder ----
    vocoder = HiFiGANXvectorLightningModule.load_from_checkpoint("https://huggingface.co/spaces/Wataru/Miipher/resolve/main/vocoder_finetuned.ckpt",map_location='cpu')


    xvector_model = hydra.utils.instantiate(vocoder.cfg.data.xvector.model)
    xvector_model = xvector_model.to('cpu')
    preprocessor = PreprocessForInfer(miipher.cfg)

    # wav_path = '/home/hy17/Projects/EXTERNAL/miipher/miipher/xxxx/DEMO_samples/voicefixer/*.wav' # DEMO_samples;
    wav_path = args.wav_path

    # note = 'w2v-bert-new'
    if not '*' in wav_path:
        main(wav_path = wav_path, transcript = '', lang_code = "eng-us", note=note)
    else:
        for path in tqdm(glob.glob(wav_path)):
            # print(path)
            name = path.split('/')
            cleaned_wav = main(wav_path = path, transcript = '', lang_code = "eng-us", re=True)
            # torchaudio.save(xxxx, cleaned_wav.view(1,-1), sample_rate=22050,format='wav')

