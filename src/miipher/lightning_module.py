import hydra
import torch
from torch import nn
from lightning.pytorch import loggers
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT

from omegaconf import DictConfig
from scipy.io.wavfile import write
from matplotlib import pyplot as plt
from typing import List, Any, Optional

from .model.miipher import Miipher
from lightning_vocoders.models.hifigan.lightning_module import MultiPeriodDiscriminator, MultiScaleDiscriminator
from lightning_vocoders.models.hifigan.xvector_lightning_module import HiFiGANXvectorLightningModule

class FeatureExtractor():
    def __init__(self,cfg) -> None:
        self.speech_ssl_model = hydra.utils.instantiate(cfg.model.ssl_models.model)
        self.speech_ssl_model.eval()
        self.phoneme_model = hydra.utils.instantiate(cfg.model.phoneme_model)
        self.phoneme_model.eval()
        self.xvector_model = hydra.utils.instantiate(cfg.model.xvector_model)
        self.xvector_model.eval()
        self.cfg = cfg

    @torch.inference_mode()
    def __call__(self, inputs):
        wav_16k = inputs["degraded_wav_16k"]
        wav_16k_lens = inputs["degraded_wav_16k_lengths"]
        feats = self.xvector_model.mods.compute_features(wav_16k)
        feats = self.xvector_model.mods.mean_var_norm(feats, wav_16k_lens)
        xvector = self.xvector_model.mods.embedding_model(feats, wav_16k_lens).squeeze(
            1
        )
        
        # def draw_feature(feature, note):
        #     normalized = (feature - torch.min(feature.flatten()))
        #     plt.imshow(torch.log(normalized + 1e-20).cpu().data.numpy().T, origin='lower', aspect='auto', vmin=-5)
        #     plt.colorbar()
        #     plt.savefig(f'/home/hy17/Projects/EXTERNAL/miipher/miipher/output_samples/ssl_feature_{note}.jpg')
        #     plt.clf()

        phone_feature = self.phoneme_model(
            **inputs["phoneme_input_ids"]
        ).last_hidden_state
        
        # print(inputs["clean_ssl_input"].keys()) #dict_keys(['input_values', 'attention_mask']) / input_features
        # signal = **inputs["clean_ssl_input"]['input_values']
        # plt.plot(signal[0].cpu().data.numpy())
        # plt.savefig(f'/home/hy17/Projects/EXTERNAL/miipher/miipher/output_samples/signal.jpg')
        # plt.clf()
        
        # clean_ssl_feature

        
        # original
        if 'clean_ssl_input' in inputs.keys():
            clean_ssl_feature = self.speech_ssl_model(
                **inputs["clean_ssl_input"], output_hidden_states=True
            )
            
            clean_ssl_feature = clean_ssl_feature.hidden_states[
                self.cfg.model.ssl_models.layer
            ]
            # print(clean_ssl_feature.shape) # (bt, 149, 1024)
            # fake()
        else:
            clean_ssl_feature = None

        degraded_ssl_feature = self.speech_ssl_model(
            **inputs["degraded_ssl_input"], output_hidden_states=True
        )
        degraded_ssl_feature = degraded_ssl_feature.hidden_states[
            self.cfg.model.ssl_models.layer
        ]
        
        # draw_feature(clean_ssl_feature[0], 'clean')
        # draw_feature(degraded_ssl_feature[0], 'degraded')
        # fake()
        

        return phone_feature, xvector, degraded_ssl_feature, clean_ssl_feature

    def to(self,device:torch.device):
        self.speech_ssl_model = self.speech_ssl_model.to(device)
        self.phoneme_model = self.phoneme_model.to(device)
        self.xvector_model = self.xvector_model.to(device)

class MiipherLightningModule(LightningModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        # miipher_path = "https://huggingface.co/spaces/Wataru/Miipher/resolve/main/miipher.ckpt"
        # self.miipher = self.load_from_checkpoint(miipher_path,map_location='cpu')
        self.miipher = Miipher(**cfg.model.miipher)
        
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.cfg = cfg
        self.feature_extractor = FeatureExtractor(cfg)
        # GANs
        self.save_hyperparameters()

    def on_fit_start(self):
        self.feature_extractor.to(self.device)

    def forward(self,phone_feature, speaker_feature, degraded_ssl_feature):
        
        cleaned_feature, intermediates = self.miipher.forward(
            phone_feature.clone(), speaker_feature.clone(), degraded_ssl_feature.clone()
        )
        return cleaned_feature, intermediates
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        (
            phone_feature,
            speaker_feature,
            degraded_ssl_feature,
            clean_ssl_feature,
        ) = self.feature_extractor(batch)

        # print(batch.keys()) # dict_keys(['degraded_wav_16k', 'degraded_wav_16k_lengths', 'clean_ssl_input', 'degraded_ssl_input', 'phoneme_input_ids'])

        cleaned_feature, intermediates = self.miipher.forward(
            phone_feature.clone(), speaker_feature.clone(), degraded_ssl_feature.clone()
        )
        # print(len(clean_ssl_feature))
        # print(clean_ssl_feature.shape)
        # fake()
        # wav = self.synthesis(degraded_ssl_feature[0], batch["degraded_wav_16k"][0], batch["degraded_wav_16k_lengths"][0])
        # write(f'/home/hy17/Projects/EXTERNAL/miipher/miipher/output_samples/degraded_feature_sound.wav', 22050, wav.data.numpy())
        # # torchaudio.save(f'/home/hy17/Projects/EXTERNAL/miipher/miipher/output_samples/clean_feature_sound.wav', cleaned_wav, 22050)
        # # self.log_audio(cleaned_wav, f'/home/hy17/Projects/EXTERNAL/miipher/miipher/output_samples/clean', 22050)
        # fake()
        
        with torch.cuda.amp.autocast(enabled=False):
            loss = self.criterion(intermediates.float(), clean_ssl_feature.float(),log=True,stage='train')
        self.log("train/loss", loss, batch_size=phone_feature.size(0),prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT | None:
        (
            phone_feature,
            speaker_feature,
            degraded_ssl_feature,
            clean_ssl_feature,
        ) = self.feature_extractor(batch)
        cleaned_feature, intermediates = self.miipher.forward(
            phone_feature, speaker_feature, degraded_ssl_feature
        )
        
        
        with torch.cuda.amp.autocast(enabled=False):
            loss = self.criterion(intermediates.float(), clean_ssl_feature.float(),log=True,stage='val')
        self.log("val/loss", loss, batch_size=phone_feature.size(0))
        
        if batch_idx < 10 and self.global_rank == 0 and self.local_rank==0:
            cleaned_wav = self.synthesis(cleaned_feature[0], batch["degraded_wav_16k"][0], batch["degraded_wav_16k_lengths"][0])
            self.log_audio(cleaned_wav, f"val/cleaned_wav/{batch_idx}", 22050)
            input_wav = self.synthesis(degraded_ssl_feature[0], batch["degraded_wav_16k"][0], batch["degraded_wav_16k_lengths"][0])
            self.log_audio(input_wav, f"val/input_wav/{batch_idx}", 22050)
            clean_wav = self.synthesis(clean_ssl_feature[0], batch["degraded_wav_16k"][0], batch["degraded_wav_16k_lengths"][0])
            self.log_audio(clean_wav, f"val/target_wav/{batch_idx}", 22050)
        return loss

    def configure_optimizers(self):
        return hydra.utils.instantiate(self.cfg.optimizers, params=self.miipher.parameters())

    def criterion(self, intermediates: List[torch.Tensor], target: torch.Tensor,log=False,stage='train'):
        # def draw_feature(feature, note):
        #     feature += 80
        #     plt.imshow(torch.log(feature).cpu().data.numpy().T, origin='lower', aspect='auto')
        #     plt.colorbar()
        #     plt.savefig(f'/home/hy17/Projects/EXTERNAL/miipher/miipher/output_samples/ssl_feature_{note}.jpg')
        #     plt.clf()
            
        loss = 0
        minimum_length = min(intermediates[0].size(1), target.size(1))
        target = target[:, :minimum_length, :].clone()
        # id = 10
        # draw_feature(target[10], 'target')
        # draw_feature(intermediates[0][10], '0')
        # draw_feature(intermediates[1][10], '1')
        # draw_feature(intermediates[2][10], '2')
        # draw_feature(intermediates[3][10], '3')
        # fake()
        for idx, intermediate in enumerate(intermediates):
            intermediate = intermediate[:, :minimum_length, :].clone()
            loss = loss + self.mae_loss(intermediate, target).clone()
            mae_loss = self.mae_loss(intermediate, target)
            mse_loss = self.mse_loss(intermediate, target)
            spectoral_loss = ( (intermediate - target + 1e-7).norm(p=2, dim=(1, 2)).pow(2) / (target.norm(p=2, dim=(1, 2)).pow(2))).mean()
            loss += mae_loss + mse_loss + spectoral_loss
            if log:
                self.log(f'{stage}/{idx}/mae_loss', mae_loss)
                self.log(f'{stage}/{idx}/mse_loss', mse_loss)
                self.log(f'{stage}/{idx}/spectoral_loss', spectoral_loss)
                

        return loss
    @torch.inference_mode()
    def synthesis(self,features:torch.Tensor,wav16k,wav16k_lens):
        vocoder = HiFiGANXvectorLightningModule.load_from_checkpoint("https://huggingface.co/Wataru/ssl-vocoder/resolve/main/wavlm-large-l8-xvector/wavlm-large-l8-xvector.ckpt",map_location='cpu')
        vocoder.eval()
        xvector_model = hydra.utils.instantiate(vocoder.cfg.data.xvector.model)
        xvector_model.eval()
        xvector = xvector_model.encode_batch(wav16k.unsqueeze(0).cpu()).squeeze(1)
        vocoder = vocoder.float()
        return vocoder.generator_forward({"input_feature": features.unsqueeze(0).cpu().float(), "xvector": xvector.cpu().float()})[0].T

    def log_audio(self, audio, name, sampling_rate):
        for logger in self.loggers:
            match type(logger):
                case loggers.WandbLogger:
                    import wandb

                    wandb.log(
                        {name: wandb.Audio(audio, sample_rate=sampling_rate)},
                        step=self.global_step,
                    )
                case loggers.TensorBoardLogger:
                    logger.experiment.add_audio(
                        name,
                        audio,
                        self.global_step,
                        sampling_rate,
                    )
