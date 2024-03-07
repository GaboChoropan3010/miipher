import hydra
from lightning.pytorch.core import datamodule
from omegaconf import DictConfig
from lightning.pytorch import Trainer, seed_everything
import torch
from miipher import lightning_module
from miipher.lightning_module import MiipherLightningModule
from miipher.dataset.datamodule import MiipherDataModule


    
@hydra.main(version_base="1.3", config_name="config", config_path="./configs")
def main(cfg: DictConfig):
    torch.set_float32_matmul_precision("medium")

    text2phone = hydra.utils.instantiate(
                cfg.preprocess.text2phone_model, language='eng-us', is_cuda=False
            )
    input_phonemes = text2phone.infer_sentence('')

    lightning_module = MiipherLightningModule(cfg)
    
    # miipher_path = "https://huggingface.co/spaces/Wataru/Miipher/resolve/main/miipher.ckpt"
    # miipher_path = "/home/hy17/Projects/EXTERNAL/miipher/miipher/tb_runs/wavlm-df/lightning_logs/version_1/checkpoints/checkpoint.ckpt"
    # lightning_module = MiipherLightningModule.load_from_checkpoint(miipher_path,map_location='cpu')
    
    # miipher_path = '/home/hy17/Projects/EXTERNAL/miipher/miipher/tb_runs/wavlm/lightning_logs/version_3/checkpoints/epoch=16-step=18343.ckpt'
    # lightning_module.miipher = MiipherLightningModule(cfg)
    # lightning_module.load_state_dict(torch.load(miipher_path)['state_dict'])
        
    datamodule = MiipherDataModule(cfg, input_phonemes)
    loggers = hydra.utils.instantiate(cfg.train.loggers)
    trainer = hydra.utils.instantiate(cfg.train.trainer, logger=loggers)
    trainer.fit(lightning_module, datamodule)


if __name__ == "__main__":
    
    # from transformers import Wav2Vec2BertModel
    # import torch

    # model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")
    # fake()
    
    main()
