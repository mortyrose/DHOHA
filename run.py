import os
import copy
import pytorch_lightning as pl

from meter.config import ex
from meter.modules import METERTransformerSS

import data
from data import F30kDataModule, MscocoDataModule
import torch

@ex.automain
def main(_config):
    print(_config)
    
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"], workers=True)

    if 'f30k' in _config['exp_name']:
        dm = F30kDataModule(_config)
    else:
        dm = MscocoDataModule(_config)
    vocab = None

    model = METERTransformerSS(_config)

    if (not _config.get("test_only", False)):
        pretrained_ckpt = _config.get("pretrained_ckpt", None) 
        if pretrained_ckpt and os.path.exists(pretrained_ckpt):
            print(f"[Load pretrained] {pretrained_ckpt}")
            ckpt = torch.load(pretrained_ckpt, map_location="cpu") 
            state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
            msg = model.load_state_dict(state_dict, strict=False) 
            print("Missing keys (expected includes dat/dsp):", msg.missing_keys[:30])
            print("Unexpected keys:", msg.unexpected_keys[:30])
        else:
            print("[Load pretrained] skipped (pretrained_ckpt not set or not found)")

    if _config.get("test_only", False):
        ckpt = torch.load(_config['checkpoint'], map_location="cpu")
        model.load_state_dict(ckpt['state_dict'], strict=False)  

    exp_name = f'{_config["exp_name"]}'

    os.makedirs(_config["log_dir"], exist_ok=True)
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=5,
        dirpath = 'runs2/i2t_freeze',
        monitor="best_irtr",
        mode="max",
        save_last=True,
        filename="{epoch:02d}-{best_irtr:.4f}", 
    )
    logger = pl.loggers.TensorBoardLogger(
        _config["log_dir"],
        name=f'{exp_name}_seed{_config["seed"]}_from_{_config["load_path"].split("/")[-1][:-5]}',
    )

    callbacks = [checkpoint_callback]

    num_gpus = (
        _config["num_gpus"]
        if isinstance(_config["num_gpus"], int)
        else len(_config["num_gpus"])
    )

    trainer = pl.Trainer(
        gpus=_config["num_gpus"],
        precision=_config["precision"],
        accelerator="ddp",
        benchmark=True,
        deterministic=True,
        max_epochs=_config["max_epoch"],
        callbacks=callbacks,
        logger=logger,
        #replace_sampler_ddp=False,
        log_every_n_steps=10,
        flush_logs_every_n_steps=10,
        weights_summary="top",
        val_check_interval=_config["val_check_interval"],
    )

    if not _config.get("test_only", False): 
        trainer.fit(model, datamodule=dm)
    else:
        trainer.test(model, datamodule=dm)