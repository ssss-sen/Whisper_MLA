from dataclasses import dataclass, asdict
import torch
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
from transformers import LlamaConfig, LlamaForCausalLM, AutoModelForCausalLM
from torch.utils.data import DataLoader
import datasets
from transformers.utils import is_datasets_available
from transformers.trainer_utils import seed_worker
from transformers import HfArgumentParser,DataCollatorForLanguageModeling
import os
import time
from typing import Dict, List
import numpy as np
import yaml
from dataclasses import dataclass, field
from typing import List, Optional

from lr_scheduler import load_scheduler as load_scheduler4constant_with_warmup_decay
from modeling_whisper import WhisperModel, WhisperForConditionalGeneration
from transformers import WhisperTokenizer
from transformers import WhisperConfig
from dataset import Whisper_Dataset
from patching_model_load import patch_model
from patching_whisper import mha2mla_mla_whisper
from safetensors.torch import load_file
from transformers import WhisperFeatureExtractor

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained model"},
    )
    partial_rope_version: str = field(
        default="high",
        metadata={
            "help": "RoPE version to use for partial RoPE in MLA. Options: 'high', 'low', 'uniform', '2-norm'"
        },
    )
    rope_dim_for_mla: int = field(
        default=0, metadata={"help": "Number of rope dimensions per head"}
    )
    uniform_start_point: int = field(
        default=0,
        metadata={
            "help": "Starting point (only used when partial_rope_version='uniform')"
        },
    )
    qk_tensor_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to pre-computed QK tensor file, e.g., 'utils/qk_tensor_135M.pth'"
        },
    )
    svd_init_method: str = field(
        default="none",
        metadata={
            "help": "Method for SVD initialization. Options: 'split' or 'joint' or 'none'"
        },
    )
    low_rank: int = field(
        default=8, metadata={"help": "Rank for low-rank approximation in MLA"}
    )
    is_baseline: bool = field(
        default=False,
        metadata={"help": "if the finetuning is the baseline"},
    )
    is_gqa2mha2mla: bool = field(
        default=False,
        metadata={"help": "if the finetuning is GQA2MHA2MLA"},
    )
    is_mla_from_scratch: bool = field(
        default=False,
        metadata={"help": "if the finetuning is from scratch"}
    )
    tokenizer_name_or_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained tokenizer"},
    )

@dataclass
class DataArguments:
    train_ann_path: str = None
    valid_ann_path: str = None
    test_ann_path: str = None
    whisper_path: str = None
    sequence_length: int = 2048

@dataclass
class MHA2MLATrainingArguments(TrainingArguments):
    use_constant_with_warmup_decay_scheduler: bool = field(
        default=False,
        metadata={"help": "Whether to use constant with warmup decay scheduler"},
    )
    is_freeze_non_attn: bool = field(
        default=False,
        metadata={"help": "if the finetuning is freeze non attention parameters"},
    )


def load_tokenizer_and_model(model_args: ModelArguments,is_mla:bool=False,mla_kwargs:Dict=None):
    """Load tokenizer and model from configuration."""
    assert (
        model_args.model_name_or_path is not None
    ), "Must provide the path to the model"
    if model_args.tokenizer_name_or_path is None:
        model_args.tokenizer_name_or_path = model_args.model_name_or_path
    config = LlamaConfig.from_pretrained(model_args.model_name_or_path)
    if is_mla:
        cfg_RoPE = mla_kwargs.get("RoPE")
        cfg_SVD = mla_kwargs.get("SVD")
        config.RoPE = cfg_RoPE
        config.SVD = cfg_SVD
    model = LlamaForCausalLM.from_pretrained(model_args.model_name_or_path,config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer
    
def load_whisper(model_args: ModelArguments,is_mla:bool=False,mla_kwargs:Dict=None):
    """Load tokenizer and model from configuration."""
    assert (
        model_args.model_name_or_path is not None
    ), "Must provide the path to the model"
    if model_args.tokenizer_name_or_path is None:
        model_args.tokenizer_name_or_path = model_args.model_name_or_path
    config = WhisperConfig.from_pretrained(model_args.model_name_or_path)
    if is_mla:
        cfg_RoPE = mla_kwargs.get("RoPE")
        cfg_SVD = mla_kwargs.get("SVD")
        config.RoPE = cfg_RoPE
        config.SVD = cfg_SVD
    model = WhisperForConditionalGeneration.from_pretrained(model_args.model_name_or_path,config=config)
    tokenizer = WhisperTokenizer.from_pretrained(model_args.tokenizer_name_or_path, language = "english", task = "transcribe")
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def load_optimizer_scheduler(model, training_args, model_args):
    """Load optimizer and scheduler from configuration."""
    optimizer_name = training_args.optim
    if "adam" in optimizer_name:
        optimizer = torch.optim.AdamW(
            params=model.parameters(),
            lr=training_args.learning_rate,
            betas=(
                training_args.adam_beta1,
                training_args.adam_beta2,
            ),
            eps=training_args.adam_epsilon,
            weight_decay=training_args.weight_decay,
            fused=bool(training_args.optim=="adamw_torch_fused"),
        )
    else:
        raise ValueError(
            f"Unknown optimizer factory {optimizer_name}"
        )
    if training_args.use_constant_with_warmup_decay_scheduler:
        lr_scheduler = load_scheduler4constant_with_warmup_decay(
            optimizer, training_args
        )
    else:
        from transformers import get_scheduler
        lr_scheduler = get_scheduler(
            training_args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=training_args.warmup_steps,
            num_training_steps=training_args.max_steps,
        )
    return optimizer, lr_scheduler
    
def get_dataloader(dataset, config):
    loader = DataLoader(
        dataset,
        batch_size=config.per_device_train_batch_size,
        num_workers=1,
        pin_memory=True,
        sampler=None,
        shuffle=True,
        collate_fn=dataset.collater,
        drop_last=True,
    )
    loader = IterLoader(loader, use_distributed=False)

    return loader
    

def apply_to_sample(f, sample):
    if len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        else:
            return x

    return _apply(sample)


def move_to_cuda(sample):
    def _move_to_cuda(tensor):
        return tensor.cuda()

    return apply_to_sample(_move_to_cuda, sample)


def prepare_sample(samples, cuda_enabled=True):
    if cuda_enabled:
        samples = move_to_cuda(samples)

    # TODO fp16 support

    return samples

def expand_embeddings(new_emb, orig_emb, orig_size):
    """扩展嵌入层并保留原始参数"""
    # 复制原始参数
    new_emb.weight.data[:orig_size] = orig_emb.weight.data[:orig_size]
    
    # 初始化新增部分
    global_mean = orig_emb.weight.data.mean().item()
    global_std = orig_emb.weight.data.std().item()
   
    new_emb.weight.data[orig_size:].normal_(mean=global_mean, std=global_std)

    return new_emb
 
        
class IterLoader:
    """
    A wrapper to convert DataLoader as an infinite iterator.

    Modified from:
        https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/iter_based_runner.py
    """

    def __init__(self, dataloader: DataLoader, use_distributed: bool = False):
        self._dataloader = dataloader
        self.iter_loader = iter(self._dataloader)
        self._use_distributed = use_distributed
        self._epoch = 0

    @property
    def epoch(self) -> int:
        return self._epoch

    def __next__(self):
        try:
            data = next(self.iter_loader)
        except StopIteration:
            self._epoch += 1
            if hasattr(self._dataloader.sampler, "set_epoch") and self._use_distributed:
                self._dataloader.sampler.set_epoch(self._epoch)
            time.sleep(2)  # Prevent possible deadlock during epoch transition
            self.iter_loader = iter(self._dataloader)
            data = next(self.iter_loader)

        return data

    def __iter__(self):
        return self

    def __len__(self):
        return len(self._dataloader)
    

def main():
    import argparse

    cmd_parser = argparse.ArgumentParser()
    cmd_parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )
   
    args = cmd_parser.parse_args()
    config = load_config(args.config_file)
    parser = HfArgumentParser((MHA2MLATrainingArguments, ModelArguments,DataArguments))
    training_args, model_args, dataset_args = parser.parse_dict(config)

    # Monkey Pacth
    name = model_args.model_name_or_path
    config = WhisperConfig.from_pretrained(model_args.model_name_or_path)
    tokenizer = WhisperTokenizer.from_pretrained(model_args.tokenizer_name_or_path, language = "english", task = "transcribe")
    processor = WhisperFeatureExtractor.from_pretrained(model_args.model_name_or_path, language="english", task="transcribe")
    tokenizer.pad_token = tokenizer.eos_token
    resume_from_checkpoint = training_args.resume_from_checkpoint
    mha_model = WhisperForConditionalGeneration.from_pretrained(name, config=config)
    print(mha_model)
    if not model_args.is_baseline or model_args.is_mla_from_scratch:
        mla_model, q_idx, k_idx = patch_model(mha_model, config, model_args)
        # TODO: move instance selection to patch_func.py
    if isinstance(mha_model, WhisperForConditionalGeneration):
        mha2mla_mla_whisper()
    model = mha_model if model_args.is_baseline else mla_model
    model.config.mha2mla = asdict(model_args)
    print(model)
    
    if training_args.bf16:
        model = model.to(dtype=torch.bfloat16)
    elif training_args.fp16:
        model = model.to(dtype=torch.float16)

    train_dataset = Whisper_Dataset(dataset_args.train_ann_path, dataset_args.whisper_path)
    resume_from_checkpoint = training_args.resume_from_checkpoint
    optimizer, lr_scheduler = load_optimizer_scheduler(model, training_args, model_args)
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        optimizers=(optimizer, lr_scheduler),
        data_collator=train_dataset.collater,
    )
    # train
    trainer.train(resume_from_checkpoint)
    trainer.save_model(output_dir=training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
