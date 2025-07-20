from dataclasses import dataclass
import torch
from transformers import TrainingArguments
from transformers import HfArgumentParser, DataCollatorForLanguageModeling
import os
from tqdm import tqdm
import datasets
import re
from modeling_whisper import WhisperModel, WhisperForConditionalGeneration
from transformers import WhisperTokenizer
from transformers import WhisperConfig
from patching_model_load import patch_model
from patching_whisper import mha2mla_mla_whisper
from safetensors.torch import load_file
from torch import nn

from run_train import (
    ModelArguments,
    DataArguments,
    load_config,
    load_tokenizer_and_model,
    load_whisper,
    get_dataloader,
    prepare_sample
)
from dataset import Whisper_Dataset

def make_linear_from_emb(emb):
    vocab_size, emb_size = emb.weight.shape
    lin_layer = nn.Linear(vocab_size, emb_size, bias=False)
    lin_layer.weight.data = emb.weight.data
    return lin_layer

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
    parser = HfArgumentParser((TrainingArguments, ModelArguments, DataArguments))
    test_args, mha2mla_args, dataset_args = parser.parse_dict(config)
    # assert config["DataArguments"]["DP"] == int(os.environ.get("WORLD_SIZE", 1)), "DP is not equal to WORLD_SIZE"

    gt_path = os.path.join(test_args.output_dir, "gt.txt")
    infer_path = os.path.join(test_args.output_dir, "pred.txt")
    file_gt = open(gt_path, "w")
    file_infer = open(infer_path, "w")
    name = mha2mla_args.model_name_or_path
    config = WhisperConfig.from_pretrained(name)
    tokenizer = WhisperTokenizer.from_pretrained(name, language = "english", task = "transcribe")
    tokenizer.pad_token = tokenizer.eos_token
    mha_model = WhisperForConditionalGeneration.from_pretrained(name, config=config)
    mla_model, q_idx, k_idx = patch_model(mha_model, config, mha2mla_args)
    mha2mla_mla_whisper()
    test_dataset = Whisper_Dataset(dataset_args.test_ann_path, dataset_args.whisper_path)
    model = mla_model
    model.proj_out = make_linear_from_emb(model.model.decoder.embed_tokens)
    data_loader = get_dataloader(test_dataset, test_args, is_train=False, use_distributed=False)

    model.eval()
    model.to("cuda")
    with torch.no_grad():
        for _,batch in enumerate(data_loader):
            audio_path = batch["audio_path"]
            spectrogram = batch["input_features"]
            spectrogram = spectrogram.to(model.device)
            text = batch["text"]
            generate_text = model.generate(spectrogram)
            generate_text_list = tokenizer.batch_decode(generate_text)
            for idx, generate_text in enumerate(generate_text_list):
                file_gt.write(audio_path[idx] + "\t" + text[idx] + "\n")
                file_gt.flush()
                parts = generate_text.lower().split("<|startoftranscript|>")
                target_part = parts[1].split("<|endoftext|>")[0]
                clean_str = re.sub(r'[^\w\s\']', '', target_part)
                result = clean_str.strip()
                #print(result)
                file_infer.write(audio_path[idx] + "\t" + result + "\n")
                file_infer.flush()

if __name__ == "__main__":
    main()
