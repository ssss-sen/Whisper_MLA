from dataclasses import dataclass
import torch
from transformers import TrainingArguments
from transformers import HfArgumentParser, DataCollatorForLanguageModeling
import os
from tqdm import tqdm
import datasets

from run_train import (
    ModelArguments,
    DataArguments,
    MHA2MLATrainingArguments,
    load_config,
    load_tokenizer_and_model,
    load_whisper,
    get_dataloader,
    prepare_sample
)
from dataset import Whisper_Dataset

hidden_states_dict = {}

def create_hook_fn(name):
    def hook(module, args, kwargs, output):
        hidden_states_dict[name] = kwargs["hidden_states"]

    return hook


def main():
    import argparse

    cmd_parser = argparse.ArgumentParser()
    cmd_parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )
    cmd_parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        help="Path to the output directory.",
    )
    cmd_parser.add_argument(
        "--sample_size",
        type=int,
        default=1024,
    )
    args = cmd_parser.parse_args()
    config = load_config(args.config_file)
    parser = HfArgumentParser((MHA2MLATrainingArguments, ModelArguments, DataArguments))
    training_args, model_args, dataset_args = parser.parse_dict(config)
    # assert config["DataArguments"]["DP"] == int(os.environ.get("WORLD_SIZE", 1)), "DP is not equal to WORLD_SIZE"

    # Trainer
    model, tokenizer = load_whisper(model_args)
    train_dataset = Whisper_Dataset(dataset_args.train_ann_path, dataset_args.whisper_path)
    assert (
        int(os.getenv("WORLD_SIZE", 1)) == 1
    ), "Only support single process." 
    
    data_loader = get_dataloader(train_dataset, training_args)

    num = args.sample_size
    model.eval()
    model.to("cuda")
    for name, module in model.named_modules():
        from modeling_whisper import WhisperAttention
        if not isinstance(module, WhisperAttention):
            continue
        hook_fn = create_hook_fn(name)
        module.register_forward_hook(hook_fn,with_kwargs=True)
    p_bar = tqdm(total=num)
    model_config = model.config
    head_dim = model_config.d_model // model_config.encoder_attention_heads
    num_layers = model_config.num_hidden_layers
    query_states = [[] for _ in range(num_layers*3)]
    key_states = [[] for _ in range(num_layers*3)]
    def cal_2_norm(states):
        states = torch.norm(
            states.reshape(states.shape[0],states.shape[1],states.shape[2],2,-1).transpose(-1,-2),
            p=2,
            dim=4,
        )
        return states
    with torch.no_grad():
        for _,batch in enumerate(data_loader):
            batch = prepare_sample(batch, cuda_enabled=True)
            spectrogram = batch["input_features"]
            text = [t for t in batch["text"]]
            to_regress_tokens = tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=2048,
                add_special_tokens=False
            ).to(spectrogram.device)
            text_tokens = to_regress_tokens.input_ids
            model(input_features = spectrogram, labels = text_tokens)
            num -= text_tokens.shape[0]
            p_bar.update(text_tokens.shape[0])
            idx = 0
            for name,module in model.named_modules():
                if not isinstance(module, WhisperAttention):
                    continue
                bsz,q_len,_ = hidden_states_dict[name].shape
                q = module.q_proj(hidden_states_dict[name]).reshape(bsz,q_len,model_config.encoder_attention_heads,head_dim) # [bsz,q_len,num_heads,head_dim]
                k = module.k_proj(hidden_states_dict[name]).reshape(bsz,q_len,model_config.encoder_attention_heads,head_dim)
                query_states[idx].append(cal_2_norm(q).mean(dim=1,keepdim=False).cpu()) # [bsz,num_heads,head_dim//2]
                key_states[idx].append(cal_2_norm(k).mean(dim=1,keepdim=False).cpu())
                idx = idx + 1
            if num <= 0:
                break
    query_states = torch.stack([torch.cat(query_states[i],dim=0) for i in range(num_layers*3)],dim=0) # [num_layers,sample_size,num_heads,head_dim//2]
    key_states = torch.stack([torch.cat(key_states[i],dim=0) for i in range(num_layers*3)],dim=0)
    query_states = torch.mean(query_states,dim=1,keepdim=False) # [num_layers,num_heads,head_dim//2]
    key_states = torch.mean(key_states,dim=1,keepdim=False)
    group_size = model_config.encoder_attention_heads // model_config.encoder_attention_heads
    key_states = (
        key_states.unsqueeze(2)
        .expand(
            num_layers*3,
            model_config.encoder_attention_heads,
            group_size,
            head_dim // 2,
        )
        .reshape(num_layers*3, model_config.encoder_attention_heads, head_dim // 2)
    )
    qk_states = query_states * key_states
    #if group_size > 1:
    #    qk_states = qk_states.reshape(num_layers,model_config.num_key_value_heads,group_size,head_dim//2).sum(dim=2,keepdim=False)
    _, sorted_indices = torch.sort(qk_states, dim=-1, descending=True)
    ranks = torch.empty_like(sorted_indices, dtype=torch.uint8)  # ,dtype=torch.uint8
    rank_values = torch.arange(qk_states.shape[-1], dtype=torch.uint8).expand_as(
        qk_states
    )
    ranks.scatter_(-1, sorted_indices, rank_values)
    ranks = torch.cat([ranks, ranks], dim=-1)
    with open(model_args.qk_tensor_path, "wb") as f:
        torch.save(ranks, f)

if __name__ == "__main__":
    main()
