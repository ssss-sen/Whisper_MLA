import time
import torch
import argparse
import numpy as np
import os
import pandas as pd
import psutil
from transformers import AutoTokenizer, GenerationConfig, LogitsProcessor, LogitsProcessorList, AutoModelForCausalLM, HfArgumentParser, TrainingArguments
from transformers import WhisperTokenizer, WhisperFeatureExtractor, WhisperConfig
from modeling_whisper import WhisperModel, WhisperForConditionalGeneration
from patching_model_load import patch_model
from patching_whisper import mha2mla_whisper, mha2mla_mla_whisper
from safetensors.torch import load_file
from run_train import (
    ModelArguments,
    DataArguments,
    load_config,
    expand_embeddings
)
from inference_mla import make_linear_from_emb

DEVICE = "cuda:0"

class TimeMeasuringLogitsProcessor(LogitsProcessor):
    def __init__(self):
        self.token_times = [time.time()]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        """The logit processor is called after the model forward."""

        # cuda runs async operates, so we synchronize for accurate time measurement
        if DEVICE == "cuda:0":
            torch.cuda.synchronize()
        elif DEVICE == "xpu:0":
            torch.xpu.synchronize()

        # measure time
        start_time = time.time()
        self.token_times.append(start_time)
        return scores

    def get_prefill_duration(self):
        return self.token_times[1] - self.token_times[0]

    def get_decode_durations(self):
        token_times = self.token_times[1:]
        token_durations = [token_times[i + 1] - token_times[i] for i in range(len(token_times) - 1)]

        return token_durations

def warmup(model):
    warm_up = torch.randn((4096,4096)).to(next(model.parameters()).device)
    torch.mm(warm_up,warm_up)

def generate_torch(model, input_ids, n_generate):
    context_time = 0
    generate_time = []

    with torch.inference_mode():
        for i in range(n_generate):
            if DEVICE == "cuda:0":
                torch.cuda.synchronize()
            elif DEVICE == "xpu:0":
                torch.xpu.synchronize()
            start = time.time()

            if i == 0:
                # prefill context
                inputs = torch.as_tensor(input_ids, device=next(model.parameters()).device)
            else:
                # decode tokens
                inputs = torch.as_tensor(token, device=next(model.parameters()).device)

            decoder_start_token_id = model.config.decoder_start_token_id
            decoder_input_ids = torch.ones((input_ids.size(0), 1), dtype=torch.long, device=inputs.device) * decoder_start_token_id
            out = model(inputs, decoder_input_ids=decoder_input_ids, use_cache=True)

            if DEVICE == "cuda:0":
                torch.cuda.synchronize()
            elif DEVICE == "xpu:0":
                torch.xpu.synchronize()
            token = out[0][:, -1].max(1)[1].unsqueeze(1)

            if i == 0:
                context_time += time.time() - start
            else:
                generate_time.append(time.time() - start)

    return context_time, generate_time

def generate_hf(model, input_ids, n_generate):
    generation_config = GenerationConfig(
        min_new_tokens=n_generate,
        max_new_tokens=n_generate,
        use_cache=True,
        forced_eos_token_id=1,
        eos_token_id=1,
    )

    time_processor = TimeMeasuringLogitsProcessor()

    output = model.generate(
        input_features=input_ids,
        generation_config=generation_config,
        logits_processor=LogitsProcessorList([time_processor]),
        #cache_implementation="quantized",
        #cache_config={"backend": "HQQ", "nbits": 4, "q_group_size": 128, "residual_length": 64, "device":model.device}
    )

    context_time = time_processor.get_prefill_duration()
    generate_time = time_processor.get_decode_durations()

    return context_time, generate_time

def run_round(generator, model_path, config_path, n_generate, context, input_ids, batch_size, no_safetensors, use_mla):
    print(f" -- Loading model...")

    if use_mla:
        config = load_config(config_path)
        parser = HfArgumentParser((TrainingArguments, ModelArguments, DataArguments))
        test_args, mha2mla_args, dataset_args = parser.parse_dict(config)
        config = WhisperConfig.from_pretrained(model_path)
        mha_model = WhisperForConditionalGeneration.from_pretrained(
            model_path,
            trust_remote_code=True, 
            torch_dtype=torch.float32,
        )
        mla_model, q_idx, k_idx = patch_model(mha_model, config, mha2mla_args)
        mha2mla_mla_whisper()
        signle_weight_file = os.path.join(model_path, "model.safetensors")
        state_dict = load_file(signle_weight_file)
        mla_model.load_state_dict(state_dict, strict=False)
        model = mla_model
        model.proj_out = make_linear_from_emb(model.model.decoder.embed_tokens)
        model = model.to(DEVICE)
    else:
        model = WhisperForConditionalGeneration.from_pretrained(
            model_path,
            trust_remote_code=True, 
            torch_dtype=torch.float32,
        ).to(DEVICE)

    print(f" -- Warming up...")
    warmup(model)
    print(model.dtype)
    input_ids = input_ids.to(model.dtype)
    print(input_ids.dtype)

    print(f" -- Generating {n_generate} tokens, {input_ids.shape[1]} in context...")

    try:
        context_time, generate_time = generator(model, input_ids, n_generate)
        successful_generate = True
    except RuntimeError as ex:
        if 'out of memory' in str(ex).lower():
            successful_generate = False
        else:
            raise RuntimeError(ex)

    total_memory_used = 0
    memory_pct = 100
    if successful_generate:
        # number of tokens in context / time for processing context * batch size
        prefill_tokens_per_second = round(input_ids.shape[1] / context_time * batch_size, 2)
        # 1 second / median time per token in seconds * batch size
        decode_tokens_per_second = round(1 / np.median(generate_time) * batch_size, 2)

        print(f" ** Speed (Prefill): {prefill_tokens_per_second:.2f} tokens/second")
        print(f" ** Speed (Decode): {decode_tokens_per_second:.2f} tokens/second")

        if DEVICE == "cpu":
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            memory_info = psutil.virtual_memory()
            memory_pct = mem_info.rss / memory_info.total
            total_memory_used = float(mem_info.rss) / (1024 ** 3)
            print(f" ** Max Memory (device: {DEVICE}): {total_memory_used:.2f} GB ({memory_pct:.2f}%)")
        elif DEVICE == "xpu:0":
            for device in range(torch.xpu.device_count()):
                memory_used = torch.xpu.max_memory_allocated(device) / (1024 ** 3)
                total_memory_used += memory_used
                memory_pct = memory_used / (torch.xpu.get_device_properties(device).total_memory / (1024 ** 3)) * 100
                print(f" ** Max Memory (device: {device}): {memory_used:.2f} GB ({memory_pct:.2f}%)")
        else:
            for device in range(torch.cuda.device_count()):
                memory_used = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                total_memory_used += memory_used
                memory_pct = memory_used / (torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)) * 100
                print(f" ** Max Memory (device: {device}): {memory_used:.2f} GB ({memory_pct:.2f}%)")
    else:
        prefill_tokens_per_second = 'OOM'
        decode_tokens_per_second = 'OOM'

    if use_mla:
        version = "mla"
    else:
        version = "mha"

    return {
        "Batch Size": batch_size,
        "Prefill Length": input_ids.shape[1],
        "Decode Length": n_generate,
        "Prefill tokens/s": prefill_tokens_per_second,
        "Decode tokens/s": decode_tokens_per_second,
        "Memory (VRAM)": f"{total_memory_used:.2f} GB ({memory_pct:.2f}%)"
    }, version

def main(args):
    rounds = [
        {"context": 3000, "n_generate": 32},
        {"context": 3000, "n_generate": 64},
        {"context": 3000, "n_generate": 128},
        {"context": 3000, "n_generate": 256},
        {"context": 3000, "n_generate": 512},
        {"context": 3000, "n_generate": 1024},
        {"context": 3000, "n_generate": 2048},
        {"context": 3000, "n_generate": 4096},
    ]

    if args.generator == "torch":
        generator = generate_torch
    elif args.generator == "hf":
        generator = generate_hf
    else:
        raise ValueError(f"Unknown generator method passed: {args.generator}")

    all_stats = []
    tokenizer = WhisperTokenizer.from_pretrained(
        args.model_path, 
        trust_remote_code=True, 
        language = "english", 
        task = "transcribe",
        padding_side="left"
    )

    for settings in rounds:
        mel_dim = 80
        input_ids = torch.randn(args.batch_size, mel_dim, settings["context"])
        if DEVICE == "cuda:0":
            input_ids = input_ids.cuda()
        elif DEVICE == "xpu:0":
            input_ids = input_ids.to("xpu:0")

        stats, model_version = run_round(
            generator,
            args.model_path,
            args.mha2mla_config_path,
            settings["n_generate"],
            settings["context"],
            input_ids,
            args.batch_size,
            args.no_safetensors,
            args.use_mla,
        )

        all_stats.append(stats)

        if stats["Prefill tokens/s"] == 'OOM':
            break

    df = pd.DataFrame(all_stats)
    print('Device:', DEVICE)
    if DEVICE == "cuda:0":
        print('GPU:', torch.cuda.get_device_name())
    elif DEVICE == "xpu:0":
        print('XPU:', torch.xpu.get_device_name())
    print('Model:', args.model_path)
    print('Version:', model_version)
    print(df.to_markdown(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="casperhansen/mistral-7b-instruct-v0.1-awq", help="path to the model")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for cache and generation")
    parser.add_argument("--no_safetensors", default=False, action="store_true", help="Use for disabling safetensors")
    parser.add_argument("--generator", type=str, default="hf", choices=["torch", "hf"], help="weights filename")
    parser.add_argument("--use_mla", default=False, action="store_true", help="Whether to use mla.")
    parser.add_argument("--mha2mla_config_path", type=str, help="Path to config file")
    args = parser.parse_args()

    main(args)