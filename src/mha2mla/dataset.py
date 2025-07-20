# Copyright (2024) Tsinghua University, Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import soundfile as sf
import numpy as np
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
import torch.nn.functional as F


class Whisper_Dataset(Dataset):
    def __init__(self, ann_path, whisper_path):
        super().__init__()

        self.annotation = json.load(open(ann_path, "r"))["annotation"]

        self.wav_processor = WhisperFeatureExtractor.from_pretrained(whisper_path)
        self.tokenizer = WhisperTokenizer.from_pretrained(whisper_path, language = "english", task = "transcribe")

    def __len__(self):
        return len(self.annotation)

    def collater(self, samples):
        audio_list = [s["audio_path"] for s in samples]
        text_list = [s["text"] for s in samples]
        samples_spectrogram = [s["input_features"] for s in samples]
        cat_spectrogram = torch.stack(samples_spectrogram, dim=0)

        samples_text = [s["labels"] for s in samples]
        max_len = max([t.size(1) for t in samples_text])
        padded_samples = []
        for t in samples_text:
            pad_size = max_len - t.size(1)
            padded_samples.append(F.pad(t, (0, pad_size), value=self.tokenizer.eos_token_id))
        
        cat_label = torch.cat(padded_samples, dim=0)

        return {
            #"audio_path": audio_list,
            "input_features": cat_spectrogram,
            "labels": cat_label,
            #"text": text_list,
        }

    def infer_collater(self, samples):
        audio_list = [s["audio_path"] for s in samples]
        text_list = [s["text"] for s in samples]
        samples_spectrogram = [s["input_features"] for s in samples]
        cat_spectrogram = torch.stack(samples_spectrogram, dim=0)

        samples_text = [s["labels"] for s in samples]
        max_len = max([t.size(1) for t in samples_text])
        padded_samples = []
        for t in samples_text:
            pad_size = max_len - t.size(1)
            padded_samples.append(F.pad(t, (0, pad_size), value=self.tokenizer.eos_token_id))
        
        cat_label = torch.cat(padded_samples, dim=0)

        return {
            "audio_path": audio_list,
            "input_features": cat_spectrogram,
            "labels": cat_label,
            "text": text_list,
        }

    def __getitem__(self, index):
        ann = self.annotation[index]

        audio, sr = sf.read(ann["path"])
        if len(audio.shape) == 2: # stereo to mono
            audio = audio[:, 0]
        if "expand_wav" in ann:
            for p in ann["expand_wav"]:
                expand_audio, _ = sf.read(p)
                if len(expand_audio.shape) == 2:
                    expand_audio = expand_audio[:, 0]
                sil = np.zeros(1600, dtype=float)
                audio = np.concatenate((audio, sil, expand_audio), axis=0)
        if len(audio) < sr: # pad audio to at least 1s
            sil = np.zeros(sr - len(audio), dtype=float)
            audio = np.concatenate((audio, sil), axis=0)
        audio = audio[: sr * 300] # truncate audio to at most 30s

        spectrogram = self.wav_processor(audio, sampling_rate=sr, return_tensors="pt")["input_features"].squeeze()
        text = ann["text"]
        task = ann.get("task", "asr")
        Q = ann.get("Q", "")
        to_regress_tokens = self.tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=4096,
            add_special_tokens=False
        )  

        return {
            "audio_path": ann["path"],
            "input_features": spectrogram,
            "labels": to_regress_tokens.input_ids,
            "text": text,
        }