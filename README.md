# Whisper with MLA

Whisper_MLA is a framework for fine-tuning the MHA structure of the Whisper model into the MLA structure. Whisper_MLA significantly reduces gpu memory during the inference process without causing damage to ASR performance. This code is modified from the MHA2MLA code.(https://github.com/JT-Ushio/MHA2MLA)

## News

- [2025.07.20] Released the first version of the Whisper_MLA code, providing usage code for Whisper_MLA fine-tuning and evaluating.

## TO-DO

- [ ] Fine-tune the Whisper model of the MLA architecture using a dataset with long audios. Explore the improvement of long audio recognition capability by the MLA structure.

## Models

- Whisper: https://huggingface.co/openai/whisper-small

## Datasets

First download the datasets.

- LibriSpeech(train-other-500, test-other-500): https://www.openslr.org/12/

## Environment

Install pytorch and other packages.

```sh
conda create -n whisper-mla python=3.11
pip install torch==2.4.0 torchvision==0.19.0
pip install -r requirements.txt
```

## Whisper_MLA Fine-Tuning with LibriSpeech Dataset

First, prepare the configuration files:
1. If you want to decompose the dimensions of the Key using the 2norm approach, you can refer to the configuration file [2norm-joint.yaml](configs/2norm-joint.yaml)
2. If you want to decompose the dimensions of the Key using the uniform approach, you can refer to the configuration file [uniform-joint.yaml](configs/uniform-joint.yaml)


Then, use the following command for MLA fine-tuning:
1. If you want to use the 2norm method, you can run the following command to get the `qk_tensor` first.
    ```sh
    bash scripts/2_norm.sh
    ```
    then, you can run the following command to finetune the Whisper_MLA model.
    ```sh
    bash scripts/train_2norm.sh
    ```

2. If you want to use the uniform method, you can run the following command to finetune the Whisper_MLA model.
    ```sh
    bash scripts/train_uniform.sh
    ```

## ASR Evaluation

If you want to test the ASR results of Whisper_MLA on the LibriSpeech-test-other-500 dataset, you can use the following command:

```sh
bash scripts/infer_mla.sh
```

## GPU Memory Usage Evaluation During Inference

For the gpu memory usage evaluation during inference, you can run the following command:

```sh
bash scripts/eval_speed.sh
```
If you want to compare with the Whisper model of the MHA architecture, you can run the following command to test the gpu memory usage of MHA.

```sh
bash scripts/eval_speed_mha.sh
```

