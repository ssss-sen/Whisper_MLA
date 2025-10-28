# modeling_whisper_mla.py
from modeling_whisper import WhisperForConditionalGeneration
from dataclasses import asdict
from patching_model_load import patch_model
from patching_whisper import mha2mla_mla_whisper
from argument import ModelArguments, DataArguments, MHA2MLATrainingArguments

# 假设你已有的函数
# from your_patch import patch_model, mha2mla_mla_whisper, ModelArguments

class WhisperMLAForConditionalGeneration(WhisperForConditionalGeneration):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        mha2mla_args = kwargs.pop("mha2mla", None)
        # 其次从 config 取（已保存到 config.json 的情况）
        if mha2mla_args is None:
            mha2mla_args = getattr(config, "mha2mla", None)

        # 没有就直接跳过，避免 AttributeError
        if not mha2mla_args:
            return

        # 如果需要，转成你定义的 dataclass
        if isinstance(mha2mla_args, dict):
            model_args = ModelArguments(**mha2mla_args)
        else:
            model_args = mha2mla_args

        # 就地打补丁：直接修改 self 的子模块
        patch_model(self, config, model_args)
        mha2mla_mla_whisper(self)
