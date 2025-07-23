torchrun --nproc_per_node 1 \
    src/whisper_mla/run_train.py \
    --config_file configs/2norm-joint.yaml