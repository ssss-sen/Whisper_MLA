torchrun --nproc_per_node 1 \
  src/whisper_mla/2_norm.py \
  --config_file configs/2norm-joint.yaml \
  --output_dir output/mha2mlaqk_tensor_hf_test.pth \
  --sample_size 1024
  