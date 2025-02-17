deepspeed --num_gpus 1 scripts/precompute_teacher.py \
  --model_path models/teacher \
  --dataset_path data/dataset \
  --output_dir outputs