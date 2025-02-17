deepspeed --num_gpus 1 scripts/train_distill.py \
  --student_path models/student \
  --dataset_path data/dataset \
  --cache_path outputs/teacher_outputs.pt \
  --output_dir outputs/final_model