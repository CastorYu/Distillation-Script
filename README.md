# Distillation-Script
This script is for distilling two models using ZeRO. (Though may contain bugs)

The files should be organized as following: 

distillation-script/  
├── configs/  
│   ├── ds_teacher_config.json    # ZeRO-3 config for teacher  
│   └── ds_student_config.json    # ZeRO-2 config for student  
├── scripts/  
│   ├── precompute_teacher.py     # Teacher output caching  
│   └── train_distill.py          # Main training script  
├── models/  
│   ├── teacher/                  # Teacher model files  
│   └── student/                  # Student model files  
├── data/  
│   └── dataset/                  # Parquet dataset files  
├── outputs/                      # Cached outputs & final model  
└── requirements.txt  

First run install_dependencies.sh then precompute_teacher_outputs.sh and finally train_student_model.sh. 
Dataset should be stored in parquet version such as 'wikitext' on huggingface.co. 
The dataset tokenized by the teacher model will be stored on disk so make sure there are ample space available. 
Teacher and Student models should be in safetensors format and config.json is needed for the script to run. 
The final result will be outputted in the results folder. 
It is recommanded to be ran on gpu under linux platform as deepspeed has compatibility problems on Windows10/11. 
