import os
import torch
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import deepspeed
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    # Initialize DeepSpeed
    deepspeed.init_distributed()

    # Load model
    teacher = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16
    )
    
    # Load dataset
    dataset = load_dataset("parquet", data_dir=args.dataset_path, split="train")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # DeepSpeed engine
    ds_engine, *_ = deepspeed.initialize(
        args=args,
        model=teacher,
        config="configs/ds_teacher_config.json"
    )

    # Precompute outputs
    all_outputs = []
    ds_engine.eval()
    for example in tqdm(dataset, desc="Precomputing"):
        inputs = tokenizer(
            example["text"],
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(ds_engine.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = ds_engine(**inputs).logits.cpu().half()
            
        all_outputs.append(outputs)

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(all_outputs, f"{args.output_dir}/teacher_outputs.pt")

if __name__ == "__main__":
    main()