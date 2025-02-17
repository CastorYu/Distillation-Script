import os
import torch
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from tqdm import tqdm
import deepspeed
import argparse
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class DistillDataset(Dataset):
    def __init__(self, tokenizer, dataset_path, cache_path):
        self.dataset = load_dataset("parquet", data_dir=dataset_path, split="train")
        self.tokenizer = tokenizer
        self.outputs = torch.load(cache_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]["text"]
        inputs = self.tokenizer(
            text,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            **{k: v.squeeze(0) for k, v in inputs.items()},
            "teacher_logits": self.outputs[idx]
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--cache_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    # Initialize DeepSpeed
    deepspeed.init_distributed()

    # Load student
    student = AutoModel.from_pretrained(args.student_path)
    tokenizer = AutoTokenizer.from_pretrained(args.student_path)

    # Dataset
    dataset = DistillDataset(tokenizer, args.dataset_path, args.cache_path)
    train_loader = DataLoader(
        dataset,
        batch_size=16,
        num_workers=4,
        pin_memory=True
    )

    # DeepSpeed engine
    student_engine, *_ = deepspeed.initialize(
        args=args,
        model=student,
        config="configs/ds_student_config.json"
    )

    # Training loop
    for epoch in range(3):
        student_engine.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            inputs = {k: v.to(student_engine.device) for k, v in batch.items()}
            student_outputs = student_engine(**inputs)
            
            loss = torch.nn.functional.kl_div(
                torch.log_softmax(student_outputs.logits / 2.0, dim=-1),
                torch.softmax(inputs["teacher_logits"].to(student_engine.device) / 2.0,
                reduction="batchmean"
            ) * 4.0  # T=2.0
            
            student_engine.backward(loss)
            student_engine.step()
            total_loss += loss.item()

        # Save checkpoint
        if student_engine.local_rank == 0:
            student_engine.save_checkpoint(args.output_dir, tag=f"epoch{epoch}")

    # Final save
    if student_engine.local_rank == 0:
        student_engine.module.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()