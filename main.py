import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_from_disk
import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, List
from transformers import AutoTokenizer


@dataclass
class POSDataCollator:
    pad_token_id: int
    pad_pos_id: int = 16  # UPOS_MAP["X"] (Unknown/Other), used as the padding tag

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Find the max length in this specific batch
        max_length = max(len(feature["input_ids"]) for feature in features)

        batch = {"input_ids": [], "pos_tags": []}
        if "attention_mask" in features[0]:
            batch["attention_mask"] = []

        for feature in features:
            seq_len = len(feature["input_ids"])
            pad_len = max_length - seq_len

            # Pad input_ids with the tokenizer's pad token
            padded_input_ids = feature["input_ids"] + [self.pad_token_id] * pad_len
            batch["input_ids"].append(padded_input_ids)

            # Pad pos_tags with '16' (the ID for "X" / None)
            padded_pos_tags = feature["pos_tags"] + [self.pad_pos_id] * pad_len
            batch["pos_tags"].append(padded_pos_tags)

            # Pad attention_mask with 0s (ignore padding) if it exists
            if "attention_mask" in feature:
                padded_mask = feature["attention_mask"] + [0] * pad_len
                batch["attention_mask"].append(padded_mask)

        # Convert everything to PyTorch tensors
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


class ContextualPOSAwareLoss(nn.Module):
    def __init__(self, vocab_pos_map, epsilon=0.1, vocab_size=50257):
        super().__init__()
        self.epsilon = epsilon

        # PRE-COMPUTATION: Calculate the distributions once during setup.
        # There are 17 UPOS tags (0 through 16).
        precomputed = torch.zeros((17, vocab_size), dtype=torch.float32)

        for tag_id in range(17):
            mask = (vocab_pos_map == tag_id).float()
            counts = mask.sum().clamp(min=1.0)
            # Store the normalized "class-only" distribution for this tag
            precomputed[tag_id] = mask / counts

            # register_buffer moves it to the GPU automatically with the model
        self.register_buffer('precomputed_class_dists', precomputed)

    def forward(self, logits, targets, context_tags, mode='partial'):
        logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1)
        context_tags = torch.clamp(context_tags.view(-1), min=0, max=16)

        if mode == 'standard':
            return F.cross_entropy(logits, targets)

        base_dist = self.precomputed_class_dists[context_tags].to(logits.dtype)

        if mode == 'class_only':
            return F.cross_entropy(logits, base_dist)

        elif mode == 'partial':
            # Mathematically equivalent to scattering the (1-epsilon) mass,
            # but uses WAY less memory because F.cross_entropy with 1D targets is hyper-optimized!
            loss_hard = F.cross_entropy(logits, targets)
            loss_soft = F.cross_entropy(logits, base_dist)

            # Combine them based on your epsilon weight
            return ((1.0 - self.epsilon) * loss_hard) + (self.epsilon * loss_soft)





class POSAwareTrainer(Trainer):
    def __init__(self, *args, vocab_pos_map=None, mode='partial', epsilon=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_loss = ContextualPOSAwareLoss(vocab_pos_map, epsilon).to(self.args.device)
        self.mode = mode

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 1. Extract our inputs
        labels = inputs.pop("labels", inputs["input_ids"].clone())
        pos_tags = inputs.pop("pos_tags")

        # 2. Forward pass
        outputs = model(**inputs)
        logits = outputs.logits

        # 3. Shift the logits, labels, and tags for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_pos_tags = pos_tags[..., 1:].contiguous()

        # 4. Calculate our custom loss
        loss = self.custom_loss(shift_logits, shift_labels, shift_pos_tags, mode=self.mode)

        return (loss, outputs) if return_outputs else loss


def main():
    parser = argparse.ArgumentParser(description="BabyLM Partial Credit Training")

    # Ablation Arguments
    parser.add_argument("--mode", type=str, default="partial",
                        choices=["standard", "partial", "class_only"],
                        help="Loss function mode: standard (CE), partial (smoothed), or class_only")
    parser.add_argument("--epsilon", type=float, default=0.5,
                        help="Amount of probability mass to distribute to same-POS tokens")

    # Model/Data Paths
    parser.add_argument("--dataset_path", type=str, default="./babylm_packed_1024")
    parser.add_argument("--vocab_map_path", type=str, default="vocab_pos_map.pt")
    parser.add_argument("--output_root", type=str, default="./results")

    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Set seed for reproducibility
    torch.manual_seed(args.seed)

    # 1. Load the processed dataset
    dataset = load_from_disk(args.dataset_path)

    # 2. Load the static vocab map
    vocab_pos_map = torch.load(args.vocab_map_path)

    # 3. Initialize model
    config = AutoConfig.from_pretrained(
        "gpt2",
        vocab_size=50257,
        n_positions=1024,
    )
    model = AutoModelForCausalLM.from_config(config)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    custom_collator = POSDataCollator(
        pad_token_id=tokenizer.pad_token_id,
        pad_pos_id=16  # Assuming 16 is your 'X' tag
    )

    # Construct unique output directory name
    run_name = f"{args.mode}_eps{args.epsilon}" if args.mode == "partial" else args.mode
    output_dir = os.path.join(args.output_root, run_name)

    # 4. Define Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        num_train_epochs=args.epochs,

        # 1. THE VRAM SAVER: Drop the physical batch size dramatically
        per_device_train_batch_size=8,

        # 2. THE MATH COMPENSATOR: Multiply this to keep your "effective" batch size large
        # 4 (batch) * 16 (accum) = 64 Effective Batch Size
        gradient_accumulation_steps=16,

        # 3. THE ULTIMATE SAFETY NET: Trades ~10% speed for massive memory savings
        gradient_checkpointing=True,
        dataloader_num_workers=4,

        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=500,
        weight_decay=0.1,
        bf16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        remove_unused_columns=False,
        logging_steps=50,  # Lower this so you can see updates faster!
    )

    # 5. Initialize our Custom Trainer
    # (Assuming POSAwareTrainer and ContextualPOSAwareLoss are defined above)
    trainer = POSAwareTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation", dataset["train"].select(range(1000))),  # Fallback for val
        vocab_pos_map=vocab_pos_map,
        mode=args.mode,
        epsilon=args.epsilon,
        data_collator=custom_collator,
    )

    if args.mode == "standard":
        args.epsilon = 0.0
    elif args.mode == "class_only":
        args.epsilon = 1.0

    # 6. Train!
    print(f"--- Starting Experiment ---")
    print(f"Mode:    {args.mode}")
    print(f"Epsilon: {args.epsilon}")
    print(f"Output:  {output_dir}")
    print(f"---------------------------")

    trainer.train()


if __name__ == "__main__":
    main()