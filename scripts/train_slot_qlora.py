"""
슬롯 추출 전용 QLoRA 파인튜닝 스크립트

모델: Qwen/Qwen2.5-1.5B-Instruct (HF 로그인 불필요, ~3GB)
방법: Unsloth + LoRA (r=16) + SFTTrainer
데이터: data/finetune/train.jsonl, val.jsonl
출력: models/slot_extractor_adapter/  (LoRA 어댑터만 저장, ~50MB)

실행:
    python scripts/train_slot_qlora.py
    python scripts/train_slot_qlora.py --model Qwen/Qwen2.5-3B-Instruct --epochs 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# ── 하이퍼파라미터 기본값 ─────────────────────────────────────────────────────
DEFAULTS = {
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "train_file": "data/finetune/train.jsonl",
    "val_file": "data/finetune/val.jsonl",
    "output_dir": "models/slot_extractor_adapter",
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "epochs": 3,
    "batch_size": 4,
    "grad_accum": 4,         # effective batch = 16
    "lr": 2e-4,
    "max_seq_len": 1024,
    "warmup_ratio": 0.05,
    "save_steps": 50,
    "eval_steps": 50,
    "logging_steps": 10,
    "seed": 42,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        p.add_argument(f"--{k}", type=type(v), default=v)
    return p.parse_args()


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    args = parse_args()

    # ── 1. Unsloth 모델 로드 (4-bit QLoRA) ──────────────────────────────────
    from unsloth import FastLanguageModel
    import torch

    print(f"[1/5] 모델 로드: {args.model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_len,
        dtype=None,           # auto (fp16/bf16)
        load_in_4bit=True,
    )

    # ── 2. LoRA 어댑터 추가 ──────────────────────────────────────────────────
    print("[2/5] LoRA 어댑터 추가")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",  # VRAM 절약
        random_state=args.seed,
    )

    # ── 3. 데이터 로드 ────────────────────────────────────────────────────────
    print("[3/5] 데이터 로드")
    root = Path(__file__).parent.parent
    train_data = load_jsonl(root / args.train_file)
    val_data = load_jsonl(root / args.val_file)
    print(f"  train: {len(train_data)}개  val: {len(val_data)}개")

    # datasets.Dataset 변환 + chat template 적용
    from datasets import Dataset

    def apply_template(sample: dict) -> dict:
        """messages → 단일 text 문자열 (chat template 적용)."""
        text = tokenizer.apply_chat_template(
            sample["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    def to_hf_dataset(records: list[dict]) -> Dataset:
        ds = Dataset.from_list(records)
        return ds.map(apply_template, remove_columns=["messages"])

    train_ds = to_hf_dataset(train_data)
    val_ds   = to_hf_dataset(val_data)

    # ── 4. SFTTrainer 설정 ────────────────────────────────────────────────────
    print("[4/5] 학습 설정")
    from trl import SFTTrainer, SFTConfig

    training_args = SFTConfig(
        output_dir=str(root / args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=args.seed,
        max_seq_length=args.max_seq_len,
        dataset_text_field="text",
        packing=False,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=training_args,
    )

    # ── 5. 학습 ───────────────────────────────────────────────────────────────
    print("[5/5] 학습 시작")
    trainer.train()

    # 어댑터만 저장 (base model 제외 → ~50MB)
    adapter_path = root / args.output_dir
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"\n완료! 어댑터 저장 → {adapter_path}")
    print("추론 통합: app/agents/preference_agent.py 의 _extract_slots_llm 교체 필요")


if __name__ == "__main__":
    main()
