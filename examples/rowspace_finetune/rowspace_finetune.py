import argparse

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from peft import LoraConfig, PeftType, RowSpaceConfig, TaskType, get_peft_model, prepare_model_for_kbit_training


def main(
    model_id: str = "LiquidAI/LFM2.5-230M",
    tokenizer_id: str | None = None,
    rowspace_split_method: str = "naive",
    rank: int = 8,
):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=False,  # quantization_config=BitsAndBytesConfig(load_in_4bit=True)
    )
    prepare_model_for_kbit_training(model)
    if tokenizer_id is None:
        tokenizer_id = model_id
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    model = get_peft_model(
        model,
        LoraConfig(
            TaskType.CAUSAL_LM,
            PeftType.LORA,
            r=rank,
            rowspace_config=RowSpaceConfig(split_method=rowspace_split_method),
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embedding"],
        ),
    )

    dataset = load_dataset("openai/gsm8k", "main", split="train[:64]")

    def tokenize(examples: dict[str, str]):
        inputs = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": examples["question"]},
                {"role": "user", "content": examples["answer"]},
            ],
            return_dict=True,
            return_tensors="pt",
        )
        # inputs = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=8196)
        inputs["input_ids"] = inputs["input_ids"][0]
        inputs["attention_mask"] = inputs["attention_mask"][0]
        inputs["labels"] = inputs["input_ids"]
        return inputs

    tokenized_dataset = dataset.map(tokenize)

    trainer = Trainer(
        model,
        TrainingArguments(
            per_device_train_batch_size=8,
            optim="adamw_torch_fused",
            num_train_epochs=1,
        ),
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, max_length=8196, return_tensors="pt"),
        train_dataset=tokenized_dataset,
    )
    output = trainer.train()
    print(f"Train output: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="LiquidAI/LFM2.5-230M", required=False, type=str)
    parser.add_argument("--tokenizer", default=None, required=False, type=str)
    parser.add_argument("--split_method", default="naive", required=False, type=str)
    args = parser.parse_args()
    main(args.model, args.tokenizer, args.split_method)
