import argparse

from transformers import (
    AutoModelForCausalLM,
)

from peft import LoraModel, PeftModel


def main(model_id: str, lora_path: str, output_path: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="float32",
        trust_remote_code=False,  # quantization_config=BitsAndBytesConfig(load_in_4bit=True)
    )
    model: LoraModel = PeftModel.from_pretrained(model, lora_path)
    model = model.merge_and_unload(progressbar=True)
    model.save_pretrained(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("output", type=str)
    parser.add_argument("--base_model", default="LiquidAI/LFM2.5-230M", required=False, type=str)
    args = parser.parse_args()
    main(args.base_model, args.checkpoint_path, args.output)
