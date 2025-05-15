from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType
import torch

# ✅ LLaMA-2 HF model (requires login + access)
model_id = "meta-llama/Llama-2-7b-hf"

# ✅ Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# ✅ Load model (no GPU quantization)
model = AutoModelForCausalLM.from_pretrained(model_id).to("cpu")

# ✅ Apply LoRA (PEFT) config
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    inference_mode=False
)
model = get_peft_model(model, peft_config)

# ✅ Load your dataset
dataset = load_dataset("json", data_files="../data/full_train.json")

# ✅ Format prompts
def format_prompt(example):
    text = f"{example['instruction']}\n{example['input']}\n### {example['output']}"
    return tokenizer(text, truncation=True, padding="max_length", max_length=512)

tokenized = dataset.map(format_prompt)

# ✅ Training arguments
training_args = TrainingArguments(
    output_dir="./llama2-finetuned",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    no_cuda=True,               # ✅ Important for CPU
    save_strategy="epoch",
    logging_steps=10,
    fp16=False                  # ✅ Do not use mixed precision on CPU
)

# ✅ Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

# ✅ Train
trainer.train()

# ✅ Save the finetuned model
model.save_pretrained("./llama2-finetuned")
tokenizer.save_pretrained("./llama2-finetuned")
