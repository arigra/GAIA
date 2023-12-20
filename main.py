########### Imports ###########
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel, get_peft_model
from trl import SFTTrainer
from guardrail.client import (
    run_metrics,
    run_simple_metrics,
    create_dataset)
from config import *
from my_utils import *

########### load the base model (the model we want to train) ###########
model, tokenizer, peft_config = load_model(model_name)

########### load the dataset (the dataset we want to train the model with) ###########
dataset = load_dataset("helpModels/mlabData", split="train") # check if .txt file also fits. if not, find a solution
dataset_shuffled = dataset.shuffle(seed=42)

########### Choosing the first 100 rows from the shuffled dataset ###########
dataset = dataset_shuffled.select(range(100))
'''
prompt = "how long does an American football match REALLY last, if you substract all the downtime?"

pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)

result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result[0]['generated_text'])

prompt = "What is the airspeed velocity of an unladen swallow?"

pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result[0]['generated_text'])
'''
prompt="Who were the children of the legendary Garth Greenhand, the High King of the First Men in the series A Song of Ice and Fire?"
text_gen_eval_wrapper(model, tokenizer, prompt, show_metrics=False)

# Inference and evaluate outputs/prompts
prompt = "### Human: Sophie's parents have three daughters: Amy, Jessy, and what’s the name of the third daughter?"
text_gen_eval_wrapper(model, tokenizer, prompt, show_metrics=False)

prompt = "### Human: Why can camels survive for long without water? ### Assistant:"
generated_text = text_gen_eval_wrapper(model, tokenizer, prompt, show_metrics=False, max_length=250)
print(generated_text)

training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="tensorboard"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
)

trainer.train()

# Ignore warnings
logging.set_verbosity(logging.CRITICAL)

# Run text generation pipeline with our next model
prompt = "What is baba ganoush?"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result[0]['generated_text'])

# Empty VRAM
del model
del pipe
del trainer
import gc
gc.collect()
gc.collect()

# Reload model in FP16 and merge it with LoRA weights
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map,
)
model = PeftModel.from_pretrained(base_model, output_dir)
merged_model = model.merge_and_unload()

merged_model.save_pretrained('/home/leshkar/Desktop/gaia/merged')

# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


#full_path = "llamaFine"

#model, tokenizer, peft_config= load_model(full_path)

prompt="Who were the children of the legendary Garth Greenhand, the High King of the First Men in the series A Song of Ice and Fire?"
text_gen_eval_wrapper(merged_model, tokenizer, prompt, show_metrics=False)

# Inference and evaluate outputs/prompts
prompt = "### Human: Sophie's parents have three daughters: Amy, Jessy, and what’s the name of the third daughter?"
text_gen_eval_wrapper(merged_model, tokenizer, prompt, show_metrics=False)

prompt = "### Human: Why can camels survive for long without water? ### Assistant:"
generated_text = text_gen_eval_wrapper(merged_model, tokenizer, prompt, show_metrics=False, max_length=250)
print(generated_text)

merged_model.save_pretrained('/home/leshkar/Desktop/gaia/merged')