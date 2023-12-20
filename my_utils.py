import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
    logging,
)
from peft import LoraConfig
from guardrail.client import run_metrics
from config import *

def load_model(model_name):
    # Load tokenizer and model with QLoRA configuration
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )

    if compute_dtype == torch.float16 and use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
            print("=" * 80)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        quantization_config=bnb_config
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Load LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer, peft_config

def format_dolly(sample):
    instruction = f"<s>[INST] {sample['instruction']}"
    context = f"Here's some context: {sample['context']}" if len(sample["context"]) > 0 else None
    response = f" [/INST] {sample['response']}"
    # join all the parts together
    prompt = "".join([i for i in [instruction, context, response] if i is not None])
    return prompt

# template dataset to add prompt to each sample
def template_dataset(sample):
    sample["text"] = f"{format_dolly(sample)}{tokenizer.eos_token}"
    return sample


def text_gen_eval_wrapper(model, tokenizer, prompt, model_id=1, show_metrics=True, temp=0.7, max_length=200):
    """
    A wrapper function for inferencing, evaluating, and logging text generation pipeline.

    Parameters:
        model (str or object): The model name or the initialized text generation model.
        tokenizer (str or object): The tokenizer name or the initialized tokenizer for the model.
        prompt (str): The input prompt text for text generation.
        model_id (int, optional): An identifier for the model. Defaults to 1.
        show_metrics (bool, optional): Whether to calculate and show evaluation metrics.
                                       Defaults to True.
        max_length (int, optional): The maximum length of the generated text sequence.
                                    Defaults to 200.

    Returns:
        generated_text (str): The generated text by the model.
        metrics (dict): Evaluation metrics for the generated text (if show_metrics is True).
    """
    # Suppress Hugging Face pipeline logging
    logging.set_verbosity(logging.CRITICAL)

    # Initialize the pipeline
    pipe = pipeline(task="text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_length=max_length,
                    do_sample=True,
                    temperature=temp)

    # Generate text using the pipeline
    pipe = pipeline(task="text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_length=200)
    result = pipe(f"<s>[INST] {prompt} [/INST]")
    generated_text = result[0]['generated_text']

    # Find the index of "### Assistant" in the generated text
    index = generated_text.find("[/INST] ")
    if index != -1:
        # Extract the substring after "### Assistant"
        substring_after_assistant = generated_text[index + len("[/INST] "):].strip()
    else:
        # If "### Assistant" is not found, use the entire generated text
        substring_after_assistant = generated_text.strip()

    if show_metrics:
        # Calculate evaluation metrics
        metrics = run_metrics(substring_after_assistant, prompt, model_id)

        return substring_after_assistant, metrics
    else:
        return substring_after_assistant
