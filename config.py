# This is used for multi-GPU training. A value of -1 means that the device is not part of a distributed group.
local_rank = -1

# batch size for each training device (GPU or CPU), a larger batch size may improve the stability of the training, but also require more memory.
per_device_train_batch_size = 4
per_device_eval_batch_size = 4

learning_rate = 2e-4

# number of steps to accumulate gradients before performing a parameter update
gradient_accumulation_steps = 1 # check

# maximum norm for the gradient vector. A large gradient norm may indicate a steep or noisy loss surface, which can cause instability or divergence in the training.
max_grad_norm = 0.3

# a regularization technique that adds a penalty term to the loss function based on the magnitude of the model’s parameters
weight_decay = 0.001

# controls the trade-off between the rank and the approximation error of the LoRA layer.
lora_alpha = 16

# dropout in the LoRA layer
lora_dropout = 0.1

# the rank of the LoRA layer. Determines the dimensionality of the low-rank approximation. Smaller rank reduces the number of parameters and memory footprint, but increases the approximation error.
lora_r = 64

# maximum length of input sequences for tokenization. 
max_seq_length = None


############################################################
# The model that you want to train
model_name = "helpModels/base_mod"

# Fine-tuned model name
new_model = "llama-2-7b-guanaco-dolly-mini"

# The instruction dataset to use
dataset_name = "databricks/databricks-dolly-15k"

# Activate 4-bit precision base model loading
use_4bit = True

# Activate nested quantization for 4-bit base models
use_nested_quant = False

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Number of training epochs
num_train_epochs = 2

# Enable fp16 training, (bf16 to True with an A100)
fp16 = False

# Enable bf16 training
bf16 = False

# Use packing dataset creating
packing = False

# Enable gradient checkpointing
gradient_checkpointing = True

# Optimizer to use, original is paged_adamw_32bit
optim = "paged_adamw_32bit"

# Learning rate schedule (constant a bit better than cosine, and has advantage for analysis)
lr_scheduler_type = "cosine"

# Number of optimizer update steps, 10K original, 20 for demo purposes
max_steps = -1

# Fraction of steps to do a warmup for
warmup_ratio = 0.03

# Group sequences into batches with same length (saves memory and speeds up training considerably)
group_by_length = True

# Save checkpoint every X updates steps
save_steps = 10

# Log every X updates steps
logging_steps = 1

# The output directory where the model predictions and checkpoints will be written
output_dir = "./results"

# Load the entire model on the GPU 0
device_map = {"": 0}

# Visualize training
report_to = "tensorboard"

# Tensorboard logs
tb_log_dir = "./results/logs"