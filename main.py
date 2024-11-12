from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, HfArgumentParser, TrainingArguments
import os
from trl import SFTTrainer
from datasets import load_dataset
from train import train_model
from inference import inference_model
tqdm.pandas()


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with SFTTrainer
    """
    name: Optional[str] = field(default="exp", metadata={"help": "the experiment name"})
    mode: Optional[str] = field(default="train", metadata={"help": "the experiment mode (train or test)"})
    lora_folder: Optional[str] = field(default="./cp", metadata={"help": "path to adapters folder"})
    model_name: Optional[str] = field(default="meta-llama/Llama-2-7b-hf", metadata={"help": "the model name"})

    dataset_op: Optional[str] = field(default="*", metadata={"help": "'*', '+', '-', 'sin', 'sqrt"})
    dataset_num_digit: Optional[int] = field(default=2, metadata={"help": "operands num of digits"})
    dataset_format: Optional[str] = field(default='plain', metadata={"help": "plain ot num_digit"})
    dataset_num_train: Optional[int] = field(default=3000, metadata={"help": "number of training samples"})
    dataset_num_test: Optional[int] = field(default=1000, metadata={"help": "number of test samples"})
    tokenized_dataset_path: Optional[str] = field(
        default="", metadata={"help": "path to dir with npy files"}
    )
    dataset_name: Optional[str] = field(default="", metadata={"help": "hf dataset name"})
    # test_dataset_path: Optional[str] = field(
    #     default="../teaching_arithmetic/data/multiplication/plain/train_examples_3000.txt", metadata={"help": "the csv dataset path"}
    # )
    dataset_text_field: Optional[str] = field(default="text", metadata={"help": "the text field of the dataset"})
    log_with: Optional[str] = field(default="none", metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    batch_size: Optional[int] = field(default=128, metadata={"help": "the batch size"})
    seq_length: Optional[int] = field(default=2048, metadata={"help": "Input sequence length"})
    gradient_accumulation_steps: Optional[int] = field(
        default=2, metadata={"help": "the number of gradient accumulation steps"}
    )
    try_q: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 precision on test"})

    bf16: Optional[bool] = field(default=False, metadata={"help": "train the model in bf16 bits precision"})
    fsdp: Optional[bool] = field(default=False, metadata={"help": "train the model with fsdp"})
    load_in_8bit: Optional[bool] = field(default=True, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})
    use_peft: Optional[bool] = field(default=True, metadata={"help": "Whether to use PEFT or not to train adapters"})
    debug: Optional[bool] = field(default=False, metadata={"help": "debug"})
    debug_ip: Optional[str] = field(default='', metadata={"help": ""})
    debug_port: Optional[int] = field(default=12345, metadata={"help": ""})
    trust_remote_code: Optional[bool] = field(default=False, metadata={"help": "Enable `trust_remote_code`"})
    output_dir: Optional[str] = field(default="output", metadata={"help": "the output directory"})
    peft_lora_r: Optional[int] = field(default=64, metadata={"help": "the r parameter of the LoRA adapters"})
    peft_lora_alpha: Optional[int] = field(default=16, metadata={"help": "the alpha parameter of the LoRA adapters"})
    logging_steps: Optional[int] = field(default=1, metadata={"help": "the number of logging steps"})
    use_auth_token: Optional[bool] = field(default=True, metadata={"help": "Use HF auth token to access the model"})
    num_train_epochs: Optional[int] = field(default=100, metadata={"help": "the number of training epochs"})
    max_steps: Optional[int] = field(default=-1, metadata={"help": "the number of training steps"})
    save_steps: Optional[int] = field(
        default=100, metadata={"help": "Number of updates steps before two checkpoint saves"}
    )
    save_total_limit: Optional[int] = field(default=10, metadata={"help": "Limits total number of checkpoints."})
    push_to_hub: Optional[bool] = field(default=False, metadata={"help": "Push the model to HF Hub"})
    hub_model_id: Optional[str] = field(default=None, metadata={"help": "The name of the model on HF Hub"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

if script_args.debug:
    import pydevd_pycharm
    ip = script_args.debug_ip if script_args.debug_ip else os.environ['SSH_CONNECTION'].split()[0]
    pydevd_pycharm.settrace(ip, port=script_args.debug_port,
                            stdoutToServer=True,
                            stderrToServer=True, suspend=False)

if script_args.mode == "train":
    train_model(script_args)
elif script_args.mode == "test":
    inference_model(script_args)
