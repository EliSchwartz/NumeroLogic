from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator
from peft import LoraConfig
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, AutoTokenizer, DataCollatorForLanguageModeling
import csv
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import load_dataset, Dataset
from gen_arithmetic_data import generate_equations
import numpy as np
import os

from dataset_utils import get_hf_dataset
CUDA_LAUNCH_BLOCKING=1.

from torch.utils.data import Dataset

class NpyDataset(torch.utils.data.Dataset):
    def __init__(self, directory, max_samples=0, seq_length=2048):
        """
        Initializes the dataset object.
        :param directory: Path to the directory containing .npy files.
        """
        self.directory = directory
        self.file_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npy')]
        self.file_paths = sorted(self.file_paths)
        if max_samples and len(self.file_paths) > max_samples:
            self.file_paths = self.file_paths[:max_samples]
        self.seq_length = seq_length

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        Returns a sample from the dataset at the specified index.
        :param idx: Index of the sample to return.
        """
        # Load the numpy array from the file
        npy_data = np.load(self.file_paths[idx]).astype(np.int16)
        npy_data = npy_data[:self.seq_length]
        # Convert the numpy array to a torch tensor
        tensor_data = torch.from_numpy(npy_data)

        # Return the sample as a dictionary
        return {'input_ids': tensor_data}


def write_dicts_to_csv(dicts, csv_path):
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = dicts[0].keys() # Determine the fieldnames from the first dictionary (keys are column headers)
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in dicts:
            writer.writerow(row)

def train_model(script_args):
    # Step 1: Load the model
    if script_args.load_in_8bit and script_args.load_in_4bit:
        raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
    elif script_args.load_in_8bit or script_args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=script_args.load_in_8bit, load_in_4bit=script_args.load_in_4bit
        )
        # Copy the model to each device
        device_map = {"": Accelerator().local_process_index}
        torch_dtype = torch.bfloat16
    else:
        device_map = None
        quantization_config = None
        torch_dtype = None

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=script_args.trust_remote_code,
        torch_dtype=torch_dtype,
        use_auth_token=script_args.use_auth_token,
    )

    tokenizer = AutoTokenizer.from_pretrained(model.name_or_path)
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = tokenizer.eos_token

    # Step 2: Load the dataset
    # dataset = load_dataset('csv', data_files=script_args.csv_dataset_path)
    # dataset = dataset.remove_columns(['image_id'])
    # if script_args.csv_dataset_path == 'multiplication_2dig':
    if script_args.tokenized_dataset_path:
        train_dataset = NpyDataset(script_args.tokenized_dataset_path,
                                   max_samples=script_args.dataset_num_train,
                                   seq_length=script_args.seq_length)
        eval_dataset = None
        max_seq_len = script_args.seq_length
        collator = None
    elif script_args.dataset_name:
        # train_dataset = get_hf_dataset(script_args, tokenizer)
        # train_dataset = load_dataset(script_args.dataset_name, "20231101.en", split="train", streaming=True)
        train_dataset = load_dataset(script_args.dataset_name, split="train", streaming=True)
        train_dataset = train_dataset.rename_columns({'content': 'text'})
        train_dataset = train_dataset.remove_columns(['url', 'timestamp', 'dump', 'segment', 'image_urls'])
        eval_dataset = None
        max_seq_len = script_args.seq_length
        collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    else:
        train_list, eval_list = generate_equations(script_args.dataset_num_train, script_args.dataset_num_test,
                                                   script_args.dataset_num_digit, script_args.dataset_op,
                                                   script_args.dataset_format)
        train_dataset, eval_dataset = Dataset.from_dict({script_args.dataset_text_field: train_list}), Dataset.from_dict({script_args.dataset_text_field: eval_list})
        # dataset = load_dataset('text', data_files=script_args.csv_dataset_path)
        # dataset = dataset['train']

        max_seq_len = max([len(x) for x in train_list]) + 10 # *2 to accomodate extra tokens, e.g. for start and end tokens
        if script_args.dataset_format=='num_digit':
            response_template_with_context = "{2:71}*{2:63}={" if script_args.dataset_op in '+-*' else "sin({1.1:0.5})={"  # We added context here: "=" is encoded differently when appear after numbers
            response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[-2:]
        else:
            response_template_with_context = "0=" if script_args.dataset_op in '+-*' else "sin(0.5)=" # We added context here: "=" is encoded differently when appear after numbers
            response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[-1:]
        collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

    # Step 3: Define the training arguments
    training_args = TrainingArguments(
        output_dir=script_args.output_dir,
        per_device_train_batch_size=script_args.batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        learning_rate=script_args.learning_rate,
        logging_steps=script_args.logging_steps,
        num_train_epochs=script_args.num_train_epochs,
        max_steps=script_args.max_steps,
        report_to=script_args.log_with,
        save_steps=script_args.save_steps,
        save_total_limit=script_args.save_total_limit,
        push_to_hub=script_args.push_to_hub,
        hub_model_id=script_args.hub_model_id,
        evaluation_strategy='epoch',
        logging_strategy='steps',
        bf16=script_args.bf16,
        fsdp=script_args.fsdp,
        dataloader_num_workers=4,
    )

    # Step 4: Define the LoraConfig
    if script_args.use_peft:
        peft_config = LoraConfig(
            r=script_args.peft_lora_r,
            lora_alpha=script_args.peft_lora_alpha,
            bias="none",
            task_type="CAUSAL_LM",
        )
    else:
        peft_config = None


    # if script_args.dataset_format=='num_digit':
    #     response_template_with_context = "{2:71}*{2:63}={" if script_args.dataset_op in '+-*' else "sin({1.1:0.5})={"  # We added context here: "=" is encoded differently when appear after numbers
    #     response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[-2:]
    # else:
    #     response_template_with_context = "0=" if script_args.dataset_op in '+-*' else "sin(0.5)=" # We added context here: "=" is encoded differently when appear after numbers
    #     response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[-1:]
    #
    # if script_args.tokenized_dataset_path or script_args.dataset_name:
    #     collator = None
    # else:
    #     collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

    # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    # instruction_template = "###Human:"
    # response_template = "\n###Assistant:"
    # collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template,
    #                                            response_template=response_template, tokenizer=tokenizer, mlm=False)

    def compute_metrics(eval_preds):
        # metric = evaluate.load("glue", "mrpc")
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        labels = labels[:,1:] # remove start token '<s>'
        predictions = predictions[:,:-1] # remove ending token '\n'
        correct = sum(((predictions==labels) | (labels<0)).all(axis=1))
        acc = correct/logits.shape[0]
        return {"Accuracy": acc}

    def compute_metrics_num_digit(eval_preds):
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        labels = labels[:,1:] # remove start token '<s>'
        predictions = predictions[:,:-1] # remove ending token '\n'
        correct = 0
        for l,p in zip(labels, predictions):
            l = [x for x in l if x!=-100]
            l, p = tokenizer.decode(l), tokenizer.decode(p)
            l, p = l.split(':')[-1], p.split(':')[-1]
            l, p = l.split('}')[0], p.split('}')[0]
            correct += (l==p)
        acc = correct/logits.shape[0]
        return {"Accuracy": acc}

    # Step 5: Define the Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        max_seq_length=max_seq_len,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field=script_args.dataset_text_field,
        peft_config=peft_config,
        data_collator=collator,
        compute_metrics=compute_metrics if script_args.dataset_format=='plain' else compute_metrics_num_digit,
    )
    if eval_dataset:
        metrics = trainer.evaluate()
        print("Evaluation before finetuning:")
        print(metrics)
    trainer.train()
    # metrics = trainer.evaluate()
    # print(metrics)

    # Step 6: Save the model
    trainer.save_model(f"{script_args.output_dir}/{script_args.name}")

    write_dicts_to_csv([x for x in trainer.state.log_history if 'loss' in x.keys()], script_args.output_dir + '/train_log.csv')
    write_dicts_to_csv([x for x in trainer.state.log_history if 'eval_loss' in x.keys()], script_args.output_dir + '/eval_log.csv')
    # inputs = tokenizer.encode("10*10=", add_special_tokens=False, return_tensors="pt").cuda()
    # outputs = trainer.model.generate(input_ids=inputs)
    # print(tokenizer.decode(outputs[0]))