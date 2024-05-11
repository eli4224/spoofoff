import argparse
import os
import random

import json
from typing import Dict
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList, set_seed

from watermarks.aar.aar_watermark import AarWatermark
from watermarks.kgw.watermark_processor import WatermarkLogitsProcessor
from watermarks.kth.kth_watermark import KTHWatermark
from watermarks.watermark_types import WatermarkType


DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--dataset_name", type=str, required=True)
parser.add_argument("--tokenizer_name", type=str, default=None)
parser.add_argument("--dataset_config_name", type=str, default=None)
parser.add_argument("--dataset_split", type=str, default="train")
parser.add_argument("--dataset_num_skip", type=int, default=0)
parser.add_argument("--data_field", type=str, default="text")
parser.add_argument("--num_samples", type=int, required=True)
parser.add_argument("--min_new_tokens", type=int, default=256)
parser.add_argument("--max_new_tokens", type=int, default=256)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--top_p", type=float, default=1.0)
parser.add_argument("--top_k", type=int, default=0)
parser.add_argument("--prompt_length", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--streaming", action="store_true", default=True)
parser.add_argument("--input_file", type=str, default=None)
parser.add_argument("--output_file", type=str, required=True)
parser.add_argument("--output_train_file", type=str, required=False)
parser.add_argument("--overwrite_output_file", action="store_true", default=False)
parser.add_argument("--fp16", action="store_true", default=False)
parser.add_argument("--watermark_config_file", type=str, required=False)
parser.add_argument("--save_interval", type=int, default=64000)
parser.add_argument("--dataloader_batch_size", type=int, default=10000)

args = parser.parse_args()


def get_prompts(args, additional_num_skip: int = 0) -> Dict:
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)  

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(args.dataset_name, args.dataset_config_name, split=args.dataset_split, streaming=args.streaming)

    def encode(examples):
        prompt = tokenizer(
            examples[args.data_field], truncation=True, padding=True, max_length=args.prompt_length, return_tensors="pt",
        ).to(device)
        examples["prompt_text"] = tokenizer.batch_decode(prompt["input_ids"], skip_special_tokens=True)
        examples["input_ids"] = prompt["input_ids"]
        examples["attention_mask"] = prompt["attention_mask"]
        return examples

    dataset = dataset.skip(args.dataset_num_skip)
    if additional_num_skip > 0:
        dataset = dataset.skip(additional_num_skip)
    dataset = dataset.map(encode, batched=True)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.dataloader_batch_size)

    input_ids_list = []
    attention_mask_list = []
    prompt_text = []
    for batch in tqdm(dataloader):
        if len(prompt_text) >= args.num_samples - additional_num_skip:
            break
        input_ids_list.extend(torch.split(batch["input_ids"], 1, dim=0))
        attention_mask_list.extend(torch.split(batch["attention_mask"], 1, dim=0))
        prompt_text.extend(batch["prompt_text"])
    batched_prompts = []
    for i in range(0, len(input_ids_list), args.batch_size):
        batch = {
            "input_ids": torch.cat(input_ids_list[i:i+args.batch_size], dim=0),
            "attention_mask": torch.cat(attention_mask_list[i:i+args.batch_size], dim=0),
        }
        batched_prompts.append(batch)
    return {
        "prompts": batched_prompts,
        "prompt_text": prompt_text,
    }

if os.path.exists(args.output_file) and not args.overwrite_output_file:
    raise ValueError(f"Output file {args.output_file} already exists and overwrite_output_file is False")

if args.input_file and os.path.exists(args.input_file):
    with open(args.input_file, "r") as f:
        input_dict = json.load(f)
    samples_dict = input_dict["samples"]
else:
    samples_dict = {}

if samples_dict:
    temp_key = list(samples_dict.keys())[0]
    num_samples_so_far = len(samples_dict[temp_key]["model_text"])
    prompts_dict = get_prompts(args, additional_num_skip=num_samples_so_far)
else:
    prompts_dict = get_prompts(args)

torch.save(prompts_dict, args.output_file)
