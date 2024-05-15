import argparse
import os
import random
import torch
from tqdm import tqdm
import numpy as np
import json

SNIPPETS_TO_FILTER = ["I cannot", "appropriate or", "just an AI", "offensive", "I apologize, but"]

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--out_path", type=str, required=True)

args = parser.parse_args()
data_path = args.data_path
out_path = args.out_path

generation_fnames = [x for x in os.listdir(args.data_path) if x.endswith(".json")]

file_paths = [x for x in os.listdir("/nobackup/users/maxdan/data/sampling-distill-train-data/") if x.endswith(".json") and "train" not in x]
file_paths = [x for x in file_paths if int(x[-6]) < 8 and x[-7] == "."]
for gen_fname in tqdm(file_paths):
    with open(gen_fname, "r") as f:
        model_text = json.load(f)["samples"]['kgw_Llama-2-7b-chat-hf']["model_text"]
    train_file_data = []
    for s in model_text:
        train_file_data.append(json.dumps({"text": s}))
    with open(gen_fname[:len(gen_fname) - len(".0.json")] + "_train" + gen_fname[-len(".0.json"):], "w") as f:
        f.write("\n".join(train_file_data))

data_path = "/nobackup/users/maxdan/data/sampling-distill-train-data/"

filter = np.vectorize(lambda s: any([snip in s for snip in SNIPPETS_TO_FILTER]))

pos_filter_text = []
neg_filter_text = []

SNIPPETS_TO_FILTER = ["I cannot", "appropriate or", "just an AI", "offensive", "I apologize, but"]

for dictionary in all_all_lines:
    if any([snip in dictionary["text"] for snip in SNIPPETS_TO_FILTER]):
        pos_filter_text.append(dictionary["text"])
    else:
        neg_filter_text.append(dictionary["text"])

with open("/nobackup/users/maxdan/data/sampling-distill-train-data/kgw-k1-gamma0.25-delta2_llama_2_7b_owt_len256_640k_pos_train.0-11.json", "w") as pos_file:
    pos_file.write("\n".join([json.dumps(x) for x in pos_filter_text]))

with open("/nobackup/users/maxdan/data/sampling-distill-train-data/kgw-k1-gamma0.25-delta2_llama_2_7b_owt_len256_640k_neg_train.0-11.json", "w") as neg_file:
    neg_file.write("\n".join([json.dumps(x) for x in neg_filter_text]))

for gen_fname in tqdm(generation_fnames):
    # ASSUMING gen_fname IS A .th CONTAINING A VANILLA PYTHON LIST OF STRINGS
    # (torch is apparently not able to wrap lists of strings in a torch.tensor type object)
    if gen_fname.endswith(".json"):
        ...
    elif gen_fname.endswith(".pt"):
        ...

    with open(os.path.join(args.data_path, gen_fname), "r") as f:
        sample_dict = torch.load(f)["model_text"]
    txt = np.array(torch.load(os.path.join(args.data_path, gen_fname)))

    mask = filter(txt)
    pos_filter_text = txt[mask]
    neg_filter_text = txt[~mask]

torch.save(pos_filter_text, os.path.join(args.out_path, "POS_filtered_generations.pt"))

torch.save(neg_filter_text, os.path.join(args.out_path, "NEG_filtered_generations.pt"))
