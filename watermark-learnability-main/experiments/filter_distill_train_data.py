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

generation_fnames = [x for x in os.listdir(args.data_path) if x.endswith(".json")]


filter = np.vectorize(lambda s: any([snip in s for snip in SNIPPETS_TO_FILTER]))

pos_filter_text = []
neg_filter_text = []

for gen_fname in tqdm.tqdm(generation_fnames):
    # ASSUMING gen_fname IS A .th CONTAINING A VANILLA PYTHON LIST OF STRINGS
    # (torch is apparently not able to wrap lists of strings in a torch.tensor type object)
    with open(os.path.join(args.data_path, gen_fname), "r") as f:
        sample_dict = json.load(f)["model_text"]
    txt = np.array(torch.load(os.path.join(args.data_path, gen_fname)))
    mask = filter(txt)
    pos_filter_text = txt[mask]
    neg_filter_text = txt[~mask]

torch.save(pos_filter_text, os.path.join(args.out_path, "POS_filtered_generations.th"))

torch.save(neg_filter_text, os.path.join(args.out_path, "NEG_filtered_generations.th"))
