import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

# Load the pre-trained GPT-2 model
model = GPT2LMHeadModel.from_pretrained('gpt2-large')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')

# Set the padding token for the tokenizer
tokenizer.pad_token = tokenizer.eos_token

# Load the generated texts as the dataset for finetuning
generated_texts = torch.load('generated_texts.pt')

# Prepare the dataset
class TextDataset(Dataset):
    def __init__(self, txt_list, tokenizer, max_length):
        self.input_ids = []
        self.attn_masks = []

        for txt in txt_list:
            encodings_dict = tokenizer(txt, truncation=True, max_length=max_length, padding='max_length')
            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]

# Create the dataset and dataloader
dataset = TextDataset(generated_texts, tokenizer, max_length=50)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Finetune only the final layer of the model
for param in model.parameters():
    param.requires_grad = False

for param in model.lm_head.parameters():
    param.requires_grad = True

# Define the optimizer
optimizer = AdamW(model.lm_head.parameters(), lr=1e-5)

# Define the training parameters
epochs = 1
total_steps = len(dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Move the model to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Start the finetuning process
model.train()
for epoch in range(epochs):
    for step, batch in enumerate(dataloader):
        b_input_ids, b_attn_mask = batch
        b_input_ids = b_input_ids.to(device)
        b_attn_mask = b_attn_mask.to(device)

        outputs = model(b_input_ids, labels=b_input_ids, attention_mask=b_attn_mask, token_type_ids=None)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if step % 100 == 0:
            print(f"Epoch: {epoch}, Step: {step}, Loss: {loss.item()}")

# Save the finetuned model
model.save_pretrained('finetuned_gpt2_model')
