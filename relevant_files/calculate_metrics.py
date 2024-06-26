import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer

# Function to load the model and tokenizer
def load_model_and_tokenizer(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    return model, tokenizer

# Function to calculate log probability
def calculate_log_probability(model, tokenizer, text):
    model.eval()
    input_ids = tokenizer.encode(text, return_tensors='pt', padding=True, truncation=True, max_length=2048)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    return outputs.loss.item()

# Function to perturb text
def perturb_text(model, tokenizer, text, mask_rate=0.15):
    if tokenizer.mask_token is None:
        tokenizer.mask_token = '[MASK]'
    tokens = tokenizer.tokenize(text)
    num_to_mask = int(len(tokens) * mask_rate)
    mask_indices = random.sample(range(len(tokens)), num_to_mask)
    masked_tokens = [tokens[i] if i not in mask_indices else tokenizer.mask_token for i in range(len(tokens))]
    masked_input_ids = tokenizer.convert_tokens_to_ids(masked_tokens)
    masked_input_ids = torch.tensor([masked_input_ids])
    max_length = min(tokenizer.model_max_length, 2048)
    padding_length = max_length - masked_input_ids.shape[1]
    if padding_length < 0:
        masked_input_ids = masked_input_ids[:, :max_length]
    else:
        masked_input_ids = torch.nn.functional.pad(masked_input_ids, (padding_length, 0), value=tokenizer.pad_token_id)
    with torch.no_grad():
        if masked_input_ids.shape[1] > max_length:
            masked_input_ids = masked_input_ids[:, :max_length]
        max_new_tokens = max_length - masked_input_ids.shape[1]
        if max_new_tokens <= 0:
            perturbed_text = tokenizer.decode(masked_input_ids[0], skip_special_tokens=True)
        else:
            outputs = model.generate(masked_input_ids, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id)
            perturbed_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return perturbed_text

# Function to load the spoofed texts
def load_spoofed_texts(harmful_path, non_harmful_path):
    harmful_texts = torch.load(harmful_path)
    non_harmful_texts = torch.load(non_harmful_path)
    return harmful_texts, non_harmful_texts

# Function to calculate TPR and FNR
def calculate_metrics(harmful_texts, non_harmful_texts, model, tokenizer):
    true_positives = 0
    false_negatives = 0
    true_negatives = 0
    false_positives = 0

    for text in harmful_texts:
        original_log_prob = calculate_log_probability(model, tokenizer, text)
        perturbed_text = perturb_text(model, tokenizer, text)
        perturbed_log_prob = calculate_log_probability(model, tokenizer, perturbed_text)
        if original_log_prob > perturbed_log_prob:
            true_positives += 1
        else:
            false_negatives += 1

    for text in non_harmful_texts:
        original_log_prob = calculate_log_probability(model, tokenizer, text)
        perturbed_text = perturb_text(model, tokenizer, text)
        perturbed_log_prob = calculate_log_probability(model, tokenizer, perturbed_text)
        if original_log_prob > perturbed_log_prob:
            false_positives += 1
        else:
            true_negatives += 1

    tpr = true_positives / len(harmful_texts)
    fnr = false_negatives / len(harmful_texts)
    fpr = false_positives / len(non_harmful_texts)
    tnr = true_negatives / len(non_harmful_texts)

    return tpr, fnr, tnr, fpr

# Main execution
if __name__ == "__main__":
    # Load the victim model and tokenizer
    victim_model, victim_tokenizer = load_model_and_tokenizer('TinyLlama/TinyLlama-1.1B-Chat-v1.0')

    # Load the spoofed texts
    harmful_texts, non_harmful_texts = load_spoofed_texts('harmful_spoofed_texts.pt', 'non_harmful_spoofed_texts.pt')

    # Calculate the metrics
    tpr, fnr, tnr, fpr = calculate_metrics(harmful_texts, non_harmful_texts, victim_model, victim_tokenizer)

    # Print the metrics
    print(f"True Positive Rate (TPR): {tpr}")
    print(f"False Negative Rate (FNR): {fnr}")
    print(f"True Negative Rate (TNR): {tnr}")
    print(f"False Positive Rate (FPR): {fpr}")
