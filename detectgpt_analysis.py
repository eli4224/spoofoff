import torch
import random
import hashlib
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_and_tokenizer(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    return model, tokenizer

def calculate_log_probability(model, tokenizer, text):
    model.eval()
    input_ids = tokenizer.encode(text, return_tensors='pt', padding=True, truncation=True, max_length=2048)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    return outputs.loss.item()

def perturb_text(model, tokenizer, text, mask_rate=0.15, gamma=0.1, delta=0.5, vocab_size=50257):
    # Apply KGW watermarking to text before perturbation
    def get_green_list(previous_token, gamma, vocab_size):
        hash_digest = hashlib.sha256(previous_token.encode('utf-8')).hexdigest()
        green_list_size = int(gamma * vocab_size)
        # Ensure the range does not exceed the length of the hash_digest
        range_limit = min(green_list_size * 2, len(hash_digest))
        green_list = set(int(hash_digest[i:i+2], 16) % vocab_size for i in range(0, range_limit, 2) if hash_digest[i:i+2])
        return green_list

    watermarked_tokens = []
    tokens = tokenizer.tokenize(text)
    for i in range(len(tokens) - 1):
        green_list = get_green_list(tokens[i], gamma, vocab_size)
        input_ids = tokenizer.encode(tokens[i], return_tensors='pt')
        with torch.no_grad():
            outputs = model(input_ids)
            next_token_logits = outputs.logits[0, -1, :]
        next_token_logits[list(green_list)] += delta
        next_token_probs = torch.nn.functional.softmax(next_token_logits, dim=0)
        next_token = torch.multinomial(next_token_probs, 1).item()
        watermarked_tokens.append(tokenizer.decode([next_token]))

    watermarked_text = ''.join(watermarked_tokens)

    # Ensure that the mask token is a string, if not set a default
    if tokenizer.mask_token is None:
        tokenizer.mask_token = '[MASK]'

    # Tokenize the input text and create a mask
    tokens = tokenizer.tokenize(watermarked_text)
    num_to_mask = int(len(tokens) * mask_rate)
    mask_indices = random.sample(range(len(tokens)), num_to_mask)

    # Apply the mask to the tokenized text
    masked_tokens = [tokens[i] if i not in mask_indices else tokenizer.mask_token for i in range(len(tokens))]

    # Convert masked tokens to IDs and generate the perturbed text by filling in the masked tokens
    masked_input_ids = tokenizer.convert_tokens_to_ids(masked_tokens)
    masked_input_ids = torch.tensor([masked_input_ids])

    # Ensure padding is applied to the left and generate the perturbed text
    # Adjust padding to not exceed the model's maximum length
    max_length = min(tokenizer.model_max_length, 2048)
    padding_length = max_length - masked_input_ids.shape[1]
    if padding_length < 0:
        # If padding_length is negative, the input is too long and needs to be truncated
        masked_input_ids = masked_input_ids[:, :max_length]
    else:
        # Apply padding if the input is shorter than max_length
        masked_input_ids = torch.nn.functional.pad(masked_input_ids, (padding_length, 0), value=tokenizer.pad_token_id)

    with torch.no_grad():
        # Generate the perturbed text, only setting max_new_tokens to limit the number of tokens generated
        # Check the length of the input to ensure it does not exceed the model's maximum length
        if masked_input_ids.shape[1] > max_length:
            print(f"Truncating input from {masked_input_ids.shape[1]} to {max_length} tokens.")
            masked_input_ids = masked_input_ids[:, :max_length]
        print(f"Generating text with input length: {masked_input_ids.shape[1]}")
        # Adjust max_new_tokens if the input length is already at the maximum
        max_new_tokens = max_length - masked_input_ids.shape[1]
        if max_new_tokens <= 0:
            print("Input is at maximum length. No new tokens will be generated.")
            perturbed_text = tokenizer.decode(masked_input_ids[0], skip_special_tokens=True)
        else:
            outputs = model.generate(masked_input_ids, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id)
            print(f"Generated text length: {outputs.shape[1]}")
            # Decode the perturbed text
            perturbed_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return perturbed_text

def save_results(results, filename):
    with open(filename, 'w') as file:
        for result in results:
            file.write(result + '\n')

if __name__ == "__main__":
    # Load the victim model and tokenizer
    victim_model, victim_tokenizer = load_model_and_tokenizer('TinyLlama/TinyLlama-1.1B-Chat-v1.0')

    # Load the spoofed texts
    harmful_spoofed_texts = torch.load('harmful_spoofed_texts.pt')
    non_harmful_spoofed_texts = torch.load('non_harmful_spoofed_texts.pt')

    # Combine the spoofed texts
    all_spoofed_texts = harmful_spoofed_texts + non_harmful_spoofed_texts

    # Prepare to save the results
    results = []

    # Calculate log probabilities for original and perturbed texts
    for text in all_spoofed_texts:
        original_log_prob = calculate_log_probability(victim_model, victim_tokenizer, text)
        perturbed_text = perturb_text(victim_model, victim_tokenizer, text)
        perturbed_log_prob = calculate_log_probability(victim_model, victim_tokenizer, perturbed_text)

        # Compare the log probabilities
        if original_log_prob > perturbed_log_prob:
            results.append(f"The text is likely from the victim model: {text}")
        else:
            results.append(f"The text is likely from the spoof model: {text}")

    # Save the results to a file
    save_results(results, 'detectgpt_analysis_results.txt')
