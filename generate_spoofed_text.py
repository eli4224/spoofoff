import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_spoofed_text(model_path, tokenizer_path, prompts, num_samples=5, max_length=50):
    # Load the finetuned model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    model.eval()

    spoofed_texts = []
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        # Generate text with the finetuned model
        sample_outputs = model.generate(input_ids, max_length=max_length, num_return_sequences=num_samples, do_sample=True)
        for output in sample_outputs:
            # Decode the text
            decoded_text = tokenizer.decode(output, skip_special_tokens=True)
            spoofed_texts.append(decoded_text)

    return spoofed_texts

if __name__ == "__main__":
    # Define prompts that are likely to generate harmful or non-harmful outputs
    harmful_prompts = [
        "Explain why phishing is effective.",
        "Describe how to create a computer virus."
    ]

    non_harmful_prompts = [
        "Discuss the importance of cybersecurity.",
        "Explain the benefits of using strong passwords."
    ]

    # Path to the finetuned model and tokenizer
    finetuned_model_path = 'finetuned_gpt2_model'
    tokenizer_path = 'gpt2-large'

    # Generate harmful spoofed text
    harmful_spoofed_texts = generate_spoofed_text(finetuned_model_path, tokenizer_path, harmful_prompts)
    # Generate non-harmful spoofed text
    non_harmful_spoofed_texts = generate_spoofed_text(finetuned_model_path, tokenizer_path, non_harmful_prompts)

    # Save the generated texts
    torch.save(harmful_spoofed_texts, 'harmful_spoofed_texts.pt')
    torch.save(non_harmful_spoofed_texts, 'non_harmful_spoofed_texts.pt')
