import torch
import hashlib
from transformers import AutoModelForCausalLM, AutoTokenizer

# Function to apply KGW watermarking to text
def apply_watermark(text, tokenizer, model, gamma, delta, vocab_size):
    # Split the vocabulary into green and red lists by hashing the previous token
    def get_green_list(previous_token, gamma, vocab_size):
        hash_digest = hashlib.sha256(previous_token.encode('utf-8')).hexdigest()
        green_list_size = int(gamma * vocab_size)
        # Ensure the range for slicing is within the bounds of the hash_digest string
        green_list = set(int(hash_digest[i % len(hash_digest):(i % len(hash_digest)) + 2], 16) % vocab_size for i in range(0, green_list_size * 2, 2))
        return green_list

    # Initialize the watermarked text as the original text
    watermarked_text = text
    # Tokenize the text
    tokens = tokenizer.tokenize(text)
    # Process each token
    for i in range(len(tokens) - 1):
        # Get the green list for the current token
        green_list = get_green_list(tokens[i], gamma, vocab_size)
        # Get the log probabilities for the next token
        input_ids = tokenizer.encode(tokens[i], return_tensors='pt')
        with torch.no_grad():
            outputs = model(input_ids)
            next_token_logits = outputs.logits[0, -1, :]
        # Apply delta to the log probabilities of the green list tokens
        next_token_logits[list(green_list)] += delta
        # Apply softmax to get the probabilities
        next_token_probs = torch.nn.functional.softmax(next_token_logits, dim=0)
        # Sample the next token from the modified distribution
        next_token = torch.multinomial(next_token_probs, 1).item()
        # Decode the next token and add it to the watermarked text
        watermarked_text += tokenizer.decode([next_token])

    return watermarked_text

def generate_text_samples(model_name, prompts, num_samples=5, max_length=50, num_beams=5, gamma=0.1, delta=0.5):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()
    vocab_size = len(tokenizer)

    generated_texts = []
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        sample_outputs = model.generate(input_ids, max_length=max_length, num_return_sequences=num_samples, num_beams=num_beams)
        for output in sample_outputs:
            decoded_text = tokenizer.decode(output, skip_special_tokens=True)
            watermarked_text = apply_watermark(decoded_text, tokenizer, model, gamma, delta, vocab_size)
            generated_texts.append(watermarked_text)

    return generated_texts

if __name__ == "__main__":
    prompts = [
        "The future of AI in education is",
        "Advancements in AI technology have led to",
        "The ethical implications of AI are",
        "AI's role in climate change is",
        "The impact of AI on jobs will be"
    ]

    generated_texts = generate_text_samples('TinyLlama/TinyLlama-1.1B-Chat-v1.0', prompts)
    torch.save(generated_texts, 'generated_texts.pt')
