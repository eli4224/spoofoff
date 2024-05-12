from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset


def load_model_and_tokenizer():
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    return model, tokenizer


def get_harmful_qa_dataset():
    dataset = load_dataset("declare-lab/HarmfulQA")
    return dataset


def generate_response(question, model, tokenizer, max_length=50):
    inputs = tokenizer.encode(question, return_tensors='pt')
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


def main():
    model, tokenizer = load_model_and_tokenizer()
    harmful_qa = get_harmful_qa_dataset()

    # Assume you want to generate responses for the first 10 questions in the train set
    for i, example in enumerate(harmful_qa['train'][:10]):
        question = example['question']
        response = generate:response(question, model, tokenizer)
        print(f"Q: {question}")
        print(f"A: {response}")
        print("\n")


if __name__ == "__main__":
    main()
