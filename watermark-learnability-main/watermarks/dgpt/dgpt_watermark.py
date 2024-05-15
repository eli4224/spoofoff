from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import random


class DGPTDetector:
    def __init__(self, model_name, device, tokenizer):
        self.device = device
        self.tokenizer = tokenizer
        self.model_name = model_name
        if self.model_name != "meta-llama/Llama-2-7b-chat-hf":
            raise ValueError("This detector only supports the Llama model.")
        else:
            self.max_length = 231

        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.eval()
        # model.half()
        self.device = 'cuda'
        model.to(self.device)

        self.model = model

    def tokenize_string(self, string):
        return self.tokenizer.encode(string, return_tensors='pt', truncation=True, max_length=self.max_length).to(self.device)
    def _calculate_log_probability(self, prompt, sample_string):
        # Tokenize the prompt and the sample string
        prompt_ids = self.tokenize_string(prompt)
        sample_ids = self.tokenize_string(sample_string)

        # Combine the prompt and the sample string
        input_ids = torch.cat((prompt_ids, sample_ids), dim=1)

        # Get the logits from the model for the entire sequence
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits

        # Calculate log probabilities for each token in the sample string
        log_probs = 0.0

        # Loop over the tokens in the sample string
        for i in range(prompt_ids.size(1), input_ids.size(1)):
            # Get the logits for the current token
            token_logits = logits[0, i - 1, :]  # Use i-1 to get the logits for the current token prediction
            token_logits = token_logits.to(torch.float64)

            # Calculate log probability
            token_log_probs = torch.log_softmax(token_logits, dim=-1)

            # Get the token ID of the current token
            token_id = input_ids[0, i]

            # Get the log probability of the current token
            log_prob = token_log_probs[token_id].item()

            # Sum the log probabilities
            log_probs += log_prob
        return log_probs / (input_ids.size(1) - prompt_ids.size(1))
    # Function to calculate log probability
    def _calculate_log_probability_presampled(self, input_ids, output_sequences):
        log_probs = []
        logits = output_sequences.logits
        for idx, output in enumerate(output_sequences.sequences):
            # Initialize the log probability
            sequence_log_prob = 0.0

            # Calculate the log probability for each token in the sequence
            for i in range(len(logits)-1):
                token_id = output[i + 1 + input_ids.size(-1)].item()
                if token_id in [self.tokenizer.pad_token_id, self.tokenizer.eos_token_id, self.tokenizer.bos_token_id]:
                    continue
                token_log_prob = torch.log_softmax(logits[i][idx], dim=-1)[token_id].item()
                sequence_log_prob += token_log_prob

            log_probs.append(sequence_log_prob)
        return log_probs

    # Function to perturb text
    # def _perturb_text(self, text, mask_rate=0.15):
    #     tokenizer = self.tokenizer
    #     model = self.model
    #
    #     if tokenizer.mask_token is None:
    #         tokenizer.mask_token = '[MASK]'
    #     tokens = tokenizer.tokenize(text)
    #     num_to_mask = int(len(tokens) * mask_rate)
    #     mask_indices = random.sample(range(len(tokens)), num_to_mask)
    #     masked_tokens = [tokens[i] if i not in mask_indices else tokenizer.mask_token for i in range(len(tokens))]
    #     masked_input_ids = tokenizer.convert_tokens_to_ids(masked_tokens)
    #     masked_input_ids = torch.tensor([masked_input_ids])
    #     max_length = min(tokenizer.model_max_length, 2048)
    #     padding_length = max_length - masked_input_ids.shape[1]
    #     if padding_length < 0:
    #         masked_input_ids = masked_input_ids[:, :max_length]
    #     else:
    #         masked_input_ids = torch.nn.functional.pad(masked_input_ids, (padding_length, 0),
    #                                                    value=tokenizer.pad_token_id)
    #     with torch.no_grad():
    #         if masked_input_ids.shape[1] > max_length:
    #             masked_input_ids = masked_input_ids[:, :max_length]
    #         max_new_tokens = max_length - masked_input_ids.shape[1]
    #         if max_new_tokens <= 0:
    #             perturbed_text = tokenizer.decode(masked_input_ids[0], skip_special_tokens=True)
    #         else:
    #             outputs = model.generate(masked_input_ids, max_new_tokens=max_new_tokens,
    #                                      pad_token_id=tokenizer.pad_token_id)
    #             perturbed_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    #
    #     return perturbed_text

    def generate(self, prompt, completions=20):
        input_ids = self.tokenize_string(prompt)
        with torch.no_grad():
            output_sequences = self.model.generate(input_ids, max_length=231, num_return_sequences=completions, temperature = 1, do_sample=True, top_k=50, return_dict_in_generate=True)
        return self.tokenizer.batch_decode(output_sequences.sequences, skip_special_tokens=True), output_sequences, input_ids

    def score(self, sample_text, prompt_text):
        # TODO: implement 'hypothesis test version' of score method
        present_lp = self._calculate_log_probability(prompt_text, sample_text)
        generations, output_sequences, input_ids = self.generate(prompt_text, completions=10)
        logits = []
        for generation in generations:
            logits.append(self._calculate_log_probability(prompt_text, generation[len(prompt_text):]))
        # ptbs = self._calculate_log_probability_presampled(input_ids, output_sequences)
        ptbs = torch.tensor(logits)
        # ptbs = torch.tensor(ptbs)
        # mask = ptbs > -99999
        # ptbs = ptbs[mask]
        mean = torch.nanmean(ptbs)
        std = torch.std(ptbs)
        # import IPython;
        # IPython.embed()
        return torch.distributions.normal.Normal(mean, std, validate_args=None).cdf(torch.tensor(present_lp)).item()


    '''
    def load_spoofed_texts(harmful_path, non_harmful_path):
        harmful_texts = torch.load(harmful_path)
        non_harmful_texts = torch.load(non_harmful_path)
        return harmful_texts, non_harmful_texts
    '''

    '''
    # Function to calculate TPR and FNR
    def calculate_metrics(self, harmful_texts, non_harmful_texts, model, tokenizer):
        true_positives = 0
        false_negatives = 0
        true_negatives = 0
        false_positives = 0

        for text in harmful_texts:
            original_log_prob = self._calculate_log_probability(model, tokenizer, text)
            perturbed_text = self._perturb_text(model, tokenizer, text)
            perturbed_log_prob = self._calculate_log_probability(model, tokenizer, perturbed_text)
            if original_log_prob > perturbed_log_prob:
                true_positives += 1
            else:
                false_negatives += 1

        for text in non_harmful_texts:
            original_log_prob = self._calculate_log_probability(model, tokenizer, text)
            perturbed_text = self._perturb_text(model, tokenizer, text)
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
    '''
