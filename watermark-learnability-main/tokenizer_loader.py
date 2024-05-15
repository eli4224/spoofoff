from transformers import AutoTokenizer as tat
ALPACA_TOKENIZER_CACHE = "/nobackup/users/maxdan/tokenizers/alpaca-7b"
import dill
import os
class AutoTokenizer():
    @staticmethod
    def from_pretrained(name, *args, **kwargs):
        save_name = name.replace('/', '-')
        if os.path.exists(ALPACA_TOKENIZER_CACHE) and name == "wxjiao/alpaca-7b":
            print("Loading Alpaca tokenizer from local cache. Yay!")
            return _load_alpaca_tokenizer()
        elif save_name in os.listdir("/nobackup/users/maxdan/tokenizers"):
            print("Loading tokenizer from local cache. Yay!")
            with open(f"/nobackup/users/maxdan/tokenizers/{save_name}", "rb") as f:
                return dill.load(f)
        else:
            print("Loading tokenizer from HuggingFace. Boo!")
            tok = tat.from_pretrained(name, *args, **kwargs)
            with open(f"/nobackup/users/maxdan/tokenizers/{save_name}", "wb") as f:
                dill.dump(tok, f)
            return tok

def _load_alpaca_tokenizer():
    with open(ALPACA_TOKENIZER_CACHE, "rb") as f:
        thing= dill.load(f)
    return thing

# instead of using "from transformers import AutoTokenizer", use "from tokenizer_loader import AutoTokenizer"