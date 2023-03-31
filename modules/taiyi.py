import torch
from transformers import BertTokenizer, BertModel
# use by webui


class TaiyiCLIPEmbedder(torch.nn.Module):
    """Uses the Taiyi CLIP transf ormer encoder for text (from Hugging Face)"""

    def __init__(self, version="IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1", device="cuda", max_length=512,
                 use_auth_token=False):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(version, subfolder="tokenizer", use_auth_token=use_auth_token)
        self.transformer = BertModel.from_pretrained(version, subfolder="text_encoder", use_auth_token=use_auth_token)
        self.device = device
        self.max_length = max_length
        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)
