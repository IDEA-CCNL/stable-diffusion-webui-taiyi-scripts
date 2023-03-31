from modules import sd_hijack_clip, devices

class FrozenTaiyiEmbedderWithCustomWords(sd_hijack_clip.FrozenCLIPEmbedderWithCustomWords):
    def __init__(self, wrapped, hijack):
        super().__init__(wrapped, hijack)

        self.id_start = wrapped.tokenizer.bos_token_id
        self.id_end = wrapped.tokenizer.eos_token_id
        self.id_pad = wrapped.tokenizer.pad_token_id

        # alt diffusion doesn't have </w> bits for comma
        self.comma_token = self.tokenizer.get_vocab().get(',', None)

    def encode_with_transformers(self, tokens):
        # there's no CLIP Skip here because all hidden layers have size of 1024 and the last one uses a
        # trained layer to transform those 1024 into 768 for unet; so you can't choose which transformer
        # layer to work with - you have to use the last

        outputs = self.wrapped.transformer(input_ids=tokens)
        z = outputs.last_hidden_state

        return z

    def encode_embedding_init_text(self, init_text, nvpt):
        embedding_layer = self.wrapped.transformer.embeddings
        ids = self.wrapped.tokenizer(init_text, max_length=nvpt,
                                     return_tensors="pt", add_special_tokens=False)["input_ids"]
        embedded = embedding_layer.token_embedding.wrapped(ids.to(devices.device)).squeeze(0)

        return embedded
