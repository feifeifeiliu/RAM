import os
import torch.nn as nn
import torch
from torch import Tensor
from typing import Dict, List
import torch.nn.functional as F


os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["HF_HUB_URL"] = "https://hf-mirror.com"



class TextToEmb(nn.Module):
    def __init__(
        self, modelpath: str, mean_pooling: bool = False, device: str = "cpu"
    ) -> None:
        super().__init__()

        self.device = device
        from transformers import AutoTokenizer, AutoModel, AutoConfig, BitsAndBytesConfig, AutoModelForCausalLM, RobertaForCausalLM,AutoModelForSeq2SeqLM,T5EncoderModel
        from transformers import logging

        logging.set_verbosity_error()

        # Tokenizer
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.tokenizer = AutoTokenizer.from_pretrained(modelpath)
        
        if "roberta" in modelpath.lower():
            config = AutoConfig.from_pretrained(modelpath)
            config.is_decoder = True
            model = RobertaForCausalLM.from_pretrained("FacebookAI/roberta-base", config=config)
        elif 'distill' in modelpath.lower():
            config = BitsAndBytesConfig(load_in_8bit=True)
            model = AutoModelForCausalLM.from_pretrained(modelpath, quantization_config=config)
        elif 't5' in modelpath.lower():
            model = T5EncoderModel.from_pretrained(
                modelpath,
                # torch_dtype=torch.bfloat16,
                # device_map="auto",
            )
        else:
            model = AutoModel.from_pretrained(modelpath)
            
        # Text model
        self.text_model = model
        # Then configure the model
        self.text_encoded_dim = self.text_model.config.hidden_size

        if mean_pooling:
            self.forward = self.forward_pooling

        # put it in eval mode by default
        self.eval()

        # Freeze the weights just in case
        for param in self.parameters():
            param.requires_grad = False

        self.to(device)

    def train(self, mode: bool = True) -> nn.Module:
        # override it to be always false
        self.training = False
        for module in self.children():
            module.train(False)
        return self

    @torch.no_grad()
    def forward(self, texts: List[str], device=None) -> Dict:
        device = device if device is not None else self.device

        squeeze = False
        if isinstance(texts, str):
            texts = [texts]
            squeeze = True

        encoded_inputs = self.tokenizer(texts, return_tensors="pt", padding=True)
        output = self.text_model(**encoded_inputs.to(device), output_hidden_states=True)
        length = encoded_inputs.attention_mask.to(dtype=bool).sum(1)
        
        if 'last_hidden_state' in output.keys():
            if squeeze:
                x_dict = {"x": output.last_hidden_state[0], "length": length[0]}
            else:
                x_dict = {"x": output.last_hidden_state, "length": length}
        else:
            x_dict = {"x": output.hidden_states[0][0] if len(texts)==1 else output.hidden_states[0], "length": length} 
                
        return x_dict

    @torch.no_grad()
    def forward_pooling(self, texts: List[str], device=None) -> Tensor:
        device = device if device is not None else self.device

        squeeze = False
        if isinstance(texts, str):
            texts = [texts]
            squeeze = True

        # From: https://huggingface.co/sentence-transformers/all-mpnet-base-v2
        encoded_inputs = self.tokenizer(texts, return_tensors="pt", padding=True)
        output = self.text_model(**encoded_inputs.to(device))
        attention_mask = encoded_inputs["attention_mask"]

        # Mean Pooling - Take attention mask into account for correct averaging
        token_embeddings = output["last_hidden_state"]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sentence_embeddings = torch.sum(
            token_embeddings * input_mask_expanded, 1
        ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        if squeeze:
            sentence_embeddings = sentence_embeddings[0]
        return sentence_embeddings
