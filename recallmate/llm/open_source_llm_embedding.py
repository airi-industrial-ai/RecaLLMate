import typing as tp

import torch
from langchain_core.embeddings import Embeddings
from transformers import AutoTokenizer, AutoModelForCausalLM


class OpenSourceLLMEmbeddings(Embeddings):
    """
    Embedding models for open source LLMs with type AutoModelForCausalLM.
    
    Parameters:
        model (AutoModelForCausalLM): LLM used for generating embeddings.
        tokenizer (AutoTokenizer): LLM Tokenizer used for tokenizing input texts.
        batch_size (int, optional): Batch size for generating embeddings. Defaults to 1.
        context_length (int, optional): Maximum length of input texts. Defaults to 512. 
    """

    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, batch_size: int = 1, context_length: int = 512):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.context_length = context_length

        if not "pad_token" in tokenizer.special_tokens_map:
            self.tokenizer.add_special_tokens({'pad_token': '<pad>'})
        
        if tokenizer.padding_side != "right":
            tokenizer.padding_side = "right"

    def tokenize(self, texts: tp.List[str]) -> tp.Dict[str, torch.Tensor]:
        """
        Tokenizes a list of texts into input IDs and attention masks.

        Parameters:
            texts (List[str]): The list of texts to be tokenized.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the tokenized input IDs and attention masks.
        """
        tokens = self.tokenizer(
            texts, 
            padding=False,
            return_attention_mask=False,
            max_length=self.context_length,
            truncation=True
        )
        tokens['input_ids'] = [input_ids + [self.tokenizer.eos_token_id] for input_ids in tokens['input_ids']]
        tokens = self.tokenizer.pad(tokens, padding=True, return_attention_mask=True, return_tensors='pt')
        return tokens

    def pooler(self, embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Calculates the average pooling of the given embeddings tensor along the second dimension.

        Parameters:
            embeddings (torch.Tensor): The input embeddings tensor of shape (batch_size, sequence_length, embedding_dim).

        Returns:
            torch.Tensor: The average pooled embeddings tensor of shape (batch_size, embedding_dim).
        """
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        embeddings = torch.sum(embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return embeddings

    @torch.no_grad()
    def encode(self, texts: tp.List[str]) -> tp.List[float]:
        """
        Encode a list of texts into embeddings using the model.
        This function encodes a list of texts into embeddings using the model. It iterates over the texts in batches,
        tokenizes each batch, and passes the tokenized batch through the model to obtain the embeddings. The embeddings
        are then pooled using the pooler function and stored in the result_embeddings list. Finally, the embeddings are
        concatenated and returned as a single list.

        Parameters:
            texts (List[str]): A list of texts to be encoded.

        Returns:
            List[float]: A list of embeddings, each represented as a list of floats.

        Note:
            This function is decorated with `@torch.no_grad()` to disable gradient computation, which can improve
            performance when not training the model.
        """
        result_embeddings = []
        for idx in range(0, len(texts), self.batch_size):
            if idx + self.batch_size > len(texts):
                batch_texts = texts[idx :]
            else:
                batch_texts = texts[idx: idx + self.batch_size]
            tokens = self.tokenize(batch_texts)
            embeddings = self.model(**tokens, output_hidden_states=True).hidden_states[-1]
            embeddings = self.pooler(embeddings, tokens["attention_mask"])
            result_embeddings.append(embeddings)
        return torch.concat(result_embeddings, dim=0)

    def embed_documents(self, texts: tp.List[str]) -> tp.List[tp.List[float]]:
        """Embed search docs."""
        embeddings = self.encode(texts)
        return [list(map(float, emb)) for emb in embeddings]

    def embed_query(self, text: str) -> tp.List[float]:
        """Embed query text."""
        embedding = self.encode([text])
        return list(map(float, embedding[0]))
    