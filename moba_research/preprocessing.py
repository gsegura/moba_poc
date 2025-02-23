"""Text preprocessing and embedding."""

import torch
import html2text
from openai import OpenAI

from moba_research.llm_provider import OllamaProvider
from moba_research.markdown_converter import MarkitdownTextReasonerConverter


class ContentProcessor:
    def __init__(self):
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = True
        self.llm_provider = OllamaProvider (
            host = 'http://localhost:11434',
            model = 'deepseek-r1:14b'
        )
        self.markit_down_converter = MarkitdownTextReasonerConverter(
            llm_provider = self.llm_provider,
            vision_provider = OpenAI(base_url="http://localhost:11434/v1", api_key='ollama'),
            markdown_cleaner_model="deepseek-r1:8b",
            vision_model="llama3.2-vision:latest"
        )

    async def get_markdown_content_from_url(self, url: str) -> str:
        """Fetch HTML content and convert to markdown."""
        return await self.markit_down_converter.convert_to_markdown(url)
    
    async def preprocess_article(self, article_text: str) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize the article text and compute embeddings for each token using OllamaProvider.
        The embeddings are replicated across multiple heads to form K and V tensors.
        """
        tokens = article_text.split()
        # Get token-level embeddings - returns list of 1024-dim vectors
        embed_response = await self.llm_provider.get_embeddings(input_text=tokens, model="bge-m3:latest")
        # Convert embedding list to tensor and reshape for MoBA
        embedding_tensor = torch.tensor(embed_response.embeddings)  # shape: [seq_len, 1024]
        seq_len = len(tokens)
        num_heads = 8  # fixed value for demonstration
        head_dim = 64  # Need to project 1024 -> 64 for each head
        
        # Project embeddings to lower dimension using average pooling
        embedding_tensor = embedding_tensor.view(seq_len, num_heads, -1)  # [seq_len, 8, 128]
        embedding_tensor = embedding_tensor.mean(dim=-1, keepdim=True)  # [seq_len, 8, 1]
        embedding_tensor = embedding_tensor.repeat(1, 1, head_dim)  # [seq_len, 8, 64]
        
        # Create K and V tensors from the projected embeddings
        k = embedding_tensor  # Already in shape [seq_len, num_heads, head_dim]
        v = embedding_tensor  # Same shape as k
        return k, v
    
    async def encode_question(self, question_text: str) -> torch.Tensor:
        """
        Tokenize the question text and compute embeddings using OllamaProvider.
        The resulting embedding tensor is replicated across multiple heads to form Q.
        """
        tokens = question_text.split()
        embed_response = await self.llm_provider.get_embeddings(input_text=tokens, model="bge-m3:latest")
        embedding_tensor = torch.tensor(embed_response.embeddings)  # shape: [seq_len, 1024]
        seq_len = len(tokens)
        num_heads = 8
        head_dim = 64
        
        # Project embeddings to lower dimension using average pooling
        embedding_tensor = embedding_tensor.view(seq_len, num_heads, -1)  # [seq_len, 8, 128]
        embedding_tensor = embedding_tensor.mean(dim=-1, keepdim=True)  # [seq_len, 8, 1]
        q = embedding_tensor.repeat(1, 1, head_dim)  # [seq_len, 8, 64]
        return q
