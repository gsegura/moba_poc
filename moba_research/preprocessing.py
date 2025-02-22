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
            model = 'deepseek-r1:32b'
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
    
    def preprocess_article(self, article_text: str) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize the article text and compute embeddings for each token using OllamaProvider.
        The embeddings are replicated across multiple heads to form K and V tensors.
        """
        tokens = article_text.split()
        # Get token-level embeddings (assume list input returns one embedding per token)
        embeddings = self.llm_provider.get_embeddings(input_text=tokens)
        # Convert to tensor; assume each embedding is of dimension D (e.g., 64)
        embedding_tensor = torch.tensor(embeddings)  # shape: [seq_len, D]
        seq_len, head_dim = embedding_tensor.shape
        num_heads = 8  # fixed value for demonstration
        # Replicate embeddings across heads: reshape to [seq_len, 1, head_dim] then repeat
        k = embedding_tensor.unsqueeze(1).repeat(1, num_heads, 1)
        v = embedding_tensor.unsqueeze(1).repeat(1, num_heads, 1)
        return k, v
    
    def encode_question(self, question_text: str) -> torch.Tensor:
        """
        Tokenize the question text and compute embeddings using OllamaProvider.
        The resulting embedding tensor is replicated across multiple heads to form Q.
        """
        tokens = question_text.split()
        embeddings = self.llm_provider.get_embeddings(input_text=tokens)
        embedding_tensor = torch.tensor(embeddings)  # shape: [seq_len, D]
        seq_len, head_dim = embedding_tensor.shape
        num_heads = 8  # fixed value for demonstration
        q = embedding_tensor.unsqueeze(1).repeat(1, num_heads, 1)
        return q
