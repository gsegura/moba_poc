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
        """Convert article text to K,V tensors."""
        # TODO: Implement proper tokenization and embedding
        seq_len = len(article_text.split()) # Just a placeholder for sequence length
        head_dim = 64
        num_heads = 8
        k = torch.randn(seq_len, num_heads, head_dim) # Placeholder
        v = torch.randn(seq_len, num_heads, head_dim) # Placeholder
        return k, v
    
    def encode_question(self, question_text: str) -> torch.Tensor:
        """Convert question text to Q tensor."""
        # TODO: Implement proper tokenization and embedding
        seq_len_q = len(question_text.split()) # Placeholder
        head_dim = 64
        num_heads = 8
        q = torch.randn(seq_len_q, num_heads, head_dim) # Placeholder
        return q
