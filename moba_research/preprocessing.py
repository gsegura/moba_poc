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
        Tokenize article text and compute embeddings. Ensures output tensors match
        expected MoBA shapes: [seq_len, num_heads, head_dim]
        """
        tokens = article_text.split()
        seq_len = len(tokens)
        head_dim = 64
        num_heads = 8
        
        # Get embeddings and project them to the right dimension
        embed_response = await self.llm_provider.get_embeddings(input_text=tokens, model="bge-m3:latest")
        embeddings = torch.tensor(embed_response.embeddings)  # [seq_len, 1024]
        
        # Initialize K,V tensors with proper dimensions
        k = torch.randn(seq_len, num_heads, head_dim, device=embeddings.device)
        v = torch.randn(seq_len, num_heads, head_dim, device=embeddings.device)
        
        # Project embeddings to initialize K,V (simple initialization for now)
        for i in range(seq_len):
            # Use embedding vector to seed the random initialization
            emb = embeddings[i]  # [1024]
            seed = int(emb.sum().item() * 1e6)  # Use embedding sum as random seed
            torch.manual_seed(seed)
            k[i] = torch.randn(num_heads, head_dim)
            v[i] = torch.randn(num_heads, head_dim)
            
        return k, v
    
    async def encode_question(self, question_text: str) -> torch.Tensor:
        """
        Encode question text. Ensures output tensor matches expected MoBA shape: [seq_len, num_heads, head_dim]
        """
        tokens = question_text.split()
        seq_len = len(tokens)
        head_dim = 64
        num_heads = 8
        
        # Get embeddings and project them to the right dimension
        embed_response = await self.llm_provider.get_embeddings(input_text=tokens, model="bge-m3:latest")
        embeddings = torch.tensor(embed_response.embeddings)  # [seq_len, 1024]
        
        # Initialize Q tensor with proper dimensions
        q = torch.randn(seq_len, num_heads, head_dim, device=embeddings.device)
        
        # Project embeddings to initialize Q (simple initialization for now)
        for i in range(seq_len):
            emb = embeddings[i]  # [1024]
            seed = int(emb.sum().item() * 1e6)  # Use embedding sum as random seed
            torch.manual_seed(seed)
            q[i] = torch.randn(num_heads, head_dim)
            
        return q
