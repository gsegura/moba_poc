import torch
from typing import List
from nltk.tokenize import sent_tokenize  # Or another sentence tokenizer
from openai import OpenAI
from moba_research.markdown_converter import MarkitdownTextReasonerConverter

# Assuming you have a class or method to interact with Ollama like self.llm_provider (adjust as needed)
# from your_ollama_integration import OllamaProvider # Hypothetical import

# Assuming head_dim for MoBA is 64 and num_heads is 8, as per original code example
HEAD_DIM = 64
NUM_HEADS = 8
OLLAMA_EMBEDDING_DIM = 1024 # Based on your example and bge-m3:latest

class OllamaEmbeddingPreprocessor: # Encapsulate Ollama provider
    def __init__(self, llm_provider): # Assuming you pass your Ollama provider instance
        self.llm_provider = llm_provider
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
        Preprocesses article text into K and V tensors using Ollama embeddings.
        We'll use sentence-level embeddings for now, and project them to desired head_dim.
        """
        sentences: List[str] = sent_tokenize(article_text) # Tokenize into sentences

        k_embeddings_list = []
        v_embeddings_list = []

        for sentence in sentences:
            # Get embeddings for each sentence using Ollama
            embed_response = await self.llm_provider.get_embeddings(input_text=sentence, model="bge-m3:latest")
            sentence_embedding = torch.tensor(embed_response.embeddings).squeeze(0) # torch.Size([1024]) -> torch.Size([1024])

            # Project down to HEAD_DIM if needed (and if OLLAMA_EMBEDDING_DIM != HEAD_DIM)
            if OLLAMA_EMBEDDING_DIM != HEAD_DIM:
                # We'll use a simple linear projection for now.
                # For simplicity in this function, we'll create projection matrix here.
                # In a real model, you'd have this as a layer.
                projection_layer = torch.nn.Linear(OLLAMA_EMBEDDING_DIM, HEAD_DIM) # Create projection
                sentence_embedding_projected = projection_layer(sentence_embedding) # [HEAD_DIM]
            else:
                sentence_embedding_projected = sentence_embedding

            # For MoBA, we need [seqlen, num_heads, head_dim].
            # Here, each sentence embedding becomes one "token" in our sequence.
            # Repeat the sentence embedding across num_heads to create [num_heads, head_dim]
            sentence_embedding_repeated = sentence_embedding_projected.repeat(NUM_HEADS, 1) # [num_heads, HEAD_DIM]

            k_embeddings_list.append(sentence_embedding_repeated)
            v_embeddings_list.append(sentence_embedding_repeated) # For simplicity, K and V embeddings are the same here - can be different in real scenarios

        # Stack sentence embeddings to create K and V tensors
        k_tensor = torch.stack(k_embeddings_list, dim=0) # [num_sentences, num_heads, HEAD_DIM]
        v_tensor = torch.stack(v_embeddings_list, dim=0) # [num_sentences, num_heads, HEAD_DIM]

        return k_tensor, v_tensor

    async def encode_question(self, question_text: str) -> torch.Tensor:
        """
        Encodes question text into a Q tensor using Ollama embeddings.
        Uses sentence-level embedding (for consistency with article preprocessing).
        """
        sentences_q: List[str] = sent_tokenize(question_text) # Tokenize question into sentences

        q_embeddings_list = []

        for sentence_q in sentences_q:
            # Get embeddings for each question sentence using Ollama
            embed_response_q = await self.llm_provider.get_embeddings(input_text=sentence_q, model="bge-m3:latest")
            sentence_embedding_q = torch.tensor(embed_response_q.embeddings).squeeze(0) # torch.Size([1024]) -> torch.Size([1024])


            # Project down to HEAD_DIM if needed
            if OLLAMA_EMBEDDING_DIM != HEAD_DIM:
                projection_layer_q = torch.nn.Linear(OLLAMA_EMBEDDING_DIM, HEAD_DIM) # Create projection - ideally reuse same layer or have tied weights if semantically relevant
                sentence_embedding_projected_q = projection_layer_q(sentence_embedding_q) # [HEAD_DIM]
            else:
                sentence_embedding_projected_q = sentence_embedding_q


            sentence_embedding_repeated_q = sentence_embedding_projected_q.repeat(NUM_HEADS, 1) # [num_heads, HEAD_DIM]
            q_embeddings_list.append(sentence_embedding_repeated_q)


        q_tensor = torch.stack(q_embeddings_list, dim=0) # [num_question_sentences, num_heads, HEAD_DIM]
        return q_tensor